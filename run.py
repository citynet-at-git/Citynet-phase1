import numpy as np
import argparse
from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import logging


parser = argparse.ArgumentParser()

parser.add_argument('-p','--path_prefix',default='../')
parser.add_argument('-n','--city_name',default='beijing')
parser.add_argument('-y','--output_length',default=1)
parser.add_argument('-s','--service',default='inflow')
parser.add_argument('-x','--input_lag_mode',default=1)
parser.add_argument('-m','--model',default='HA')
parser.add_argument('-l', '--lr', type = float, default = 0.0002)
parser.add_argument('-e', '--epochs', type = int, default = 100)
parser.add_argument('-w', '--weather', type = int, default = 0, help = "Whether to use weather data. ")
parser.add_argument('-r', '--regularization', type = float, default = 0, help = "scale of weight decay")
parser.add_argument('-g', '--gpu', type = int, default = 7, help = 'the gpu to use')
parser.add_argument('-b', '--batch_size', type = int, default = 16, help = "batch size")
parser.add_argument('-t', '--transfer_data', type = int, default = 0, help = "whether to use designated data (for transfer).")

#load all args
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
gpu_available = torch.cuda.is_available()
if gpu_available:
    gpu = torch.device("cuda:0")

pref = args.path_prefix # prefix of file paths ~/citynet-phase1-data
cname = args.city_name  # bj or sh or ...
short_cname = {
    "beijing":"bj",
    "chengdu":"cd",
    "shanghai":"sh",
    "shenzhen":"sz",
    "chongqing":"cq",
    "xian":"xa",
}[cname]
ylength = args.output_length # temporal length of prediction output
service = args.service #service name -- demand/inflow/... or all
lag_mode = args.input_lag_mode #see below definition of lag
model_name = args.model # 'HA', 'LR', 'ARIMA', 'CNN', 'CONVLSTM', 'GCN', 'GAT'
patience = 15

transfer_data_range_week = {
    "beijing":(0, 336), 
    "shanghai":(624, 960), 
    "shenzhen":(576, 912),
    "chongqing":(528, 864),
    "chengdu":(1632, 1968), 
    "xian":(1632, 1968)
}
transfer_data_range_3day = {
    "beijing":(192, 336), 
    "shanghai":(816, 960), 
    "shenzhen":(768, 912),
    "chongqing":(720, 864),
    "chengdu":(1824, 1968), 
    "xian":(1824, 1968)
}
print(args)

def min_max_normalize(data,cut_off_percentile=0.99):
    sl = sorted(data.flatten())
    max_val = sl[int(len(sl)*cut_off_percentile)]
    min_val = max(0,sl[0])
    data[data>max_val]=max_val
    data[data<min_val]=min_val
    data-=min_val
    data/=(max_val-min_val)
    return data,max_val,min_val

def split(data,lags,temporal_mask,portion=[0.7,0.15,0.15]):
    assert(sum(portion)==1.0)
    x = []
    y = []
    for i in range(-lags[0],len(temporal_mask)-ylength):
        x_idx = list(map(lambda x:x+i,lags))
        y_idx = [i+o for o in range(0,ylength)]
        if (temporal_mask[x_idx+y_idx]==0).sum() == 0:
            x.append(data[x_idx])
            y.append(data[y_idx])
    x = np.stack(x,0)
    y = np.stack(y,0)
    trainx = np.array(x[:int(portion[0]*len(x))])
    trainy = np.array(y[:int(portion[0]*len(y))])
    valx = np.array(x[int(portion[0]*len(x)):int((portion[0]+portion[1])*len(x))])
    valy = np.array(y[int(portion[0]*len(y)):int((portion[0]+portion[1])*len(y))])
    testx = np.array(x[int((portion[0]+portion[1])*len(x)):])
    testy = np.array(y[int((portion[0]+portion[1])*len(x)):])
    return trainx,trainy,valx,valy,testx,testy

# loss functions
def masked_mae(T,P,mask,preserve_f_dim=False):
    mask = np.expand_dims(mask,(0,-1))
    mask = np.expand_dims(mask,(0))
    mask = np.repeat(np.repeat(np.repeat(mask,T.shape[0],0),T.shape[-1],-1),T.shape[1],1)
    if(preserve_f_dim):
        return (abs(T-P)*mask).sum((0,1,2,3))/(mask.sum()/T.shape[-1])
    else:
        return (abs(T-P)*mask).sum()/mask.sum()
        

def masked_rmse(T,P,mask,preserve_f_dim=False):
    mask = np.expand_dims(mask,(0,-1))
    mask = np.expand_dims(mask,(0))
    mask = np.repeat(np.repeat(np.repeat(mask,T.shape[0],0),T.shape[-1],-1),T.shape[1],1)
    if(preserve_f_dim):
        return np.sqrt((((T-P)**2)*mask).sum((0,1,2,3))/(mask.sum()/T.shape[-1]))
    else:
        return np.sqrt((((T-P)**2)*mask).sum()/mask.sum())

def mae(T,P,preserve_f_dim=False):
    if(preserve_f_dim):
        return (abs(T-P).sum((0,1,2)))/(T.shape[0]*T.shape[1]*T.shape[2])
    else:
        return abs(T-P).mean()


def rmse(T,P,preserve_f_dim=False):
    if(preserve_f_dim):
        return np.sqrt(((T-P)**2).sum((0,1,2))/(T.shape[0]*T.shape[1]*T.shape[2]))
    else:
        return np.sqrt(((T-P)**2).mean())
    
def idx_2d_to_1d(from_coord,from_shape):
    return from_shape[1]*from_coord[0]+from_coord[1]

def gen_adjacency_A(shape):
    lng = shape[0]
    lat = shape[1]
    A = np.zeros((lng*lat,lng*lat))
    for a in range(0,lng):
        for b in range(0,lat):
            center = idx_2d_to_1d((a,b),shape)
            # A[center,center]=1
            if(a<lng-1):
                p1 = idx_2d_to_1d((a+1,b),shape)
                A[center,p1]=1
                A[p1,center]=1
            if(a>0):
                p2 = idx_2d_to_1d((a-1,b),shape)
                A[center,p2]=1
                A[p2,center]=1
            if(b<lat-1):
                p3 = idx_2d_to_1d((a,b+1),shape)
                A[center,p3]=1
                A[p3,center]=1
            if(b>0):
                p4 = idx_2d_to_1d((a,b-1),shape)
                A[center,p4]=1
                A[p4,center]=1
    return A

def mae_pytorch(T,P):
    return torch.abs(T-P).mean()
def rmse_pytorch(T,P):
    return torch.sqrt(((T-P)**2).mean())

def gen_poi_A(poi,density=0.01):
    # density controls literally the density of the adjacency matrix. 
    affn = np.matmul(poi,poi.transpose())
    s = sorted(affn.flatten())
    threshold = s[int(len(s)*(1-density))]
    return (affn>threshold)


print('==========loading data files============')
#load mask
mask = np.load(pref + "%s/mask_%s.npy"%(cname, short_cname))
temporal_mask = (mask.mean((1,2))>0) # True-False array with shape [T] 
spatial_mask = (mask.mean(0)>0) # True-False array with shape [lng, lat]
print('temporal mask density:',temporal_mask.sum()/len(temporal_mask))
print('spatial mask density:',spatial_mask.sum()/len(spatial_mask.flatten()))



#load data and minmax normalize    

poi_raw = np.load(pref+'poi/poi_vectors_'+short_cname+'.npy') #(lng,lat,14)
lng = poi_raw.shape[0]
lat = poi_raw.shape[1]
poi_raw = np.reshape(poi_raw,(lng*lat,poi_raw.shape[2])) #(lng*lat,14)
spatial_maskf = spatial_mask.reshape(-1)
poi_mask = poi_raw[spatial_maskf,:] #(masked_n,14)
poi,poi_max,poi_min = min_max_normalize(poi_mask,0.9) #(masked_n,14), int, int
weather_hour,weather_max,weather_min = min_max_normalize(np.load(pref+'weather/weather_'+short_cname+'.npy')) # one-hour
weather = np.stack([weather_hour,weather_hour],axis=1).reshape(weather_hour.shape[0]*2,weather_hour.shape[1]) #(time,49)






# all loaded data in (Sample,x/y_length(temporal),lng,lat,feature) format -- by split() function
# for single tasks, feature = 1
# for all tasks, feature = 4 (stacked)

if(service == 'inflow'):
    data = np.expand_dims(np.load(pref + '/%s/inflow_arr_%s.npy' % (cname, short_cname)),-1) #30min
    data,data_max,data_min = min_max_normalize(data)
if(service == 'outflow'):
    data = np.expand_dims(np.load(pref + '%s/outflow_arr_%s.npy' % (cname, short_cname)),-1) #30min
    data,data_max,data_min = min_max_normalize(data)
if(service == 'demand'):
    demand_10min = np.load(pref + "/%s/%sdemand.npy" % (cname, short_cname)) #10min
    data = np.expand_dims(np.array([demand_10min[i:i+3].sum(0) for i in range(0,demand_10min.shape[0],3)]),-1)
    data,data_max,data_min = min_max_normalize(data)
if(service == 'supply'):
    supply_10min = np.load(pref + "/%s/%ssupply.npy" % (cname, short_cname)) #10min
    data = np.expand_dims(np.array([supply_10min[i:i+3].mean(0) for i in range(0,supply_10min.shape[0],3)]),-1)
    data,data_max,data_min = min_max_normalize(data)    
if(service == 'all'):
    inflow,inflow_max,inflow_min = min_max_normalize(np.load(pref+'%s/inflow_arr_%s.npy'%(cname, short_cname)))
    outflow,outflow_max,outflow_min = min_max_normalize(np.load(pref+'%s/outflow_arr_%s.npy'%(cname, short_cname)))
    demand_10min= np.load(pref+'/%s/%sdemand.npy'%(cname, short_cname))
    demand,demand_max,demand_min = min_max_normalize(np.array([demand_10min[i:i+3].sum(0) for i in range(0,demand_10min.shape[0],3)]))  
    if cname not in ['chengdu','xian']:
        supply_10min = np.load(pref+'/%s/%ssupply.npy'%(cname, short_cname))
        supply,supply_max,supply_min = min_max_normalize(np.array([supply_10min[i:i+3].mean(0) for i in range(0,supply_10min.shape[0],3)]))
        data = np.stack([inflow,outflow,demand,supply],axis=-1)
        data_max = np.array([inflow_max,outflow_max,demand_max,supply_max])
        data_min = np.array([inflow_min,outflow_min,demand_min,supply_min])
    else:
        data = np.stack([inflow, outflow, demand], axis = -1)
        data_max = np.array([inflow_max, outflow_max, demand_max])
        data_min = np.array([inflow_min, outflow_min, demand_min])
print('data loaded:',service,'-',data.shape)
# At this time, data.shape = [T, lng, lat, Task]
# Task = 1 if single task, else 4/3

if args.weather:
    print('weather data loaded:',weather.shape)

        
lng = data.shape[1]
lat = data.shape[2]

if(model_name[0]=='G'):
    data = np.reshape(data,(data.shape[0],lat*lng,data.shape[-1])) # [, lng * lat, ]
    spatial_maskf = spatial_mask.reshape(-1)
    data = data[:,spatial_maskf,:]        
    # leave only valid points
    # now, data is in the shape of [?, # valid_points, feat]

    A_euc = gen_adjacency_A((lng,lat))
    A_euc = A_euc[spatial_maskf,:][:,spatial_maskf]
    road_raw = (np.load(pref+'road/'+short_cname+'_conn.npy')>0) #(lng*lat,lng*lat)
    A_conn = road_raw[spatial_maskf,:][:,spatial_maskf] #(masked_n,masked_n)
    A_conn = A_conn.astype(float)
    A_conn *= (1 - A_euc)
    # Only those that are not euclideanly neighboring are considered for a connection.     
    A_poi = gen_poi_A(poi)
    
    print('==========preprocessing data============')
    print('adjacency matrices calculated: \n Euclidean adj: shape=',A_euc.shape,'; density = ',A_euc.sum()/(A_euc.shape[0]*A_euc.shape[1]),'\n POI adj: shape=',A_poi.shape,'; density = ',A_poi.sum()/(A_euc.shape[0]*A_euc.shape[1]),'\n road adj: shape=',A_conn.shape,'; density = ',A_conn.sum()/(A_euc.shape[0]*A_euc.shape[1]),)

#history lags, with 30min temporal sampling interval
no_lag = [-1]
hour_lag = [-5,-4,-3,-2,-1]
one_day_lag = [-48,-4,-3,-2,-1]

if(int(lag_mode)==0):
    lag=no_lag
if(int(lag_mode)==1):
    lag=hour_lag
if(int(lag_mode)==2):
    lag=one_day_lag
    
#split train-val-test dataset
train_weather,_,val_weather,_,test_weather,_ = split(weather,lag,temporal_mask)
train_x,train_y,val_x,val_y,test_x,test_y = split(data,lag,temporal_mask)
# train_x.shape: [?, lag, lng, lat, feat]
# train_y.shape: [?, 1, lng, lat, feat] (if we assume that ylength = 1)
# if graph models: 
# train_x.shape: [?, lag, # valid_points, feat]
# train_y.shape: [?, 1, # valid_points, feat]
if args.transfer_data != 0:
    if args.transfer_data == 7:
        transfer_data_range = transfer_data_range_week
    elif args.transfer_data == 3:
        transfer_data_range = transfer_data_range_3day
    # we will use designated ranges of data
    idx_start = transfer_data_range[cname][0]
    idx_end = transfer_data_range[cname][1]
    train_data = data[idx_start:idx_end]
    train_weather = weather[idx_start:idx_end]
    # process them to standard format
    # [B, lag, lng, lat, feat]
    x = []
    y = []
    w = []
    for i in range(-lag[0], len(train_data) - ylength+1):
        x_idx = list(map(lambda x:x+i, lag))
        y_idx = [i+o for o in range(ylength)]
        
        biased_idxs = np.array(x_idx + y_idx) + idx_start
        if (temporal_mask[biased_idxs] == 0).sum() == 0:
            x.append(train_data[x_idx])
            y.append(train_data[y_idx])
            w.append(train_weather[x_idx])
    x = np.stack(x, 0)
    y = np.stack(y, 0)
    w = np.stack(w, 0)
    train_x = x
    train_y = y
    train_weather = w
print('split to: train_x-%s, train_y-%s, val_x-%s, val_y-%s, test_x-%s, test_y-%s'%(train_x.shape,train_y.shape,val_x.shape,val_y.shape,test_x.shape,test_y.shape))


if(model_name in ['HA','LR','ARIMA']): 
    if(model_name in ['HA']): #no training phase
        model = HA()
        val_pred = model.predict(val_x)
        test_pred = model.predict(test_x)
        

    if(model_name in ['LR','ARIMA']): # one-shot training phase
        if(model_name == 'LR'):
            model = LR()
        if(model_name == 'ARIMA'):
            pass        
        model.train(train_x,train_y)
        val_pred = model.predict(val_x)
        test_pred = model.predict(test_x)
        
elif(model_name in ['CNN','LSTM', "CONVLSTM"]): #iterative training with euclidean data
    #to do: fill this when pytorch env available
    lag = train_x.shape[1]
    lng = train_x.shape[2]
    lat = train_x.shape[3]
    feat = train_x.shape[4]
    if args.weather:
        train_weather = torch.Tensor(train_weather)
        val_weather = torch.Tensor(val_weather)
        test_weather = torch.Tensor(test_weather)
    if model_name =='CNN':
        model = STResNet(cname+ "/" + service, 6, feat * lag, feat, lng, lat, args.epochs, args.lr, spatial_mask, data_min, data_max, use_ext=args.weather, ext_dim = weather.shape[1] if args.weather else 0, weight_decay = args.regularization)
    elif model_name == 'LSTM':
        model = LSTM(cname + '/' + service, feat, lag, feat, 64, lng, lat, args.epochs, args.lr, spatial_mask, data_min, data_max, args.weather, weather.shape[1] if args.weather else 0)
    elif model_name == 'CONVLSTM':       
        model = ConvLSTM(cname + "/" + service, feat, lat, feat, lng, lat, args.epochs, args.lr, spatial_mask, data_min, data_max, args.weather, weather.shape[1] if args.weather else 0, lstm_hidden = 64)

    if gpu_available:
        model = model.to(gpu)

    if model_name in ['CNN', 'LSTM']:
        trainx = torch.Tensor(train_x.transpose((0, 1, 4, 2, 3)).reshape((-1, lag * feat, lng, lat)))
        valx = torch.Tensor(val_x.transpose((0, 1, 4, 2, 3)).reshape((-1, lag * feat, lng, lat))) 
        testx = torch.Tensor(test_x.transpose((0, 1, 4, 2, 3)).reshape((-1, lag * feat, lng, lat))) 
        trainy = torch.Tensor(train_y.transpose((0, 1, 4, 2, 3)).reshape((-1, feat, lng, lat)))
        valy = torch.Tensor(val_y.transpose((0, 1, 4, 2, 3)).reshape((-1, feat, lng, lat))) 
        testy = torch.Tensor(test_y.transpose((0, 1, 4, 2, 3)).reshape((-1, feat, lng, lat)))   
    elif model_name == 'CONVLSTM':
        trainx = torch.Tensor(train_x.transpose(0, 1, 4, 2, 3))
        trainy = (torch.Tensor(train_y.transpose(0, 1, 4, 2, 3)).reshape(-1, feat, lng, lat))
        valx = torch.Tensor(val_x.transpose(0, 1, 4, 2, 3))
        valy = (torch.Tensor(val_y.transpose(0, 1, 4, 2, 3)).reshape(-1, feat, lng, lat))
        testx = torch.Tensor(test_x.transpose(0, 1, 4, 2, 3))
        testy = (torch.Tensor(test_y.transpose(0, 1, 4, 2, 3).reshape(-1, feat, lng, lat)))
    
    if args.weather:
        train_set = TensorDataset(trainx, trainy, train_weather)
        valid_set = TensorDataset(valx, valy, val_weather)
        test_set = TensorDataset(testx, testy, test_weather)
    else:
        train_set = TensorDataset(trainx, trainy)
        valid_set = TensorDataset(valx, valy)
        test_set = TensorDataset(testx, testy)
    
    trainloader = DataLoader(train_set, batch_size = args.batch_size, shuffle = True)
    validloader = DataLoader(valid_set, batch_size = args.batch_size)
    testloader = DataLoader(test_set, batch_size = args.batch_size)


    model.train_model(trainloader, validloader)
    model.load_model("best")

    
    if model_name == 'CNN':
        if args.weather:
            val_pred = np.expand_dims(model.predict((valx, val_weather)).cpu().numpy().transpose(0, 2, 3, 1), 1)
            test_pred = np.expand_dims(model.predict((testx, test_weather)).cpu().numpy().transpose(0, 2, 3, 1), 1)
        else:
            val_pred = np.expand_dims(model.predict(valx).cpu().numpy().transpose(0, 2, 3, 1), 1) 
            test_pred = np.expand_dims(model.predict(testx).cpu().numpy().transpose(0, 2, 3, 1), 1) 
    else:
        val_pred = np.expand_dims(model.predict_loader(validloader).transpose(0, 2, 3, 1), 1)
        test_pred = np.expand_dims(model.predict_loader(testloader).transpose(0, 2, 3, 1), 1)


"""
elif(model_name[0]=='G'):
    num_nodes = train_x.shape[2]
    feat = train_x.shape[3]
    lag = train_x.shape[1]
    gcn_trainx = torch.Tensor(train_x.transpose(0, 2, 1, 3)).reshape(train_x.shape[0], train_x.shape[2], -1)
    gcn_valx = torch.Tensor(val_x.transpose(0, 2, 1, 3)).reshape(val_x.shape[0], val_x.shape[2], -1)
    gcn_testx = torch.Tensor(test_x.transpose(0, 2, 1, 3)).reshape(test_x.shape[0], test_x.shape[2], -1)
    gcn_trainy = torch.Tensor(train_y.transpose(0, 2, 1, 3)).reshape(train_y.shape[0], train_y.shape[2], -1)
    gcn_valy = torch.Tensor(val_y.transpose(0, 2, 1, 3)).reshape(val_y.shape[0], val_y.shape[2], -1)
    gcn_testy = torch.Tensor(test_y.transpose(0, 2, 1, 3)).reshape(test_y.shape[0], test_y.shape[2], -1)
    # now gcn_trainx should have shape (?, # valid points, feat * lag)


    train_set = TensorDataset(gcn_trainx, gcn_trainy)
    valid_set = TensorDataset(gcn_valx, gcn_valy)
    test_set = TensorDataset(gcn_testx, gcn_testy)
    trainloader = DataLoader(train_set, batch_size = args.batch_size, shuffle = True)
    validloader = DataLoader(valid_set, batch_size = args.batch_size)
    testloader = DataLoader(test_set, batch_size = args.batch_size)
    
    model = GCN(cname + "/" + service, 2, feat * lag, feat, 64, args.epochs, args.lr, \
        data_min, data_max, torch.Tensor(A_euc), torch.Tensor(A_poi), torch.Tensor(A_conn), 
        "leakyrelu", args.regularization)
    if gpu_available:
        model = model.to(gpu)
    model.train_model(trainloader, validloader)
    model.load_model("best")
    val_pred = np.expand_dims(model.predict_loader(validloader), 1)
    test_pred = np.expand_dims(model.predict_loader(testloader), 1)
"""

if service =='all':
    name_list = ['inflow','outflow','demand','supply']
    if cname not in ['chengdu', 'xian']:
        scalar = np.array([inflow_max-inflow_min , outflow_max-outflow_min, demand_max-demand_min, supply_max-supply_min])
    else:
        scalar = np.array([inflow_max-inflow_min, outflow_max-outflow_min, demand_max-demand_min])
    if model_name[0] != 'G':
        test_mae = masked_mae(test_y*scalar,test_pred*scalar,spatial_mask,preserve_f_dim=True)
        test_rmse = masked_rmse(test_y*scalar,test_pred*scalar,spatial_mask,preserve_f_dim=True)
        val_mae = masked_mae(val_y*scalar,val_pred*scalar,spatial_mask,preserve_f_dim=True)
        val_rmse = masked_rmse(val_y*scalar,val_pred*scalar,spatial_mask,preserve_f_dim=True)
    else:
        val_mae = np.mean(np.abs(val_y - val_pred), axis = (0, 1, 2))
        test_mae = np.mean(np.abs(test_y - test_pred), axis = (0, 1, 2))
        val_rmse = np.sqrt(np.mean((val_y - val_pred) ** 2, axis = (0, 1, 2)))
        test_rmse = np.sqrt(np.mean((test_y - test_pred) ** 2, axis = (0, 1, 2)))
        val_mae *= scalar
        val_rmse *= scalar
        test_mae *= scalar
        test_rmse *= scalar
    if cname not in ['xian', 'chengdu']:
        print(scalar)
        print('val score (rmse/mae): %s=%.4f/%.4f, %s==%.4f/%.4f, %s==%.4f/%.4f, %s==%.4f/%.4f'%(name_list[0],val_rmse[0],val_mae[0],\
            name_list[1],val_rmse[1],val_mae[1],
            name_list[2],val_rmse[2],val_mae[2],
            name_list[3],val_rmse[3],val_mae[3]))
        print('test score (rmse/mae): %s=%.4f/%.4f, %s==%.4f/%.4f, %s==%.4f/%.4f, %s==%.4f/%.4f'%(name_list[0],test_rmse[0],test_mae[0],\
            name_list[1],test_rmse[1],test_mae[1],
            name_list[2],test_rmse[2],test_mae[2],
            name_list[3],test_rmse[3],test_mae[3]))
        logging.info('val score (rmse/mae): %s=%.4f/%.4f, %s==%.4f/%.4f, %s==%.4f/%.4f, %s==%.4f/%.4f'%(name_list[0],val_rmse[0],val_mae[0],\
            name_list[1],val_rmse[1],val_mae[1],
            name_list[2],val_rmse[2],val_mae[2],
            name_list[3],val_rmse[3],val_mae[3]))
        logging.info('test score (rmse/mae): %s=%.4f/%.4f, %s==%.4f/%.4f, %s==%.4f/%.4f, %s==%.4f/%.4f'%(name_list[0],test_rmse[0],test_mae[0],\
            name_list[1],test_rmse[1],test_mae[1],
            name_list[2],test_rmse[2],test_mae[2],
            name_list[3],test_rmse[3],test_mae[3]))
    else:
        print(scalar)
        print('val score (rmse/mae): %s=%.4f/%.4f, %s==%.4f/%.4f, %s==%.4f/%.4f'%(name_list[0],val_rmse[0],val_mae[0],\
                name_list[1], val_rmse[1], val_mae[1], 
                name_list[2], val_rmse[2], val_mae[2]))
        print('test score (rmse/mae): %s=%.4f/%.4f, %s==%.4f/%.4f, %s==%.4f/%.4f'%(name_list[0],test_rmse[0],test_mae[0],\
                name_list[1], test_rmse[1], test_mae[1], 
                name_list[2], test_rmse[2], test_mae[2]))
        logging.info('val score (rmse/mae): %s=%.4f/%.4f, %s==%.4f/%.4f, %s==%.4f/%.4f'%(name_list[0],val_rmse[0],val_mae[0],\
                name_list[1], val_rmse[1], val_mae[1], 
                name_list[2], val_rmse[2], val_mae[2]))
        logging.info('test score (rmse/mae): %s=%.4f/%.4f, %s==%.4f/%.4f, %s==%.4f/%.4f'%(name_list[0],test_rmse[0],test_mae[0],\
                name_list[1], test_rmse[1], test_mae[1], 
                name_list[2], test_rmse[2], test_mae[2]))
        

else:
    if model_name[0]!= 'G':

        val_mae = masked_mae(val_y,val_pred,spatial_mask)
        val_rmse = masked_rmse(val_y,val_pred,spatial_mask)
        test_mae = masked_mae(test_y,test_pred,spatial_mask)
        test_rmse = masked_rmse(test_y,test_pred,spatial_mask)
    else:
        val_mae = np.mean(np.abs((val_y - val_pred)))
        test_mae = np.mean(np.abs(test_y - test_pred))
        val_rmse = np.sqrt(np.mean((val_y - val_pred)**2))
        test_rmse = np.sqrt(np.mean((test_y - test_pred)**2))
    print(data_max - data_min)
    print('validation score: rmse = %.4f, mae = %.4f'%(val_rmse*(data_max-data_min),val_mae*(data_max-data_min)))
    print('test score: rmse = %.4f, mae = %.4f'%(test_rmse*(data_max-data_min),test_mae*(data_max-data_min))) 
    logging.info('validation score: rmse = %.4f, mae = %.4f'%(val_rmse*(data_max-data_min),val_mae*(data_max-data_min)))
    logging.info('test score: rmse = %.4f, mae = %.4f'%(test_rmse*(data_max-data_min),test_mae*(data_max-data_min)))
    
