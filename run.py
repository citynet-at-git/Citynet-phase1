import numpy as np
import argparse
from model import *
import torch
import torch.nn as nn
import torch.optim as optimizer


parser = argparse.ArgumentParser()

parser.add_argument('-p','--path_prefix',default='../citynet-phase1-data/')
parser.add_argument('-n','--city_name',default='bj')
parser.add_argument('-y','--output_length',default=1)
parser.add_argument('-s','--service',default='all')
parser.add_argument('-x','--input_lag_mode',default=1)
parser.add_argument('-m','--model',default='GAT')
parser.add_argument('-b','--batch_size',default=16)
parser.add_argument('-l','--learning_rate',default=0.001)
parser.add_argument('-u','--num_layers',default=8)

#load all args
args = parser.parse_args()

pref = args.path_prefix # prefix of file paths ~/citynet-phase1-data
cname = args.city_name  # bj or sh or ...
ylength = args.output_length # temporal length of prediction output
service = args.service #service name -- demand/inflow/... or all
lag_mode = args.input_lag_mode #see below definition of lag
model_name = args.model.upper() # 'HA', 'LR', 'ARIMA', 'CNN', 'CONVLSTM', 'GCN', 'GAT'
batch = int(args.batch_size)
lr = float(args.learning_rate)
nl = int(args.num_layers)

patience = 15

print(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
    cnt=-lags[0]
    for i in range(-lags[0],len(temporal_mask)-ylength):
        x_idx = list(map(lambda x:x+cnt,lags))
        y_idx = [cnt+o for o in range(0,ylength)]
        
        x_idxs = list(map(lambda x:x+i,lags))
        y_idxs = [i+o for o in range(0,ylength)]

        selected = temporal_mask[(x_idxs+y_idxs)]
        if(temporal_mask[i]==1):
            cnt+=1
        if(selected.sum()<len(selected)):
            continue
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

def mae(T,P,preserve_f_dim=False):
    if(preserve_f_dim):
        return (abs(T-P).sum((0,1,2)))/(T.shape[0]*T.shape[1]*T.shape[2])
    else:
        return abs(T-P).mean()

def masked_rmse(T,P,mask,preserve_f_dim=False):
    mask = np.expand_dims(mask,(0,-1))
    mask = np.expand_dims(mask,(0))
    mask = np.repeat(np.repeat(np.repeat(mask,T.shape[0],0),T.shape[-1],-1),T.shape[1],1)
    print(T.shape,P.shape,mask.shape)
    if(preserve_f_dim):
        return np.sqrt((((T-P)**2)*mask).sum((0,1,2,3))/(mask.sum()/T.shape[-1]))
    else:
        return np.sqrt((((T-P)**2)*mask).sum()/mask.sum())

def rmse(T,P,preserve_f_dim=False):
    if(preserve_f_dim):
        return np.sqrt(((T-P)**2).sum((0,1,2))/(T.shape[0]*T.shape[1]*T.shape[2]))
    else:
        return np.sqrt(((T-P)**2).mean())
    
def idx_2d_to_1d(from_coord,from_shape):
    if(from_coord[0]>from_shape[0]-1 or from_coord[1]>from_shape[1]-1):
        return None
    else:
        return from_shape[1]*from_coord[0]+from_coord[1]
    
def gen_adjacency_A(shape):
    lng = shape[0]
    lat = shape[1]
    A = np.zeros((lng*lat,lng*lat))
    for a in range(0,lng):
        for b in range(0,lat):
            center = idx_2d_to_1d((a,b),shape)

            for ai in [a-1,a,a+1]:
                for bi in [b-1,b,b+1]:
                    p = idx_2d_to_1d((ai,bi),shape)
                    if(p is not None):
                        A[center,p]=1
                        A[p,center]=1
    return A


def mae_pytorch(T,P):
    return torch.abs(T-P).mean()
def rmse_pytorch(T,P):
    return torch.sqrt(((T-P)**2).mean())

def gen_poi_A(poi,density=0.01):
    affn = np.matmul(poi,poi.transpose())
    s = sorted(affn.flatten())
    threshold = s[int(len(s)*(1-density))]
    return (affn>threshold)


print('==========loading data files============')
#load mask

mask = np.load(pref+'mask/mask_'+cname+'.npy') #(time,lat,lng)

temporal_mask = (mask.mean((1,2))>0) #(time)
spatial_mask = (mask.mean(0)>0).flatten() #(lng*lat)

print('temporal mask density:',temporal_mask.sum()/len(temporal_mask),'; shape:',temporal_mask.shape)
print('spatial mask density:',spatial_mask.sum()/len(spatial_mask.flatten()),'; shape:',spatial_mask.shape)

#load data and minmax normalize    
#poi,poi_max,poi_min = min_max_normalize(np.load(pref+'poi/poi_vectors_'+cname+'.npy'),0.9) #(lng,lat,14)
poi_raw = np.load(pref+'poi/poi_vectors_'+cname+'.npy') #(lng,lat,14)
lng = poi_raw.shape[0]
lat = poi_raw.shape[1]
poi_raw = np.reshape(poi_raw,(lng*lat,poi_raw.shape[2])) #(lng*lat,14)

poi_mask = poi_raw[spatial_mask,:] #(masked_n,14)
poi,poi_max,poi_min = min_max_normalize(poi_mask,0.9) #(masked_n,14), int, int

weather_hour,weather_max,weather_min = min_max_normalize(np.load(pref+'weather/weather_'+cname+'.npy')) # one-hour
weather_copy = np.stack([weather_hour,weather_hour],axis=1).reshape(weather_hour.shape[0]*2,weather_hour.shape[1]) #(time,49)
weather = weather_copy[temporal_mask,:] #(masked_t,49)

# all loaded data in (Sample,x/y_length(temporal),lng,lat,feature) format -- by split() function
# for single tasks, feature = 1
# for all tasks, feature = 4 (stacked)
if(service == 'inflow'):
    data = np.expand_dims(np.load(pref+'taxi/inflow/inflow_arr_'+cname+'.npy'),-1) #30min create extra dim for stack or concat
    data,data_max,data_min = min_max_normalize(data) 
if(service == 'outflow'):
    data = np.expand_dims(np.load(pref+'taxi/outflow/outflow_arr_'+cname+'.npy'),-1) #30min
    data,data_max,data_min = min_max_normalize(data)
if(service == 'demand'):
    demand_10min = np.load(pref+'taxi/demand/'+cname+'demand.npy') #10min
    data = np.expand_dims(np.array([demand_10min[i:i+3].sum(0) for i in range(0,demand_10min.shape[0],3)]),-1)
    data,data_max,data_min = min_max_normalize(data)
if(service == 'supply'):
    supply_10min = np.load(pref+'taxi/supply/'+cname+'supply.npy') #10min
    data = np.expand_dims(np.array([supply_10min[i:i+3].mean(0) for i in range(0,supply_10min.shape[0],3)]),-1)
    data,data_max,data_min = min_max_normalize(data)    
        
    
if(service == 'all'):
    inflow,inflow_max,inflow_min = min_max_normalize(np.load(pref+'taxi/inflow/inflow_arr_'+cname+'.npy'))
    outflow,outflow_max,outflow_min = min_max_normalize(np.load(pref+'taxi/outflow/outflow_arr_'+cname+'.npy'))
    demand_10min= np.load(pref+'taxi/demand/'+cname+'demand.npy')
    if(cname == 'xa' or cname == 'cd'):
        pass
    else:
        supply_10min = np.load(pref+'taxi/supply/'+cname+'supply.npy')
    demand,demand_max,demand_min = min_max_normalize(np.array([demand_10min[i:i+3].sum(0) for i in range(0,demand_10min.shape[0],3)]))
    if(cname=='xa' or cname=='cd'):
        data = np.stack([inflow,outflow,demand],axis=-1)
        data_max = np.array([inflow_max,outflow_max,demand_max])
        data_min = np.array([inflow_min,outflow_min,demand_min])         
    else:
        supply,supply_max,supply_min = min_max_normalize(np.array([supply_10min[i:i+3].mean(0) for i in range(0,supply_10min.shape[0],3)]))
        data = np.stack([inflow,outflow,demand,supply],axis=-1)
        data_max = np.array([inflow_max,outflow_max,demand_max,supply_max])
        data_min = np.array([inflow_min,outflow_min,demand_min,supply_min]) 
    
data = np.reshape(data,(data.shape[0],lat*lng,data.shape[-1]))
data = data[temporal_mask,:,:][:,spatial_mask,:]

print('masked poi data loaded:',poi.shape)
print('masked weather data loaded:',weather.shape)
print('masked data loaded:',service,'-',data.shape)


if(model_name[0]=='G'):        
    A_euc_raw = gen_adjacency_A((lng,lat))
    A_euc = A_euc_raw[spatial_mask,:][:,spatial_mask]
    A_poi = gen_poi_A(poi)
    
    if(cname == 'xa' or cname=='cd'):
        road_raw = np.zeros_like(A_euc_raw)
    else:
        road_raw = (np.load(pref+'road/'+cname+'_conn.npy')>0) #(lng*lat,lng*lat)
    print(road_raw.shape,spatial_mask.shape)
    A_conn = road_raw[spatial_mask,:][:,spatial_mask] #(masked_n,masked_n)
    
    print('masked road data loaded:',A_conn.shape)    
    
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
print('split data to: \n train_x-%s, train_y-%s, \n val_x-%s, val_y-%s, \n test_x-%s, test_y-%s'%(train_x.shape,train_y.shape,val_x.shape,val_y.shape,test_x.shape,test_y.shape))


if(model_name in ['HA','LR','ARIMA']): 
    if(model_name in ['HA']): #no training phase
        model = HA()
        test_pred = model.predict(test_x)
        
    if(model_name in ['LR','ARIMA']): # one-shot training phase
        if(model_name == 'LR'):
            model = LR()
        if(model_name == 'ARIMA'):
            model = ARIMA()        
        model.train(train_x,train_y)
        test_pred = model.predict(test_x)
        
elif(model_name in ['CNN','CONVLSTM','GCN','GAT']): #iterative training
    train_x = torch.Tensor(train_x).to(device)
    train_y = torch.Tensor(train_y).to(device)
    val_x = torch.Tensor(val_x).to(device)
    val_y = torch.Tensor(val_y).to(device)
    test_x = torch.Tensor(test_x).to(device)
    test_y = torch.Tensor(test_y).to(device)

    #to do: fill this when pytorch env available
    if(model_name=='CNN'):
        pass
    if(model_name=='CONVLSTM'):
        pass
    if(model_name[0]=='G'):
        if(cname=='xa' or cname=='cd'):
            out_dim = 3*ylength if service =='all' else 1*ylength
            in_dim = 15*ylength if service =='all' else 5*ylength
        else:
            in_dim = 20*ylength if service =='all' else 5*ylength
            out_dim = 4*ylength if service =='all' else 1*ylength
            
        A_euc = torch.Tensor(A_euc+np.eye(A_euc.shape[0])).to(device)
        A_poi = torch.Tensor(A_poi+np.eye(A_poi.shape[0])).to(device)
        A_road = torch.Tensor(A_conn+np.eye(A_conn.shape[0])).to(device)
        if(model_name=='GCN'):
            model = GCN(in_dim,out_dim,A_euc,A_poi,A_road,ylength,n_layers=nl).to(device)
        if(model_name=='GAT'):
            model = GAT(in_dim,out_dim,A_euc,A_poi,A_road,ylength,n_layers=nl,device=device).to(device)
        optimizer = optimizer.Adam(model.parameters(), lr=lr, weight_decay=1e-6)        
            
        val_err = []
        tst_err = []
        preds = []
        epoch = 0
        idx = np.array([i for i in range(0,train_x.size(0))])
        while True:
            epoch += 1
            np.random.shuffle(idx)
            train_x = train_x[idx,:,:,:]
            train_y = train_y[idx,:,:,:]
            model.train()
            for i in range(0,train_x.size(0),int(batch)):
                input_x = train_x[i:i+batch,:,:,:]
                input_y = train_y[i:i+batch,:,:,:]
                optimizer.zero_grad()
                output = model(input_x)

                loss = rmse_pytorch(input_y,output)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()

                val_out_list = []
                for i in range(0,val_x.size(0),int(batch)):
                    end_pos = min(val_x.size(0),i+batch)    
                    optimizer.zero_grad()            
                    val_out_list.append(model(val_x[i:end_pos,:,:,:]))
                val_out = torch.cat(val_out_list,dim=0)

                loss = rmse_pytorch(val_y,val_out)
                val_err.append(loss.item())
                if(epoch%1==0):
                    print('epoch',epoch,' -- val loss:',loss.item())


                test_out_list = []
                for i in range(0,test_x.size(0),int(batch)):
                    end_pos = min(test_x.size(0),i+batch)    
                    optimizer.zero_grad()            
                    test_out_list.append(model(test_x[i:end_pos,:,:,:]))
                test_out = torch.cat(test_out_list,dim=0)

                tst_out = test_out.cpu().detach().numpy()
                preds.append(tst_out)
                if(len(preds)>patience):
                    preds = preds[1:]

                #early stopping criterion
                if(np.argmin(val_err)==len(val_err)-patience):
                    test_pred = preds[0]
                    break
        
if service =='all':
    if(cname == 'xa' or cname == 'cd'):
        name_list = ['inflow','outflow','demand']
        assert len(name_list) == test_pred.shape[-1] 
        scalar = np.array([inflow_max-inflow_min,outflow_max-outflow_min,demand_max-demand_min])
        #mae = masked_mae(test_y*scalar,test_pred*scalar,spatial_mask,preserve_f_dim=True)
        #rmse = masked_rmse(test_y*scalar,test_pred*scalar,spatial_mask,preserve_f_dim=True)
        mae = mae(test_y.cpu().detach().numpy()*scalar,test_pred*scalar,preserve_f_dim=True)
        rmse = rmse(test_y.cpu().detach().numpy()*scalar,test_pred*scalar,preserve_f_dim=True)
        print('test score (rmse/mae): %s=%.4f/%.4f, %s==%.4f/%.4f, %s==%.4f/%.4f'%(name_list[0],rmse[0],mae[0],name_list[1],rmse[1],mae[1],name_list[2],rmse[2],mae[2]))
    else:
        name_list = ['inflow','outflow','demand','supply']
        assert len(name_list) == test_pred.shape[-1] 
        scalar = np.array([inflow_max-inflow_min,outflow_max-outflow_min,demand_max-demand_min,supply_max-supply_min])
        #mae = masked_mae(test_y*scalar,test_pred*scalar,spatial_mask,preserve_f_dim=True)
        #rmse = masked_rmse(test_y*scalar,test_pred*scalar,spatial_mask,preserve_f_dim=True)
        mae = mae(test_y.cpu().detach().numpy()*scalar,test_pred*scalar,preserve_f_dim=True)
        rmse = rmse(test_y.cpu().detach().numpy()*scalar,test_pred*scalar,preserve_f_dim=True)
        print('test score (rmse/mae): %s=%.4f/%.4f, %s==%.4f/%.4f, %s==%.4f/%.4f, %s==%.4f/%.4f'%(name_list[0],rmse[0],mae[0],name_list[1],rmse[1],mae[1],name_list[2],rmse[2],mae[2],name_list[3],rmse[3],mae[3]))
else:
    mae = mae(test_y.cpu().detach().numpy(),test_pred)
    rmse = rmse(test_y.cpu().detach().numpy(),test_pred)

    print('test score: rmse = %.4f, mae = %.4f'%(rmse*(data_max-data_min),mae*(data_max-data_min)))

print(test_pred.sum())

