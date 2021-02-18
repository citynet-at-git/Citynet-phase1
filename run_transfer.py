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
parser.add_argument('-a', '--transfer_algorithm', type = str, default = "finetune", help = "what transfer learning algorithm to use")
parser.add_argument('--source', type = str, help = "what source city to use as transfer")
parser.add_argument('--target', type = str, help = 'what target city to use as transfer')
parser.add_argument('--source-path', type = str, help = 'file path that links to the source model')
parser.add_argument('--loss-w', type = float, default = 0, help = "the weight w in regiontrans")
parser.add_argument('-t', '--transfer_data', type = int, default = 7, help = "how many days of data to use for tuning. Currently supports 3 and 7")
parser.add_argument('--dictpath', type = str, help = "The matching dict to use for doing regiontrans")
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
gpu_available = torch.cuda.is_available()
if gpu_available:
    gpu = torch.device("cuda:0")

pref = args.path_prefix
target_cname = args.target
target_shortcname = {
    "beijing":"bj",
    "chengdu":"cd",
    "shanghai":"sh",
    "shenzhen":"sz",
    "chongqing":"cq",
    "xian":"xa",
}[target_cname]
source_cname = args.source
source_shortcname = {
    "beijing":"bj",
    "chengdu":"cd",
    "shanghai":"sh",
    "shenzhen":"sz",
    "chongqing":"cq",
    "xian":"xa",
}[source_cname]
ylength = args.output_length
service = args.service
lag_mode = args.input_lag_mode
model_name = args.model # Currently for transfer, we only support LSTM and ConvLSTM

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


def load_data(cname, short_cname, serv):
    weather_hour, weather_max, weather_min = min_max_normalize(np.load(pref + "weather/weather_" + short_cname + ".npy"))
    weather = np.stack([weather_hour,weather_hour],axis=1).reshape(weather_hour.shape[0]*2,weather_hour.shape[1]) #(time,49)
    if(serv == 'inflow'):
        data = np.expand_dims(np.load(pref + '/%s/inflow_arr_%s.npy' % (cname, short_cname)),-1) #30min
        data,data_max,data_min = min_max_normalize(data)
    if(serv == 'outflow'):
        data = np.expand_dims(np.load(pref + '%s/outflow_arr_%s.npy' % (cname, short_cname)),-1) #30min
        data,data_max,data_min = min_max_normalize(data)
    if(serv == 'demand'):
        demand_10min = np.load(pref + "/%s/%sdemand.npy" % (cname, short_cname)) #10min
        data = np.expand_dims(np.array([demand_10min[i:i+3].sum(0) for i in range(0,demand_10min.shape[0],3)]),-1)
        data,data_max,data_min = min_max_normalize(data)
    if(serv == 'supply'):
        supply_10min = np.load(pref + "/%s/%ssupply.npy" % (cname, short_cname)) #10min
        data = np.expand_dims(np.array([supply_10min[i:i+3].mean(0) for i in range(0,supply_10min.shape[0],3)]),-1)
        data,data_max,data_min = min_max_normalize(data)    
    if(serv == 'all'):
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
    return (data, data_max, data_min, weather, weather_max, weather_min)

print('==========loading data files============')
#load mask
mask_source = np.load(pref + "%s/mask_%s.npy"%(source_cname, source_shortcname))
temporal_mask_source = (mask_source.mean((1,2))>0) # True-False array with shape [T] 
spatial_mask_source = (mask_source.mean(0)>0) # True-False array with shape [lng, lat]
print('source temporal mask density:',temporal_mask_source.sum()/len(temporal_mask_source))
print('source spatial mask density:',spatial_mask_source.sum()/len(spatial_mask_source.flatten()))

mask_target = np.load(pref + "%s/mask_%s.npy"%(target_cname, target_shortcname))
temporal_mask_target = (mask_target.mean((1,2))>0) # True-False array with shape [T] 
spatial_mask_target = (mask_target.mean(0)>0) # True-False array with shape [lng, lat]
print('target temporal mask density:',temporal_mask_target.sum()/len(temporal_mask_target))
print('target spatial mask density:',spatial_mask_target.sum()/len(spatial_mask_target.flatten()))


source_data, source_datamax, source_datamin, source_weather, source_weathermax, source_weathermin = load_data(source_cname, source_shortcname, service)
target_data, target_datamax, target_datamin, target_weather, target_weathermax, target_weathermin = load_data(target_cname, target_shortcname, service)
print('source data loaded:',service,'-',source_data.shape)
print("target data loaded:", service, '-', target_data.shape)
if args.weather:
    print('source weather data loaded:',source_weather.shape)
    print('target weather data loaded', target_weather.shape)


no_lag = [-1]
hour_lag = [-5,-4,-3,-2,-1]
one_day_lag = [-48,-4,-3,-2,-1]

if(int(lag_mode)==0):
    lag=no_lag
if(int(lag_mode)==1):
    lag=hour_lag
if(int(lag_mode)==2):
    lag=one_day_lag

train_weather_source, _, val_weather_source, _, test_weather_source, _ = split(source_weather, lag, temporal_mask_source)
train_x_source, train_y_source, val_x_source, val_y_source, test_x_source, test_y_source = split(source_data, lag, temporal_mask_source)

train_weather_target, _, val_weather_target, _, test_weather_target, _ = split(target_weather, lag, temporal_mask_target)
train_x_target, train_y_target, val_x_target, val_y_target, test_x_target, test_y_target = split(target_data, lag, temporal_mask_target)

# process transfer data
if args.transfer_data == 7:
    transfer_data_range = transfer_data_range_week
elif args.transfer_data == 3:
    transfer_data_range = transfer_data_range_3day
idx_start_target = transfer_data_range[target_cname][0]
idx_end_target = transfer_data_range[target_cname][1]
train_data_target = target_data[idx_start_target:idx_end_target]
train_weather_target = target_weather[idx_start_target:idx_end_target]
idx_start_source = transfer_data_range[source_cname][0]
idx_end_source = transfer_data_range[source_cname][1]
train_data_source = source_data[idx_start_source:idx_end_source]
train_weather_source = source_weather[idx_start_source:idx_end_source]


x_source = []
y_source = []
w_source = []
x_target = []
y_target = []
w_target = []
for i in range(-lag[0], len(train_data_target) - ylength +1):
    x_idx = list(map(lambda x:x+i, lag))
    y_idx = [i+o for o in range(ylength)]
    biased_idxs = np.array(x_idx + y_idx) + idx_start_source
    if (temporal_mask_source[biased_idxs] == 0).sum() == 0:
        x_source.append(train_data_source[x_idx])
        y_source.append(train_data_source[y_idx])
        w_source.append(train_weather_source[x_idx])
        x_target.append(train_data_target[x_idx])
        y_target.append(train_data_target[y_idx])
        w_target.append(train_weather_target[x_idx])
train_x_target = np.stack(x_target, 0)
train_y_target = np.stack(y_target, 0)
train_weather_target = np.stack(w_target, 0)
train_x_source = np.stack(x_source, 0)
train_y_source = np.stack(y_source, 0)
train_weather_source = np.stack(w_source, 0)


print('Source data split to: train_x-%s, train_y-%s, val_x-%s, val_y-%s, test_x-%s, test_y-%s'%\
    (train_x_source.shape,train_y_source.shape,val_x_source.shape,val_y_source.shape,test_x_source.shape,test_y_source.shape))
print('Target data split to: train_x-%s, train_y-%s, val_x-%s, val_y-%s, test_x-%s, test_y-%s'%\
    (train_x_target.shape,train_y_target.shape,val_x_target.shape,val_y_target.shape,test_x_target.shape,test_y_target.shape))


if args.transfer_algorithm == 'finetune':
    # First load source model
    lag = train_x_target.shape[1]
    lng = train_x_target.shape[2]
    lat = train_x_target.shape[3]
    feat = train_x_target.shape[4]
    if model_name == "CONVLSTM":
        model = ConvLSTM(target_cname + "/" + service, feat, lag, feat, lng, lat, args.epochs, args.lr, spatial_mask_target, 
                target_datamin, target_datamax, use_ext = args.weather, ext_dim = target_weather.shape[1] if args.weather else 0, lstm_hidden = 64
            )
        source_dict = torch.load(args.source_path)
        model.load_state_dict(source_dict)
        
    elif model_name == "LSTM":
        model = LSTM(target_cname + "/" + service, feat, lag, feat, 64, lng, lat, args.epochs, args.lr, spatial_mask_target,\
             target_datamin, target_datamax, use_ext = args.weather, ext_dim = target_weather.shape[1] if args.weather else 0)
        source_dict = torch.load(args.source_path)
        target_dict = model.state_dict()
        num_loaded = 0
        for k in target_dict.keys():
            if k in source_dict.keys() and source_dict[k].shape == target_dict[k].shape:
                num_loaded += 1
                target_dict[k] = source_dict[k]
            elif k == 'conv1.weight':
                # convolution kernel channel mismatch
                # mismatch channel is in dimension 1
                if target_dict[k].shape[1] < source_dict[k].shape[1]:
                    # target has 3 channels, source has 4
                    target_dict[k] = source_dict[k][:, :3, :, :]    
                else:
                    print("Failed to load %s" % k)     
                num_loaded += 1
            elif k == 'linear.weight':
                if target_dict[k].shape[0] < source_dict[k].shape[0]:
                    # target has 3 channels, source has 4
                    target_dict[k] = source_dict[k][:3, :]
                else:
                    print("Failed to load %s" % k)
                num_loaded += 1
            elif k == 'linear.bias':
                if target_dict[k].shape[0] < source_dict[k].shape[0]:
                    target_dict[k] = source_dict[k][:3]
                else:
                    print("Failed to load %s" % k)
                num_loaded += 1
            else:
                print("Failed to load %s" % k)
        print("Loaded %d out of %d parameters"%(num_loaded, len(target_dict)))
    else:
        raise NotImplementedError("Other models not implemented for finetune.")
    if gpu_available:
        model = model.to(gpu)

    trainx = torch.Tensor(train_x_target.transpose((0, 1, 4, 2, 3))).contiguous()
    valx = torch.Tensor(val_x_target.transpose((0, 1, 4, 2, 3))).contiguous()
    testx = torch.Tensor(test_x_target.transpose((0, 1, 4, 2, 3))).contiguous()
    trainy = torch.Tensor(train_y_target.transpose((0, 1, 4, 2, 3)).reshape((-1, feat, lng, lat)))
    valy = torch.Tensor(val_y_target.transpose((0, 1, 4, 2, 3)).reshape((-1, feat, lng, lat)))
    testy = torch.Tensor(test_y_target.transpose((0, 1, 4, 2, 3)).reshape((-1, feat, lng, lat)))
    if model_name == "LSTM":
        trainx = trainx.view(-1, lag * feat, lng, lat)
        valx = valx.view(-1, lag * feat, lng, lat)
        testx = testx.view(-1, lag * feat, lng, lat)

    if args.weather:
        train_weather_target = torch.Tensor(train_weather_target)
        val_weather_target = torch.Tensor(val_weather_target)
        test_weather_target = torch.Tensor(test_weather_target)
        train_set = TensorDataset(trainx, trainy, train_weather_target)
        valid_set = TensorDataset(valx, valy, val_weather_target)
        test_set = TensorDataset(testx, testy, test_weather_target)
    else:
        train_set = TensorDataset(trainx, trainy)
        valid_set = TensorDataset(valx, valy)
        test_set = TensorDataset(testx, testy)
    trainloader = DataLoader(train_set, batch_size = args.batch_size, shuffle=True)
    validloader = DataLoader(valid_set, batch_size = args.batch_size)
    testloader = DataLoader(test_set, batch_size = args.batch_size)
    model.train_model(trainloader, validloader)
    model.load_model("best")
    val_pred = np.expand_dims(model.predict_loader(validloader).transpose(0, 2, 3, 1), 1)
    test_pred = np.expand_dims(model.predict_loader(testloader).transpose(0, 2, 3, 1), 1)

elif args.transfer_algorithm == 'regiontrans':
    target_lag = train_x_target.shape[1]
    target_lng = train_x_target.shape[2]
    target_lat = train_x_target.shape[3]
    target_feat = train_x_target.shape[4]
    source_lag = train_x_source.shape[1]
    source_lng = train_x_source.shape[2]
    source_lat = train_x_source.shape[3]
    source_feat = train_x_source.shape[4]
    if model_name == "LSTM":
        model = RegionTrans_LSTM(source_shortcname, target_shortcname, target_feat, target_lag, target_feat, 64, args.epochs, args.lr, \
            spatial_mask_source, spatial_mask_target, target_datamin, target_datamax, 
            use_ext = args.weather, ext_dim = target_weather.shape[1], loss_w = args.loss_w, matching_dict_path = args.dictpath)
        source_dict = torch.load(args.source_path)
        target_dict = model.state_dict()
        num_loaded = 0
        for k in target_dict.keys():
            if k in source_dict.keys() and source_dict[k].shape == target_dict[k].shape:
                num_loaded += 1
                target_dict[k] = source_dict[k]
            elif k == 'conv1.weight':
                # convolution kernel channel mismatch
                # mismatch channel is in dimension 1
                if target_dict[k].shape[1] < source_dict[k].shape[1]:
                    # target has 3 channels, source has 4
                    target_dict[k] = source_dict[k][:, :3, :, :]    
                else:
                    print("Failed to load %s" % k)     
                num_loaded += 1
            elif k == 'linear.weight':
                if target_dict[k].shape[0] < source_dict[k].shape[0]:
                    # target has 3 channels, source has 4
                    target_dict[k] = source_dict[k][:3, :]
                else:
                    print("Failed to load %s" % k)
                num_loaded += 1
            elif k == 'linear.bias':
                if target_dict[k].shape[0] < source_dict[k].shape[0]:
                    target_dict[k] = source_dict[k][:3]
                else:
                    print("Failed to load %s" % k)
                num_loaded += 1
            else:
                print("Failed to load %s" % k)
        print("Loaded %d out of %d parameters"%(num_loaded, len(target_dict)))
    elif model_name == "CONVLSTM":
        model = RegionTrans_ConvLSTM(source_shortcname, target_shortcname, target_feat, target_lag, target_feat, args.epochs, args.lr, spatial_mask_source, \
            spatial_mask_target, target_datamin, target_datamax, use_ext = args.weather, ext_dim = target_weather.shape[1], 
            loss_w = args.loss_w, matching_dict_path = args.dictpath)
        source_dict = torch.load(args.source_path)
        model.load_state_dict(source_dict)
    else:
        raise NotImplementedError("Models other than LSTM and CONVLSTM are not implemented on RegionTrans.")

    if gpu_available:
        model = model.to(gpu)

    # prepare data
    # a pair of temporally aligned data
    if model_name == "LSTM":
        rt_trainx_source = torch.Tensor(train_x_source.transpose((0, 1, 4, 2, 3)).reshape((-1, source_lag * source_feat, source_lng, source_lat)))
        rt_trainy_source = torch.Tensor(train_y_source.transpose((0, 1, 4, 2, 3)).reshape((-1, source_feat, source_lng, source_lat)))
        rt_trainx_target = torch.Tensor(train_x_target.transpose((0, 1, 4, 2, 3)).reshape((-1, target_lag * target_feat, target_lng, target_lat)))
        rt_trainy_target = torch.Tensor(train_y_target.transpose((0, 1, 4, 2, 3)).reshape((-1, target_feat, target_lng, target_lat)))

    elif model_name == 'CONVLSTM':
        rt_trainx_source = torch.Tensor(train_x_source.transpose((0, 1, 4, 2, 3)).reshape((-1, source_lag, source_feat, source_lng, source_lat)))
        rt_trainy_source = torch.Tensor(train_y_source.transpose((0, 1, 4, 2, 3)).reshape((-1, source_feat, source_lng, source_lat)))
        rt_trainx_target = torch.Tensor(train_x_target.transpose((0, 1, 4, 2, 3)).reshape((-1, target_lag, target_feat, target_lng, target_lat))) 
        rt_trainy_target = torch.Tensor(train_y_target.transpose((0, 1, 4, 2, 3)).reshape((-1, target_feat, target_lng, target_lat)))

    rt_valx_target = torch.Tensor(val_x_target.transpose((0, 1, 4, 2, 3)).reshape((-1, target_lag * target_feat, target_lng, target_lat))) 
    rt_testx_target = torch.Tensor(test_x_target.transpose((0, 1, 4, 2, 3)).reshape((-1, target_lag * target_feat, target_lng, target_lat)))
    rt_valy_target = torch.Tensor(val_y_target.transpose((0, 1, 4, 2, 3)).reshape((-1, target_feat, target_lng, target_lat))) 
    rt_testy_target = torch.Tensor(test_y_target.transpose((0, 1, 4, 2, 3)).reshape((-1, target_feat, target_lng, target_lat)))

    if args.weather:
        train_weather_source = torch.Tensor(train_weather_source)
        train_weather_target = torch.Tensor(train_weather_target)
        val_weather_source = torch.Tensor(val_weather_source)
        val_weather_target = torch.Tensor(val_weather_target)
        test_weather_source = torch.Tensor(test_weather_source)
        test_weather_target = torch.Tensor(test_weather_target)
        train_set = TensorDataset(rt_trainx_source, train_weather_source, rt_trainx_target, rt_trainy_target, train_weather_target)
        valid_set = TensorDataset(rt_valx_target, rt_valy_target, val_weather_target)
        test_set = TensorDataset(rt_testx_target, rt_testy_target, test_weather_target)
    else:
        train_set = TensorDataset(rt_trainx_source, rt_trainx_target, rt_trainy_target)
        valid_set = TensorDataset(rt_valx_target, rt_valy_target)
        test_set = TensorDataset(rt_testx_target, rt_testy_target)

    trainloader = DataLoader(train_set, batch_size = args.batch_size, shuffle=True)
    validloader = DataLoader(valid_set, batch_size = args.batch_size)
    testloader = DataLoader(test_set, batch_size = args.batch_size) 

    model.train_model(trainloader, validloader)
    model.load_model("best")
    val_pred = np.expand_dims(model.predict_loader(validloader).transpose(0, 2, 3, 1), 1)
    test_pred = np.expand_dims(model.predict_loader(testloader).transpose(0, 2, 3, 1), 1)

    

if service =='all':
    name_list = ['inflow','outflow','demand','supply']
    # assert len(name_list) == test_pred.shape[-1] 
    scalar = target_datamax - target_datamin
    test_mae = masked_mae(test_y_target*scalar,test_pred*scalar,spatial_mask_target,preserve_f_dim=True)
    test_rmse = masked_rmse(test_y_target*scalar,test_pred*scalar,spatial_mask_target,preserve_f_dim=True)
    val_mae = masked_mae(val_y_target*scalar,val_pred*scalar,spatial_mask_target,preserve_f_dim=True)
    val_rmse = masked_rmse(val_y_target*scalar,val_pred*scalar,spatial_mask_target,preserve_f_dim=True)
    if target_cname not in ['xian', 'chengdu']:
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

    val_rmse = masked_rmse(val_y_target, val_pred, spatial_mask_target)
    val_mae = masked_mae(val_y_target, val_pred, spatial_mask_target)
    test_mae = masked_mae(test_y_target, test_pred, spatial_mask_target)
    test_rmse = masked_rmse(test_y_target, test_pred, spatial_mask_target)
    print(target_datamax - target_datamin)
    print('validation score: rmse = %.4f, mae = %.4f'%(val_rmse*(target_datamax-target_datamin),val_mae*(target_datamax-target_datamin)))
    print('test score: rmse = %.4f, mae = %.4f'%(test_rmse*(target_datamax-target_datamin),test_mae*(target_datamax-target_datamin)))
    logging.info('validation score: rmse = %.4f, mae = %.4f'%(val_rmse*(target_datamax-target_datamin),val_mae*(target_datamax-target_datamin)))
    logging.info('test score: rmse = %.4f, mae = %.4f'%(test_rmse*(target_datamax-target_datamin),test_mae*(target_datamax-target_datamin)))
