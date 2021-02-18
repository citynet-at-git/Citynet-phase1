import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
from scipy.spatial.distance import euclidean, correlation
import time
from sklearn.decomposition import PCA
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", '--source', help = "Source city")
parser.add_argument('-t', '--target', help = 'Target city')
parser.add_argument('-p', '--period', type = int, default = 2, help = 'Time period. Unit is day')
parser.add_argument('-g', '--granularity', type = int, help = "Granularity of time series. The unit is one timestamp (30 minutes)")
parser.add_argument('-m', '--metric', help = "Metric to use to match regions. Including corr, dtw, poi, poi-cos")
args = parser.parse_args()

short_dict = {
    'beijing':'bj', 
    'shanghai':'sh',
    'shenzhen':'sz',
    'chongqing':'cq',
    'xian':'xa',
    'chengdu':'cd'
}
data_sample_range_week = {
    "beijing":(0, 336), 
    "shanghai":(624, 960), 
    "shenzhen":(576, 912),
    "chongqing":(528, 864),
    "chengdu":(1632, 1968), 
    "xian":(1632, 1968)
}
def min_max_normalize(data,cut_off_percentile=0.99):
    sl = sorted(data.flatten())
    max_val = sl[int(len(sl)*cut_off_percentile)]
    min_val = max(0,sl[0])
    data[data>max_val]=max_val
    data[data<min_val]=min_val
    data -= min_val
    data/=(max_val-min_val)
    return data,max_val,min_val

def load_data(cityname):
    inflow = np.load("../TaxiData/%s/inflow_arr_%s.npy" % (cityname, short_dict[cityname]))
    inflow, inflow_max, inflow_min = min_max_normalize(inflow)

    outflow = np.load("../TaxiData/%s/outflow_arr_%s.npy" % (cityname, short_dict[cityname]))
    outflow, outflow_max, outflow_min = min_max_normalize(outflow)

    demand = np.load("../TaxiData/%s/%sdemand.npy" % (cityname, short_dict[cityname]))
    demand = np.array([demand[i:i+3].sum(0) for i in range(0, len(demand), 3)])
    demand, demand_max, demand_min = min_max_normalize(demand)

    if cityname not in ['xian', 'chengdu']:
        supply = np.load("../TaxiData/%s/%ssupply.npy" % (cityname, short_dict[cityname]))
        supply = np.array([supply[i:i+3].mean(0) for i in range(0, len(supply), 3)])
        supply, supply_max, supply_min = min_max_normalize(supply)

        data = np.stack([inflow, outflow, demand, supply], axis = -1)
        data_max = np.array([inflow_max, outflow_max, demand_max, supply_max])
        data_min = np.array([inflow_min, outflow_min, demand_min, supply_min])
    else:
        data = np.stack([inflow, outflow, demand], axis = -1)
        data_max = np.array([inflow_max, outflow_max, demand_max])
        data_min = np.array([inflow_min, outflow_min, demand_min])
    mask = np.load("../TaxiData/%s/mask_%s.npy" % (cityname, short_dict[cityname]))
    poi = np.load("../TaxiData/poi/poi_vectors_%s.npy" % short_dict[cityname])
    poi, poi_max, poi_min = min_max_normalize(poi)
    return data, data_max, data_min, mask, poi

def region_match_seq_corr(source_name, target_name, period, granu = 1):
    """
    input: 
    source_name, target_name: full length city names
    period: the unit is 24 hrs, 48 timestamps and 1 day
    granu: if granu = 1, then the raw sequence is used 
    else, the sequence will be carried out a granu-step sum. 
    
    output: a dict, keys: (i, j); values: (k, l) and a value, 
            indicating that the (i, j) in target most matches (k, l) in source, and the correlation value
    """
    start_time = time.time()
    src_data, _, _, src_mask, src_poi = load_data(source_name)
    src_smask = src_mask.mean(0) > 0
    src_tmask = src_mask.mean((1, 2)) > 0
    
    tgt_data, _, _, tgt_mask, tgt_poi = load_data(target_name)
    tgt_smask = tgt_mask.mean(0) > 0
    tgt_tmask = tgt_mask.mean((1, 2)) > 0
    
    src_period = (data_sample_range_week[source_name][1] - 48 * period, data_sample_range_week[source_name][1])
    tgt_period = (data_sample_range_week[target_name][1] - 48 * period, data_sample_range_week[target_name][1])
    src_period_arr = src_data[src_period[0]:src_period[1]]
    tgt_period_arr = tgt_data[tgt_period[0]:tgt_period[1]]
    
    if (src_tmask[src_period[0]:src_period[1]]==0).sum() != 0:
        # we currently do not consider shanghai as target
        valid_indices = [src_tmask[src_period[0]:src_period[1]][i] > 0 for i in range(src_period[1] - src_period[0])]
        src_period_arr = src_period_arr[valid_indices, :, :, :]
        tgt_period_arr = tgt_period_arr[valid_indices, :, :, :]
    
    if granu != 1:
        src_period_arr = np.array([src_period_arr[i:i+granu].sum(0) for i in range(0, len(src_period_arr), granu)])
        tgt_period_arr = np.array([tgt_period_arr[i:i+granu].sum(0) for i in range(0, len(tgt_period_arr), granu)])
        
    tgt_lng = tgt_period_arr.shape[1]
    tgt_lat = tgt_period_arr.shape[2]
    src_lng = src_period_arr.shape[1]
    src_lat = src_period_arr.shape[2]
    src_feat = src_period_arr.shape[3]
    tgt_feat = tgt_period_arr.shape[3]
    print(src_period_arr.shape)
    print(tgt_period_arr.shape)
    
    matching_dict = {}
    for itgt in range(tgt_lng):
        print(time.time() - start_time, itgt)
        for jtgt in range(tgt_lat):
            if tgt_smask[itgt][jtgt] == 0:
                continue
            max_corr = -2
            max_corr_lng = 0
            max_corr_lat = 0
            for isrc in range(src_lng):
                for jsrc in range(src_lat):
                    if src_smask[isrc][jsrc] == 0:
                        continue
                    corrs = []
                    for c in range(min(src_feat, tgt_feat)):
                        corr = np.corrcoef(src_period_arr[:, isrc, jsrc, c], tgt_period_arr[:, itgt, jtgt, c])[0][1]
                        if not np.isnan(corr):
                            corrs.append(corr)
                        # else:
                            # print("Nan in tgt (%d, %d) to src (%d, %d), channel %d" % (itgt, jtgt, isrc, jsrc, c))

                    if len(corrs) != 0:
                        corr = np.mean(corrs)
                    else: 
                        corr = 0
                    if corr > max_corr:
                        max_corr = corr
                        max_corr_lng = isrc
                        max_corr_lat = jsrc
            if max_corr == -2:
                print("Possible nan: %d %d"%(itgt, jtgt))
                max_corr == 0
            matching_dict[(itgt, jtgt)] = ((max_corr_lng, max_corr_lat), max_corr)
    filename = "regiontrans_dict/s_%s_t_%s_corr_%d" % (short_dict[source_name], short_dict[target_name], period)
    if granu != 1:
        filename += "_g%d" % granu
    with open(filename, 'w') as outfile:
        outfile.write(str(matching_dict))
    return matching_dict

# Note: DTW is slow for large cities. 
def region_match_seq_dtw(source_name, target_name, period):
    """
    input: 
    source_name, target_name: full length city names
    period: the unit is 24 hrs, 48 timestamps and 1 day
    
    output: a dict, keys: (i, j); values: (k, l) and a value, 
            indicating that the (i, j) in target most matches (k, l) in source, and the correlation value
    """
    start_time = time.time()
    src_data, _, _, src_mask, src_poi = load_data(source_name)
    src_smask = src_mask.mean(0) > 0
    src_tmask = src_mask.mean((1, 2)) > 0
    
    tgt_data, _, _, tgt_mask, tgt_poi = load_data(target_name)
    tgt_smask = tgt_mask.mean(0) > 0
    tgt_tmask = tgt_mask.mean((1, 2)) > 0
    
    src_period = (data_sample_range_week[source_name][1] - 48 * period, data_sample_range_week[source_name][1])
    tgt_period = (data_sample_range_week[target_name][1] - 48 * period, data_sample_range_week[target_name][1])
    src_period_arr = src_data[src_period[0]:src_period[1]]
    tgt_period_arr = tgt_data[tgt_period[0]:tgt_period[1]]
    
    if (src_tmask[src_period[0]:src_period[1]]==0).sum() != 0:
        # we currently do not consider shanghai as target
        valid_indices = [src_tmask[src_period[0]:src_period[1]][i] > 0 for i in range(src_period[1] - src_period[0])]
        src_period_arr = src_period_arr[valid_indices, :, :, :]
        tgt_period_arr = tgt_period_arr[valid_indices, :, :, :]
    
    tgt_lng = tgt_period_arr.shape[1]
    tgt_lat = tgt_period_arr.shape[2]
    src_lng = src_period_arr.shape[1]
    src_lat = src_period_arr.shape[2]
    src_feat = src_period_arr.shape[3]
    tgt_feat = tgt_period_arr.shape[3]
    print(src_period_arr.shape)
    print(tgt_period_arr.shape)
    
    matching_dict = {}
    for itgt in range(tgt_lng):
        print(time.time() - start_time, itgt)
        for jtgt in range(tgt_lat):
            if tgt_smask[itgt][jtgt] == 0:
                continue
            min_dtw = 100
            min_dtw_lng = 0
            min_dtw_lat = 0
            for isrc in range(src_lng):
                for jsrc in range(src_lat):
                    if src_smask[isrc][jsrc] == 0:
                        continue
                    corrs = []
                    for c in range(min(src_feat, tgt_feat)):
                        corrs.append(dtw.distance(src_period_arr[:, isrc, jsrc, c], tgt_period_arr[:, itgt, jtgt, c]))
                    corr = np.mean(corrs)
                    if corr < min_dtw:
                        min_dtw = corr
                        min_dtw_lng = isrc
                        min_dtw_lat = jsrc
            matching_dict[(itgt, jtgt)] = ((min_dtw_lng, min_dtw_lat), min_dtw)
    with open("regiontrans_dict/s_%s_t_%s_dtw_%d" % (short_dict[source_name], short_dict[target_name], period), 'w') as outfile:
        outfile.write(str(matching_dict))
    return matching_dict

def region_match_poi(source_name, target_name, normalize = True, pca = False):
    """
    input: 
    source_name, target_name: full length city names
    period: the unit is 24 hrs, 48 timestamps and 1 day
    
    output: a dict, keys: (i, j); values: (k, l) and a value, 
            indicating that the (i, j) in target most matches (k, l) in source, and the correlation value
    """
    start_time = time.time()
    src_data, _, _, src_mask, src_poi = load_data(source_name)
    src_smask = src_mask.mean(0) > 0
    
    tgt_data, _, _, tgt_mask, tgt_poi = load_data(target_name)
    tgt_smask = tgt_mask.mean(0) > 0
    
    src_lng = src_smask.shape[0]
    src_lat = src_smask.shape[1]
    tgt_lng = tgt_smask.shape[0]
    tgt_lat = tgt_smask.shape[1]
    
    matching_dict = {}
    for itgt in range(tgt_lng):
        print(time.time() - start_time, itgt)
        for jtgt in range(tgt_lat):
            if tgt_smask[itgt][jtgt] == 0:
                continue
            max_sim = -5
            max_sim_lng = 0
            max_sim_lat = 0
            for isrc in range(src_lng):
                for jsrc in range(src_lat):
                    if src_smask[isrc][jsrc] == 0:
                        continue
                    sim = np.dot(src_poi[isrc][jsrc], tgt_poi[itgt][jtgt])
                    if normalize:
                        sim /= (np.linalg.norm(src_poi[isrc][jsrc]) * np.linalg.norm(tgt_poi[itgt][jtgt]))
                        if np.isnan(sim):
                            sim = 0
                    if sim > max_sim:
                        max_sim = sim
                        max_sim_lng = isrc
                        max_sim_lat = jsrc
            if max_sim == -5:
                max_sim = 0
            matching_dict[(itgt, jtgt)] = ((max_sim_lng, max_sim_lat), max_sim)
    filename = "regiontrans_dict/s_%s_t_%s_poi" % (short_dict[source_name], short_dict[target_name])
    if normalize:
        filename += 'cos'
    with open(filename, 'w') as outfile:
        outfile.write(str(matching_dict))
    return matching_dict

if args.metric == 'corr':
    region_match_seq_corr(args.source, args.target, period = args.period, granu= args.granularity)
elif args.metric == 'dtw':
    region_match_seq_dtw(args.source, args.target, period = args.period)
elif args.metric == 'poi':
    region_match_poi(args.source, args.target, normalize = False)
elif args.metric == 'poi-cos':
    region_match_poi(args.source, args.target, normalize = True)
