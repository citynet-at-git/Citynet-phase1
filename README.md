# Citynet-phase1

This is the open-source repository for the CityNet dataset

## Download data
You can download our dataset via 
- https://drive.google.com/file/d/1ywJpHcOcov09l_eMIhzrDkNvj6b4nzJa/view?usp=sharing or 
- https://pan.baidu.com/s/1Cm-lyV1csZwgN_RQYKBAHg               (**access code:** city)

(You may need to settle the path issues by yourself)

## Dataset architecture
```
citynet-phase1-data
│
└───mask
│   │   mask_{$cname}.npy               #Spatiotemporal masks to invalidate regions with scarce data in various ML tasks
│   
└───poi
|   │   poi_vector_{$cname}.npy         #Processed POI vectors
│   
└───road
|   │   {$cname}_conn_none.npy          # Region-wise road connectivity
|   │   {$cname}_conn.npy               # Region-wise road connectivity without road-type as ‘None’
|   │   {$cname}_road.npy               # Details of all roads in city, including type and corner coordinates
│   
└───taxi
│   └───inflow
│       │   inflow_arr_{$cname}.npy     # Processed region-based inflow data
│   └───outflow
│       │   outflow_arr_{$cname}.npy    # Processed region-based outflow data
│   └───demand
│       │   {$cname}demand.npy          # Processed region-based pick-up data
│   └───supply
│       │   {$cname}supply.npy          # Processed region-based idle-driving time data
│   
└───weather
|   │   Weather_clean_{$full_cname}.npy # Raw meterology log
|   │   weather_{$cname}.npy            # Processed city-based weather data
│   
└───traffic_speed
│   └───{$cname}
│       │   {$cname}_adj.npy            # Adjacency matrix of roads
│       │   {$cname}_id_arr.npy         # Road-id array
│       │   {$cname}_speed.npy          # History of road speed
```
How to load npy file:
```python
import numpy as np
np.load('fname.npy',allow_pickle=True)
```

## Run demo code

### Supervise learning

We put codes for all graph models in folder `code2`, and CNN/LSTM in folder `code1`. 

**GNN model for traffic speed prediction**:
```python
python3 code2/run_traffic.py -n [hk|xa|cd] -m [ha|gat|gcn] -l 0.001 -u 3 -b 16
```
**GNN model for taxi service prediction**:
```python
python3 code2/run.py -n [bj|sh|sz|xa|cd|cq] -s [all|demand|supply|inflow|outflow] -m [gat|gcn] -l 0.001 -u 3 -b 16
```
**CNN/LSTM model for taxi service prediction**:

```python
python3 code1/run.py -n [bj|sh|sz|xa|cd|cq] -s [all|demand|supply|inflow|outflow] -m [CNN|LSTM] -l 0.001 -b 8 -e 75 -w 1
```
**Parameters**: 
- `run.py  `parameters: 
    - `-p`: Path prefix 
    - `-y`: Output length 
    - `-n`: Abbr. name of city
    - `-s`: Name of service
    - `-m`: Name of model
    - `-l`: Learning rate
    - `-b`: Batch size
    - `-u`: Number of layers



### Transfer learning

We put codes for all transfer learning experiments in `code1`. 

**Procedure**:

- First, run

```python
python3 code1/gen_rt_dict.py -s SOURCE -t TARGET -p PERIOD -m METRIC
```

to generate the matching dictionary for RegionTrans. 

- Second, train source models (we currently support LSTM only) via

```python
python3 code1/run.py -n [bj|sh|sz|xa|cd|cq] -s [all|demand|supply|inflow|outflow] -m LSTM -l 0.001 -b 8 -e 75 -w 1
```

using the hyperparameters you like. 

- Third, run fine tuning via 

```python
python3 code1/run_transfer.py -n [bj|sh|sz|xa|cd|cq] -s [all|demand|supply|inflow|outflow] -a finetune -t T --source SOURCE --target TARGET --source-path PATH
```

or run RegionTrans via

```python
python3 code1/run_transfer.py -n [bj|sh|sz|xa|cd|cq] -s [all|demand|supply|inflow|outflow] -a regiontrans -t T --source SOURCE --target TARGET --source-path PATH --loss-w 0.01 --dictpath DICTPATH
```

**Parameters**: 

- `gen_rt_dict.py` parameters: 
    - `-s, -t`: source and target cities, including `beijing, shanghai, shenzhen, chongqing, chengdu, xian`. 
    - `-m`: the metric for matching, including `poi, poi-cos, corr, dtw`. By default we use `poi`. 
        - `poi`: inner product of poi vectors
        - `poi-cos`: cosine similarity of poi vectors
        - `corr`: Pearson correlation between task time series
        - `dtw`: dynamic time warping distance between task time series (warning: this may be very slow)

- `run_transfer.py  `parameters: 
    - `--source, --target`: source and target cities
    - `-t`: Data amount (in number of days) for the target city
    - `--source-path`: path for the model trained on source data
    - `--loss-w`: The loss for region consistency for RegionTrans. By default we use 0.01. 
    - `--dictpath`: The path for the region matching dictionary for RegionTrans





