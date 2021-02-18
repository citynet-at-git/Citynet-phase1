# Citynet-phase1

This is the open-source repository for paper **CityNet: A Multi-city, Multi-modal Dataset for Smart City** for the submission to KDD 2021

## Download data
You can download our dataset via 
- https://drive.google.com/file/d/1ywJpHcOcov09l_eMIhzrDkNvj6b4nzJa/view?usp=sharing or 
- Baiduyun_link (to be updated)

## Run demo code
### Supervise learning
**GNN model for traffic speed prediction**:
```python
python3 code2/run_traffic.py -n [hk|xa|cd] -m [ha|gat|gcn] -l 0.001 -u 3 -b 16
```
**CNN/LSTM model for taxi service prediction**:

**GNN model for taxi service prediction**:
```python
python3 code2/run.py -n [bj|sh|sz|xa|cd|cq] -s [all|demand|supply|inflow|outflow] -m [gat|gcn] -l 0.001 -u 3 -b 16
```
### Transfer learning

**Inter-city transfer learning**:
