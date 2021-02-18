import numpy as np
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import os
import logging
import datetime
import convlstm 

def masked_mae_pytorch(ypred, y, spatial_mask):
    """
    ypred and y are both tensors of shape (B, F, lng, lat)
    spatial mask is of shape (lng, lat)
    """
    abs_error = (y - ypred).abs()
    num_valid_points = (y.size(0) * y.size(1) * spatial_mask.sum()).item()
    valid_abs_error = (abs_error * spatial_mask).sum() / num_valid_points
    return valid_abs_error.item(), num_valid_points

def masked_rmse_pytorch(ypred, y, spatial_mask):
    """
    ypred and y are both tensors of shape (B, F, lng, lat)
    spatial mask is of shape (lng, lat)
    """
    sq_error = ((y - ypred) ** 2)
    num_valid_points = (y.size(0) * y.size(1) * spatial_mask.sum()).item()
    valid_ms_error = ((sq_error * spatial_mask).sum() / num_valid_points) ** (0.5)
    return valid_ms_error.item(), num_valid_points


class HA():
    def __init__(self,out_dim=None,mode=1):
        self.out_dim = out_dim
        self.mode = mode
    def predict(self,X):
        if(self.out_dim):
            return np.stack([X.mean(self.mode)]*self.out_dim,-1)
        else:
            return X.mean(self.mode,keepdims=True)


class LR():
    def __init__(self,out_dim = 1,reg =0.1):
        self.lr_reg = Ridge(alpha = reg)

    def train(self,X,Y):
        lags = X.shape[1]
        X = X.reshape(X.shape[0], X.shape[1], -1).transpose((0, 2, 1)).reshape(-1, lags)
        Y = Y.reshape(Y.shape[0], Y.shape[1], -1).transpose((0, 2, 1)).reshape(-1)
        self.lr_reg.fit(X, Y)

    def predict(self,X):
        batch = X.shape[0]
        lags = X.shape[1]
        lng = X.shape[2]
        lat = X.shape[3]
        X = X.reshape(X.shape[0], X.shape[1], -1).transpose((0, 2, 1)).reshape(-1, lags)
        Ypred = self.lr_reg.predict(X)
        Ypred = Ypred.reshape((batch, lng * lat, -1)).transpose((0, 2, 1)).reshape((batch, 1, lng, lat, 1))
        return Ypred


class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, lng = 32, lat = 32):
        # It takes in a four dimensional input (B, C, lng, lat)
        super(ResUnit, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
    
    def forward(self, x):
        z = self.bn1(x)
        z = F.relu(z)
        z = self.conv1(z)
        z = self.bn2(z)
        z = F.relu(z)
        z = self.conv2(z)
        return z + x


class BaseModel(nn.Module):
    """
    Base class for those that need iterative training. 
    """
    def __init__(self, cityname, data_min, data_max, spatial_mask, out_channels, \
            lng_dim, lat_dim, num_epochs, learning_rate, use_ext, ext_dim, weight_decay):
        super(BaseModel, self).__init__()
        self.cityname = cityname
        self.data_min = data_min
        self.data_max = data_max
        lng_dim = spatial_mask.shape[0]
        lat_dim = spatial_mask.shape[1]
        self.spatial_mask = torch.Tensor(spatial_mask).view(1, 1, lng_dim, lat_dim)

        self.out_channels = out_channels
        self.lng_dim = lng_dim
        self.lat_dim = lat_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_ext = use_ext
        self.ext_dim = ext_dim

        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu = torch.device("cuda:0")
            self.spatial_mask = self.spatial_mask.to(self.gpu)

        self.best_rmse = 9999
        self.best_mae = 9999

    def save_model(self, name):
        prefix = self.save_path
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        torch.save(self.state_dict(), prefix  + name + ".pt")
    
    def load_model(self, name):
        prefix = self.save_path
        if not name.endswith(".pt"):
            name += ".pt"
        self.load_state_dict(torch.load(prefix + "/" + name))
    
    def forward(self, X):
        pass

    def predict_loader(self, loader):
        outputs = []  
        for i, tup in enumerate(loader):
            if self.use_ext:
                X, y, ext = tup
            else:
                X, y = tup
            if self.gpu_available:
                X = X.to(self.gpu)
                if self.use_ext:
                    ext = ext.to(self.gpu)
            if self.use_ext:  
                ypred = self.predict((X, ext)).cpu().numpy()
            else:
                ypred = self.predict(X).cpu().numpy()
            outputs.append(ypred)
        return np.concatenate(outputs, axis=0)

    def train_model(self, train_loader, val_loader):
        optimizer = optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        start_time = time.time()
        
        for ep in range(self.num_epochs):
            self.train()
            epoch_loss = []
            if self.out_channels == 1:
                epoch_rmse = []
                epoch_mae = []
            else:
                epoch_rmse = [[] for i in range(self.out_channels)]
                epoch_mae = [[] for i in range(self.out_channels)]
            for i, tup in enumerate(train_loader):
                if self.use_ext:
                    X, y, ext = tup
                else:
                    X, y = tup
                if self.gpu_available:
                    y = y.to(self.gpu)
                    X = X.to(self.gpu)
                    if self.use_ext:
                        ext = ext.to(self.gpu)
                if self.use_ext:
                    z = self.forward((X, ext))
                else:
                    z = self.forward(X)
                loss = ((z - y) **2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                if i % 10 == 0:
                    print("[%.2fs]Epoch %d, Iter %d, Loss %.4f" % (time.time() - start_time, ep, i, loss.item()))
                if self.out_channels == 1:
                    rmse, _ = masked_rmse_pytorch(z, y, self.spatial_mask)
                    mae, _ = masked_mae_pytorch(z, y, self.spatial_mask)
                    epoch_rmse.append(rmse)
                    epoch_mae.append(mae)
                else:
                    for i in range(self.out_channels):
                        rmse, _ = masked_rmse_pytorch(z[:, i:i+1, :, :], y[:, i:i+1, :, :], self.spatial_mask)
                        mae, _ = masked_mae_pytorch(z[:, i:i+1, :, :], y[:, i:i+1, :, :], self.spatial_mask)
                        epoch_rmse[i].append(rmse)
                        epoch_mae[i].append(mae)
                        
            if self.out_channels == 1:                    
                print("[%.2fs]Epoch %d, Train Loss %.4f, RMSE %.4f, MAE %.4f" % (time.time() - start_time, ep, np.mean(epoch_loss), np.mean(epoch_rmse) * (self.data_max - self.data_min), np.mean(epoch_mae) * (self.data_max - self.data_min)))
                logging.info("Epoch %d, Train Loss %.4f, RMSE %.4f, MAE %.4f" % (ep, np.mean(epoch_loss), np.mean(epoch_rmse) * (self.data_max - self.data_min), np.mean(epoch_mae) * (self.data_max - self.data_min)))
                epoch_rmse = []
                epoch_mae = []
            else:
                original_rmse = []
                original_mae = []
                for i in range(self.out_channels):
                    original_rmse.append(np.mean(epoch_rmse[i]) * (self.data_max[i] - self.data_min[i]))
                    original_mae.append(np.mean(epoch_mae[i]) * (self.data_max[i] - self.data_min[i]))
                    epoch_rmse[i] = []
                    epoch_mae[i] = []
                print("[%.2fs]Epoch %d, Train Loss %.4f, RMSE" % (time.time() - start_time, ep, np.mean(epoch_loss)), original_rmse,  "MAE", original_mae)
                logging.info("Epoch %d, Train Loss %.4f, RMSE" % (ep, np.mean(epoch_loss)) + str(original_rmse) +  ", MAE " + str(original_mae))

            epoch_loss = []
            self.eval()
            if self.out_channels == 1:
                rmse, mae = self.evaluate("Validation", val_loader)   
                if rmse < self.best_rmse:
                    self.best_rmse = rmse
                    print("Saving best model...")
                    logging.info("Saving best model...")
                    self.save_model("best")
            else:
                rmse, mae = self.evaluate_multichannel("Validation", val_loader)
                if np.sum(rmse) < self.best_rmse:
                    self.best_rmse = np.sum(rmse)
                    print("Saving best model...")
                    logging.info("Saving best model...")
                    self.save_model("best")
    
    def predict(self, X):
        if self.use_ext:
            X, ext = X
        if self.gpu_available:
            X = X.to(self.gpu)
            if self.use_ext:
                ext = ext.to(self.gpu)
        self.eval()
        with torch.no_grad():
            if self.use_ext: 
                return self.forward((X, ext))
            return self.forward(X)

    def evaluate(self, mode, loader):
        sum_sq_error = 0
        sum_abs_error = 0
        sum_valid_points = 0
        for i, tup in enumerate(loader):
            if self.use_ext:
                X, y, ext = tup
            else:
                X, y = tup
            if self.gpu_available:
                X = X.to(self.gpu)
                y = y.to(self.gpu)
                if self.use_ext:
                    ext = ext.to(self.gpu)
            if self.use_ext:
                ypred = self.predict((X, ext))
            else:
                ypred = self.predict(X)
            rmse, valid_points = masked_rmse_pytorch(ypred, y, self.spatial_mask)
            mae, _ = masked_mae_pytorch(ypred, y, self.spatial_mask)
            sum_valid_points += valid_points
            sum_sq_error += valid_points * (rmse ** 2)
            sum_abs_error += valid_points * mae            
        mse = sum_sq_error / sum_valid_points
        mae = sum_abs_error / sum_valid_points
        print("%s evaulation: rmse %.4f, mae %.4f" % (mode, (mse**0.5) * (self.data_max - self.data_min), mae * (self.data_max - self.data_min)))
        logging.info("%s evaulation: rmse %.4f, mae %.4f" % (mode, (mse**0.5) * (self.data_max - self.data_min), mae * (self.data_max - self.data_min)))
        return mse ** 0.5, mae

    def evaluate_multichannel(self, mode, loader):
        # When out_channel is not 1
        sum_sq_error = np.zeros(self.out_channels)
        sum_abs_error = np.zeros(self.out_channels)
        sum_valid_points = 0
        for i, tup in enumerate(loader):
            if self.use_ext:
                X, y, ext = tup
            else:
                X, y = tup
            if self.gpu_available:
                y = y.to(self.gpu)
                X = X.to(self.gpu)
                if self.use_ext:
                    ext = ext.to(self.gpu)
            if self.use_ext:
                ypred = self.predict((X, ext))
            else:
                ypred = self.predict(X)
            for i in range(self.out_channels):
                rmse, valid_points = masked_rmse_pytorch(ypred[:, i:i+1, :, :], y[:, i:i+1, :, :], self.spatial_mask)
                mae, _ = masked_mae_pytorch(ypred[:, i:i+1, :, :], y[:, i:i+1, :, :], self.spatial_mask)
                sum_sq_error[i]  += valid_points * (rmse ** 2)
                sum_abs_error[i] += valid_points * mae
            sum_valid_points += valid_points

        mse = sum_sq_error / sum_valid_points
        mae = sum_abs_error / sum_valid_points
        print("%s evaluation: rmse" % mode, (mse**0.5) * (self.data_max - self.data_min), "mae", mae * (self.data_max - self.data_min))
        logging.info("%s evaluation: rmse " % mode + str((mse**0.5) * (self.data_max - self.data_min)) + ", mae" + str(mae * (self.data_max - self.data_min)))
        return mse ** 0.5, mae
        

class STResNet(BaseModel):
    def __init__(self, cityname, num_layers, in_channels, out_channels, lng_dim, lat_dim, \
            num_epochs, learning_rate, spatial_mask, data_min, data_max, use_ext, ext_dim, weight_decay):
        # At present we only have one branch: closeness branch. 
        # The model takes in a four dimensional input
        # (B, C, lng, lat)
        super(STResNet, self).__init__(cityname, data_min, data_max, spatial_mask, \
            out_channels, lng_dim, lat_dim, num_epochs, learning_rate, use_ext, ext_dim, weight_decay)
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)
        w = 0.01 * torch.randn((1, out_channels, lng_dim, lat_dim))
        # parameter for hadamard product for fusion
        self.w = nn.Parameter(w, requires_grad = True) 
        self.layers = []
        for i in range(num_layers):
            self.layers.append(ResUnit(in_channels = 64, out_channels = 64, lng = lng_dim, lat = lat_dim))
        self.layers = nn.ModuleList(self.layers)
        self.weight_decay = weight_decay

        self.in_channels = in_channels
        if self.use_ext:
            self.ext_linear1 = nn.Linear(ext_dim, 64)
            self.ext_linear2 = nn.Linear(64, out_channels * lng_dim * lat_dim)

        self.save_path = "../saved_models/STResNet/%s/%s" % (self.cityname, datetime.datetime.now().strftime("%m%d%H%M")) + "/"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        logging.basicConfig(filename = self.save_path + "log", format = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s", level = logging.INFO, filemode = 'w')
        logging.info("Training STResNet on %s with lr %f and weight decay %f"%(self.cityname, self.learning_rate, self.weight_decay))

    def forward(self, X):
        if self.use_ext:
            X, ext = X
        if self.gpu_available:
            X = X.to(self.gpu)
            if self.use_ext:
                ext = ext.to(self.gpu)
        X = self.conv1(X)
        for layer in self.layers:
            X = layer(X)
        X = self.conv2(X)

        Z = self.w * X
        if self.use_ext:
            # ext is originally of dimension (B, lag, dim)
            # we only use the final dimension, i.e. E_{t-1} in the paper

            ext = ext[:, -1, :]
            ext = self.ext_linear1(ext)
            ext = F.relu(ext)
            ext = self.ext_linear2(ext)
            ext = ext.view(-1, self.out_channels, self.lng_dim, self.lat_dim)
            Z = Z + ext
        return torch.sigmoid(Z)
                    
    

class LSTM(BaseModel):
    """
    A variant of DMVST, AAAI 2018. We discarded the semantic view, and use global CNNs instead of local ones for the spatial view. 
    reference https://github.com/huaxiuyao/DMVST-Net
    """
    def __init__(self, cityname, num_feat, num_lag, out_channels, spatial_feat_dim, lng_dim, lat_dim, num_epochs, learning_rate, spatial_mask, data_min, data_max, use_ext, ext_dim, lstm_hidden = 256, weight_decay = 0):
        """
        param: 
        num_spatial_layers: number of convolution layers of the spatial view
        num_feat: number of features for each place at each time. 
        num_lag: number of history lag to use
        out_channels: the output channels, equal to feat
        spatial_feat_dim: the feature dimension of each location for the spatial view, by default 64 in the paper
        lng_dim, lat_dim: map size

        """
        super(LSTM, self).__init__(cityname, data_min, data_max, spatial_mask, out_channels, lng_dim, lat_dim, num_epochs, learning_rate, use_ext, ext_dim, weight_decay)
        # model configuration
        self.num_feat = num_feat
        self.num_lag = num_lag
        self.spatial_feat_dim = spatial_feat_dim
        self.lstm_hidden = lstm_hidden
        
        # training parameters
        self.save_path = "../saved_models/LSTM/" + self.cityname + "/" + datetime.datetime.now().strftime("%m%d%H%M") + '/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # evaluation parameters
        logging.basicConfig(filename = self.save_path + "log", format = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s", level = logging.INFO, filemode = 'w')


        # build spatial view
        # 64 is the default number of filters, which we do not modify
        self.conv1 = nn.Conv2d(self.num_feat, 64, 3, 1, 1)
        self.resunit1 = ResUnit(64, 64, self.lng_dim, self.lat_dim)
        self.resunit2 = ResUnit(64, 64, self.lng_dim, self.lat_dim)
        self.resunit3 = ResUnit(64, 64, self.lng_dim, self.lat_dim)
        self.conv2 = nn.Conv2d(64, self.spatial_feat_dim, 3, 1, 1)
        # although in paper, K convs + 1 FC is used, yet the FC corresponds to cropping a S*S image and apply FC, which is equivalent to another Conv (probably with larger filter size). 

        # build temporal view
        # temporal view consists of LSTM
        # 256 is the hidden size used in the github repo
        if self.use_ext:
            self.ext_fc = nn.Linear(self.ext_dim, 16)
            self.temporal_lstm = nn.LSTM(input_size = self.spatial_feat_dim + 16, hidden_size = self.lstm_hidden)
        else:
            self.temporal_lstm = nn.LSTM(input_size = self.spatial_feat_dim, hidden_size = self.lstm_hidden)

        # build prediction layer
        self.linear = nn.Linear(2 * self.lstm_hidden, self.out_channels)

    
    def forward(self, X):
        """
        X should have shape [B, L * F, lng, lat]
        """
        if self.use_ext:
            X, ext = X
            if self.gpu_available:
                X = X.to(self.gpu)
                ext = ext.to(self.gpu)
                # ext: [B, lag, ext_dim]
        batch_size = X.size(0)
        # forward spatial 
        spatials = []
        for i in range(self.num_lag):
            input = X[:, i * self.num_feat:(i+1) * self.num_feat, :, :]
            z = self.conv1(input)
            z = self.resunit1(z)
            z = self.resunit2(z)
            z = self.resunit3(z)
            z = self.conv2(z)

            # spatial_out should have # lag tensors, each with dimension (B, spatial_output_dim, lng, lat)
            if self.use_ext:
                ext_cur = ext[:, i, :].view(batch_size, self.ext_dim)
                ext_cur = self.ext_fc(ext_cur).view(batch_size, 16, 1, 1)
                ext_cur = ext_cur.repeat(1, 1, self.lng_dim, self.lat_dim)
                # ext_cur has shape (B, self.ext_dim, lng, lat)
                z = torch.cat([z, ext_cur], dim = 1)


            # reshape for temporal view, reshaping into (seq_len, batch, input_feature)
            z = z.permute(0, 2, 3, 1).contiguous()
            if self.use_ext:
                z = z.view(-1, self.spatial_feat_dim + 16).unsqueeze(0)
            else:
                z = z.view(-1, self.spatial_feat_dim).unsqueeze(0)
            spatials.append(z)

            # now x should have shape (1, B * lng * lat, feat)
        temporal_input = torch.cat(spatials, dim = 0)
        # forward temporal view
        temporal_out, (temporal_hid, _) = self.temporal_lstm(temporal_input)
        # temporal_out should be of shape [seq_len, batch, hidden_size] (since we do not use bidirectional)
        # temporal_hid should be of shape [num_layer, batch, hidden_size], the hidden state for the final step
        # temporal_cell is the same shape of temporal_hid. 
        # in our implementation, we use num_layer = 1

        temporal_out = temporal_out[-1 :, :] # we only need the output of the final timestamp, resulting in a tensor of shape (B * lng * lat, hidden)



        temporal_view = torch.cat([
            temporal_out.view(batch_size, self.lng_dim, self.lat_dim, self.lstm_hidden), 
            temporal_hid.view(batch_size, self.lng_dim, self.lat_dim, self.lstm_hidden) 
        ], dim = -1)

        # prediction
        output = torch.sigmoid(self.linear(temporal_view)).permute(0, 3, 1, 2)
        return output



class ConvLSTM(BaseModel):
    """
    The "real" convlstm. The code is taken from https://github.com/ndrplz/ConvLSTM_pytorch
    This is the real convlstm used in RegionTrans. 
    """
    def __init__(self, cityname, num_feat, num_lag, out_channels, lng_dim, lat_dim, num_epochs, learning_rate, spatial_mask, 
                data_min, data_max, use_ext, ext_dim, lstm_hidden = 256, weight_decay = 0):
        super(ConvLSTM, self).__init__(cityname, data_min, data_max, spatial_mask, out_channels, lng_dim, lat_dim, num_epochs, learning_rate, use_ext, ext_dim, weight_decay)
        # model configuration
        self.lstm_hidden = lstm_hidden
        self.num_feat = num_feat
        self.num_lag = num_lag
        
        # training parameters
        self.save_path = "../saved_models/ConvLSTM/" + self.cityname + "/" + datetime.datetime.now().strftime("%m%d%H%M") + '/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # evaluation parameters
        logging.basicConfig(filename = self.save_path + "log", format = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s", level = logging.INFO, filemode = 'w')

        # build ConvLSTM
        # by default we use 2 layer convlstm
        self.convlstm = convlstm.ConvLSTM(input_dim = self.num_feat, hidden_dim = self.lstm_hidden, kernel_size = [(5, 5), (5, 5)], num_layers = 2, batch_first = True, bias = True, return_all_layers = False)
        
        
        if self.use_ext:
            self.ext_fc = nn.Linear(self.ext_dim, 16)
            self.conv1 = nn.Conv2d(self.lstm_hidden + 16, self.lstm_hidden, kernel_size = 3, padding = 1, stride = 1)
        else:
            self.conv1 = nn.Conv2d(self.lstm_hidden, self.lstm_hidden, kernel_size = 3, padding = 1, stride = 1)
        self.conv2 = nn.Conv2d(self.lstm_hidden, self.out_channels, kernel_size = 3, padding = 1, stride = 1)

    def forward(self, X):
        """
        The input X should have shape [B, T, C, W, H]
        where T is the number of lags
        and C is the number of channels 

        the ext should have shape [B, T, ext_dim]
        """
        if self.use_ext:
            X, ext = X
            if self.gpu_available:
                X = X.to(self.gpu)
                ext = ext.to(self.gpu)
        else:
            if self.gpu_available:
                X = X.to(self.gpu)

        out = self.convlstm(X)
        
        xrep = out[0][0][:, -1, :, :, :]
        # xrep in the shape of [B, hid, W, H]
        if self.use_ext:
            ext = ext[:, -1, :]
            ext = self.ext_fc(ext)
            # We expand it into [B, ext_dim, W, H]
            ext = ext.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, xrep.shape[2], xrep.shape[3])
            # Now ext has shape [B, ext_dim, W, H]

            combined = torch.cat([xrep, ext], dim = 1)
        else:
            combined = xrep

        hidden = self.conv1(combined)
        hidden = F.relu(hidden)
        out = self.conv2(hidden)
        # out: [B, out_channel, W, H]
        return torch.sigmoid(out)

        

class GCN(nn.Module):
    def __init__(self, cityname, num_layers, in_channel, out_channel, hidden_dim, num_epochs, learning_rate, data_min, data_max, a_euc, a_poi, a_con, act, weight_decay):
        super(GCN, self).__init__()
        # model configuratoin
        self.num_layers = num_layers
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_dim = hidden_dim

        # training configuration
        self.cityname = cityname
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.save_path = '../saved_models/GCN' + self.cityname + "/" + datetime.datetime.now().strftime("%m%d%H%M") + '/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.a_euc = a_euc
        self.a_poi = a_poi
        self.a_con = a_con       
        self.A = (self.a_euc + self.a_poi + self.a_con) / 3 
        # The A is symmetric, i.e. the graph is undirected. 
        self.weight_decay = weight_decay

        # evaluation parameters
        # Graph models do not have lng, lat, and mask
        self.best_rmse = 999
        self.best_mae = 999
        self.data_min = data_min
        self.data_max = data_max

        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu = torch.device('cuda:0')
            self.A = self.A.to(self.gpu)

        logging.basicConfig(filename = self.save_path + "log", format = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s", level = logging.INFO, filemode = 'w')


        # build model
        if self.gpu_available:
            self.Atilde = torch.eye(self.a_euc.size(0)).to(self.gpu) + self.A
        else:
            self.Atilde = torch.eye(self.a_euc.size(0)) + self.A
        D = self.Atilde.sum(1)
        self.A = D.view(1, -1).pow(-0.5) * self.Atilde * D.view(-1, 1).pow(-0.5)
        # self.A = torch.eye(self.Atilde.size(0))
        if self.gpu_available:
            self.A = self.A.to(self.gpu)

        self.layers = []
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(nn.Linear(self.in_channel, self.hidden_dim))
            elif i == self.num_layers - 1:
                self.layers.append(nn.Linear(self.hidden_dim, self.out_channel))
            else:
                self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers = nn.ModuleList(self.layers)
        self.act = {
            "relu":F.relu, 
            "leakyrelu":F.leaky_relu, 
            "tanh": torch.tanh
        }[act]
        
    def forward(self, X):
        if self.gpu_available:
            X = X.to(self.gpu)
        # X should have shape (B, num_nodes, feat)
        for i in range(self.num_layers):
            # H^i+1 = AXW
            X = X.permute(0, 2, 1).contiguous()
            # print(X.shape)
            # print(self.A.shape)
            propagated_X = torch.matmul(X, self.A)
            transformed_X = self.layers[i](propagated_X.permute(0, 2, 1).contiguous())
            X = transformed_X
            if i != self.num_layers - 1:
                X = self.act(X)
        return torch.sigmoid(X)
    
    def train_model(self, train_loader, val_loader):
        optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
        start_time = time.time()

        for ep in range(self.num_epochs):
            self.train()
            epoch_loss = []
            if self.out_channel == 1:
                epoch_rmse = []
                epoch_mae = []
            else:
                epoch_rmse = [[] for i in range(self.out_channel)]
                epoch_mae = [[] for i in range(self.out_channel)]
            for i, (X, y) in enumerate(train_loader):
                if self.gpu_available:
                    y = y.to(self.gpu)
                    X = X.to(self.gpu)
                z = self.forward(X)
                loss = ((z-y) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    print("[%.2fs]Epoch %d, Iter %d, Loss %.4f" % (time.time() - start_time, ep, i, loss.item()))
                epoch_loss.append(loss.item())
                if self.out_channel == 1:
                    rmse = loss.pow(1/2).item()
                    mae = (z-y).abs().mean().item()
                    epoch_rmse.append(rmse)
                    epoch_mae.append(mae)
                else:
                    for i in range(self.out_channel):
                        rmse = ((z - y)[:, :, i:i+1].pow(2)).mean().pow(1/2).item()
                        mae = ((z-y)[:, :, i:i+1].abs()).mean().item()
                        epoch_rmse[i].append(rmse)
                        epoch_mae[i].append(mae)
            if self.out_channel == 1:
                print("[%.2fs]Epoch %d, Train Loss %.4f, RMSE %.4f, MAE %.4f" % (time.time() - start_time, ep, np.mean(epoch_loss), np.mean(epoch_rmse) * (self.data_max - self.data_min), np.mean(epoch_mae) * (self.data_max - self.data_min)))
                logging.info("Epoch %d, Train Loss %.4f, RMSE %.4f, MAE %.4f" % (ep, np.mean(epoch_loss), np.mean(epoch_rmse) * (self.data_max - self.data_min), np.mean(epoch_mae) * (self.data_max - self.data_min)))
                epoch_rmse = []
                epoch_mae = []        
            else:
                original_rmse = []
                original_mae = []
                for i in range(self.out_channel):
                    original_rmse.append(np.mean(epoch_rmse[i]) * (self.data_max[i] - self.data_min[i]))
                    original_mae.append(np.mean(epoch_mae[i]) * (self.data_max[i] - self.data_min[i]))
                    epoch_rmse[i] = []
                    epoch_mae[i] = []
                print("[%.2fs]Epoch %d, Train Loss %.4f, RMSE" % (time.time() - start_time, ep, np.mean(epoch_loss)), original_rmse,  "MAE", original_mae)
                logging.info("Epoch %d, Train Loss %.4f, RMSE" % (ep, np.mean(epoch_loss)) + str(original_rmse) +  ", MAE " + str(original_mae))
            epoch_loss = []

            self.eval()
            if self.out_channel == 1:
                rmse, mae = self.evaluate("Validation", val_loader)
                if rmse < self.best_rmse:
                    self.best_rmse = rmse
                    print("Saving best model...")
                    logging.info("Saving best model...")
                    self.save_model('best')
            else:
                rmse, mae = self.evaluate_multichannel("Validation", val_loader)
                if np.sum(rmse) < self.best_rmse:
                    self.best_rmse = np.sum(rmse)
                    print("Saving best model...")
                    logging.info("Saving best model...")
                    self.save_model('best')

    def predict(self, X):
        if self.gpu_available:
            X = X.to(self.gpu)
        self.eval()
        with torch.no_grad():
            return self.forward(X)

    def predict_loader(self, loader):
        outputs = []
        for i, (X, y) in enumerate(loader):
            if self.gpu_available:
                X = X.to(self.gpu)
            ypred = self.predict(X).cpu().numpy()
            outputs.append(ypred)
        
        return np.concatenate(outputs, axis = 0)
    
    def evaluate(self, mode, loader):
        sum_sq_error = 0
        sum_abs_error = 0
        sum_valid_points = 0
        
        for i, (X, y) in enumerate(loader):
            if self.gpu_available:
                X = X.to(self.gpu)
                y = y.to(self.gpu)
            ypred = self.predict(X) 
            batch_size = X.size(0)
            num_nodes = X.size(1)
            sum_valid_points += batch_size * num_nodes
            sum_abs_error += (ypred - y).abs().sum()
            sum_sq_error += (ypred - y).pow(2).sum()

        mse = sum_sq_error / sum_valid_points
        mae = sum_abs_error / sum_valid_points
        print("%s evaulation: rmse %.4f, mae %.4f" % (mode, (mse**0.5) * (self.data_max - self.data_min), mae * (self.data_max - self.data_min)))
        logging.info("%s evaulation: rmse %.4f, mae %.4f" % (mode, (mse**0.5) * (self.data_max - self.data_min), mae * (self.data_max - self.data_min)))
        return mse ** 0.5, mae
    
    def evaluate_multichannel(self, mode, loader):
        sum_sq_error = np.zeros(self.out_channel)
        sum_abs_error = np.zeros(self.out_channel)
        sum_valid_points = 0

        for i, tup in enumerate(loader):
            X, y = tup
            if self.gpu_available:
                X = X.to(self.gpu)
                y = y.to(self.gpu)
                
            ypred = self.predict(X)
            batch_size = X.size(0)
            num_nodes = X.size(1)
            sum_valid_points += batch_size * num_nodes
            for i in range(self.out_channel):
                sum_abs_error[i] += (ypred[:, :, i:i+1] - y[:, :, i:i+1]).abs().sum()
                sum_sq_error[i] += (ypred[:, :, i:i+1] - y[:, :, i:i+1]).pow(2).sum()
        mse = sum_sq_error / sum_valid_points
        mae = sum_abs_error / sum_valid_points
        print("%s evaluation: rmse" % mode, (mse**0.5) * (self.data_max - self.data_min), "mae", mae * (self.data_max - self.data_min))
        logging.info("%s evaluation: rmse " % mode + str((mse**0.5) * (self.data_max - self.data_min)) + ", mae" + str(mae * (self.data_max - self.data_min)))
        return mse ** 0.5, mae

                
    def save_model(self, name):
        prefix = self.save_path
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        torch.save(self.state_dict(), prefix + name + ".pt")
    
    def load_model(self, name):
        prefix = self.save_path
        if not name.endswith(".pt"):
            name += ".pt"
        self.load_state_dict(torch.load(prefix + "/" + name))

def fix_bn(m):
    """
    Fix batchnorm within a network
    """
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()

def unfix_bn(m):
    """
    unfix batchnorm within a network
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

class RegionTrans_base(nn.Module):
    """
    A base class for RegionTrans, IJCAI 2019
    Cross-City Transfer Learning for Deep Spatio-Temporal Prediction, L. Wang et al. 
    """
    def __init__(self, source_name, target_name, num_feat, num_lag, out_channels, num_epochs, learning_rate, \
        source_smask, target_smask, target_datamin, target_datamax, use_ext, ext_dim, loss_w, 
        matching_dict_path, lstm_hidden):
        super(RegionTrans_base, self).__init__()
        self.source_name = source_name
        self.target_name = target_name
        self.num_feat = num_feat
        self.num_lag  = num_lag
        self.out_channels = out_channels
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.source_smask = torch.Tensor(source_smask).view(1, 1, source_smask.shape[0], source_smask.shape[1])
        self.target_smask = torch.Tensor(target_smask).view(1, 1, target_smask.shape[0], target_smask.shape[1])
        self.target_datamin = target_datamin
        self.target_datamax = target_datamax
        self.use_ext = use_ext
        self.ext_dim = ext_dim
        self.loss_w = loss_w
        self.lstm_hidden = lstm_hidden 

        self.best_rmse = 9999
        self.best_mae = 9999
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu = torch.device("cuda:0")
            self.source_smask = self.source_smask.to(self.gpu)
            self.target_smask = self.target_smask.to(self.gpu)
        # read in the matching dict
        with open(matching_dict_path, 'r') as infile:
            self.matching_dict = eval(infile.read())
        self.matching_indices = []
        self.matching_weight = []
        max_weight = 0
        for item in self.matching_dict:
            item = self.matching_dict[item]
            coord = item[0]
            weight = item[1]
            max_weight = max(max_weight, weight)
            self.matching_indices.append(coord[0] * source_smask.shape[1] + coord[1])
            self.matching_weight.append(weight)
            # The dicts are generated in a row-first manner, and therefore, matching_indices will be 
            # a match from target to its best match source region
        self.matching_indices = torch.Tensor(self.matching_indices).long()
        self.matching_weight = torch.Tensor(self.matching_weight)
        if max_weight > 1 and "dtw" in matching_dict_path:
            print("Normalizing...")
            self.matching_weight /= max_weight
            self.matching_weight = 1 - self.matching_weight
        elif max_weight > 1 and matching_dict_path.endswith("poi"):
            print("Normalizing...")
            self.matching_weight /= max_weight
        if self.gpu_available:
            self.matching_indices = self.matching_indices.to(self.gpu)
            self.matching_weight = self.matching_weight.to(self.gpu)
    
    def forward_region(self, X):
        pass

    def forward_from_region(self, X):
        pass

    def forward(self, X):
        pass

    def save_model(self, name):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.state_dict(), self.save_path + name + ".pt")

    def load_model(self, name):
        if not name.endswith(".pt"):
            name += '.pt'
        self.load_state_dict(torch.load(self.save_path + '/' + name))           

    def predict(self, X):
        if self.use_ext:
            X, ext = X
        if self.gpu_available:
            X = X.to(self.gpu)
            if self.use_ext:
                ext = ext.to(self.gpu)
        self.eval()
        with torch.no_grad():
            if self.use_ext:
                return self.forward((X, ext))
            else:
                return self.forward(X)

    def predict_loader(self, loader):
        outputs = []
        for i, tup in enumerate(loader):
            if self.use_ext:
                X, y, ext = tup
            else:
                X, y = tup
            if self.gpu_available:
                X = X.to(self.gpu)
                if self.use_ext:
                    ext = ext.to(self.gpu)
            if self.use_ext:
                ypred = self.predict((X, ext)).cpu().numpy()
            else:
                ypred = self.predict(X).cpu().numpy()
            outputs.append(ypred)
        return np.concatenate(outputs, axis = 0)

    def evaluate(self, mode, loader):
        sum_valid_points = 0
        sum_sq_error = 0
        sum_abs_error = 0
        for i, tup in enumerate(loader):
            if self.use_ext:
                X, y, ext = tup
            else:
                X, y = tup
            if self.gpu_available:
                X = X.to(self.gpu)
                y = y.to(self.gpu)
                if self.use_ext:
                    ext = ext.to(self.gpu)
            if self.use_ext:
                ypred = self.predict((X, ext))
            else:
                ypred = self.predict(X)
            rmse, valid_points = masked_rmse_pytorch(ypred, y, self.target_smask)
            mae, _ = masked_mae_pytorch(ypred, y, self.target_smask)
            sum_valid_points += valid_points
            sum_sq_error += valid_points * (rmse ** 2)
            sum_abs_error += valid_points * mae
        mae = sum_abs_error / sum_valid_points
        mse = sum_sq_error / sum_valid_points
        print("%s evaulation: rmse %.4f, mae %.4f" % (mode, (mse**0.5) * (self.target_datamax - self.target_datamin), mae * (self.target_datamax - self.target_datamin)))
        logging.info("%s evaulation: rmse %.4f, mae %.4f" % (mode, (mse**0.5) * (self.target_datamax - self.target_datamin), mae * (self.target_datamax - self.target_datamin)))
        return mse ** 0.5, mae

class RegionTrans_LSTM(RegionTrans_base):
    """
    Implements RegionTrans, IJCAI 2019. 
    Cross-City Transfer Learning for Deep Spatio-Temporal Prediction, L. Wang et al. 
    The backbone is the LSTM. 
    """
    def __init__(self, source_name, target_name, num_feat, num_lag, out_channels, spatial_feat_dim, \
        num_epochs, learning_rate, source_smask, target_smask, 
        target_datamin, target_datamax, use_ext, ext_dim, loss_w, matching_dict_path, lstm_hidden = 256):
        """
        param:
        source_name, target_name: short names (2 letters), used to load in the matching dict
        num_feat: number of features (or input channels) to predict. At present, we require src and target to have the same feat
        num_lag: number of history lag
        out_channels should be equal to num_feat
        spatial_feat_dim: the feature dim for each location for the spatial view. by default 64. We do not tune it. 
        """

        super(RegionTrans_LSTM, self).__init__(source_name, target_name, num_feat, num_lag, out_channels, num_epochs, learning_rate, \
            source_smask, target_smask, target_datamin, target_datamax, use_ext, ext_dim, loss_w, matching_dict_path, lstm_hidden)

        self.spatial_feat_dim = spatial_feat_dim

        # training parameters
        self.save_path = "../saved_models/RegionTrans_LSTM/%s_to_%s/%s/"%(self.source_name, self.target_name, datetime.datetime.now().strftime("%m%d%H%M%S"))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        logging.basicConfig(filename = self.save_path + "log", \
            format = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s", 
            level = logging.INFO, 
            filemode = 'w')

        # build spatial view
        # 64 is the default number of filters
        self.conv1 = nn.Conv2d(self.num_feat, 64, 3, 1, 1)
        self.resunit1 = ResUnit(64, 64)
        self.resunit2 = ResUnit(64, 64)
        self.resunit3 = ResUnit(64, 64)
        self.conv2 = nn.Conv2d(64, self.spatial_feat_dim, 3, 1, 1)

        if self.use_ext:
            self.ext_fc = nn.Linear(ext_dim, 16)
            self.temporal_lstm = nn.LSTM(input_size = self.spatial_feat_dim + 16, hidden_size = self.lstm_hidden)
        else:
            self.temporal_lstm = nn.LSTM(input_size = self.spatial_feat_dim, hidden_size = self.lstm_hidden)

        # prediction layer
        self.linear = nn.Linear(2 * self.lstm_hidden, self.out_channels)
    
    def forward_region(self, X):
        """
        This function forwards to the region representation
        X is a tensor with shape [B, lag * channel, lng, lat]
        """
        if self.use_ext:
            X, ext = X
            if self.gpu_available:
                X = X.to(self.gpu)
                ext = ext.to(self.gpu)
                # ext: [B, lag, ext_dim]
        batch_size = X.size(0)
        lng = X.size(2)
        lat = X.size(3)
        spatials = []
        for i in range(self.num_lag):
            input = X[:, i * self.num_feat:(i+1) * self.num_feat, :, :]
            z = self.conv1(input)
            z = self.resunit1(z)
            z = self.resunit2(z)
            z = self.resunit3(z)
            z = self.conv2(z)

            if self.use_ext:
                ext_cur = ext[:, i, :].view(batch_size, self.ext_dim)
                ext_cur = self.ext_fc(ext_cur).view(batch_size, 16, 1, 1)
                ext_cur = ext_cur.repeat(1, 1, lng, lat)
                # now ext_cur has shape (B, self.ext_dim, lng, lat)
                z = torch.cat([z, ext_cur], dim = 1)
            
            # reshape for temporal view, reshaping into (seq_len, batch, input_feature)
            z = z.permute(0, 2, 3, 1).contiguous()
            if self.use_ext:
                z = z.view(-1, self.spatial_feat_dim + 16).unsqueeze(0)
            else:
                z = z.view(-1, self.spatial_feat_dim).unsqueeze(0)
            spatials.append(z)
            # now each z should have shape (1, B * lng * lat, feat)
        
        temporal_input = torch.cat(spatials, dim = 0)
        temporal_out, (temporal_hid, _) = self.temporal_lstm(temporal_input)

        temporal_out = temporal_out[-1:, :]
        temporal_view = torch.cat([
            temporal_out.view(batch_size, lng, lat, self.lstm_hidden),
            temporal_hid.view(batch_size, lng, lat, self.lstm_hidden)
        ], dim = -1)
        
        return temporal_view
        # temporal view is the region representation, which will be of shape [B, lng, lat, f]
    

    
    def forward_from_region(self, region):
        """
        This function forwards from region representation to prediction
        """
        output = torch.sigmoid(self.linear(region)).permute(0, 3, 1, 2)
        return output

    def forward(self, X):
        """
        this function forwards to the final prediction value
        """
        region_rep = self.forward_region(X)
        output = self.forward_from_region(region_rep)
        return output

    def train_model(self, train_loader, val_loader):
        """
        note: trainloader and valloader are different
        trainloader contains source, while val_loader does not
        """
        optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
        start_time = time.time()

        for ep in range(self.num_epochs):
            self.train()
            epoch_target_loss = []
            epoch_cons_loss = []
            if self.out_channels == 1:
                epoch_rmse = []
                epoch_mae = []
            else:
                epoch_rmse = [[] for i in range(self.out_channels)]
                epoch_mae = [[] for i in range(self.out_channels)]
            
            for i, tup in enumerate(train_loader):
                if self.use_ext:
                    (X_src, ext_src, X_tgt, y_tgt, ext_tgt) = tup
                else:
                    (X_src, X_tgt, y_tgt) = tup
                if self.gpu_available:
                    X_src = X_src.to(self.gpu)
                    X_tgt = X_tgt.to(self.gpu)
                    y_tgt = y_tgt.to(self.gpu)
                    if self.use_ext:    
                        ext_tgt = ext_tgt.to(self.gpu)
                        ext_src = ext_src.to(self.gpu)
                batch_size = X_tgt.size(0)
                if self.use_ext:
                    self.apply(fix_bn)
                    region_rep_src = self.forward_region((X_src, ext_src))
                    self.apply(unfix_bn)
                    region_rep_tgt = self.forward_region((X_tgt, ext_tgt))
                else:
                    self.apply(fix_bn)
                    region_rep_src = self.forward_region(X_src)
                    self.apply(unfix_bn)
                    region_rep_tgt = self.forward_region(X_tgt)
                predict = self.forward_from_region(region_rep_tgt)
                loss_target = ((predict - y_tgt) ** 2).mean()

                # build the region consistency loss
                valid_target = region_rep_tgt.view(batch_size, -1, region_rep_tgt.size(3))[:, self.target_smask.view(-1).bool(), :]
                match_source = region_rep_src.view(batch_size, -1, region_rep_src.size(3))[:, self.matching_indices, :]
                # valid_target and match_source should both have [B, # valid_points, feat] size
                
                loss_cons = (((valid_target - match_source) ** 2).mean(2).mean(0) * self.matching_weight).mean()

                loss = (1-self.loss_w) * loss_target + self.loss_w * loss_cons
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_target_loss.append(loss_target.item())
                epoch_cons_loss.append(loss_cons.item())
                if self.out_channels == 1:
                    rmse, _ = masked_rmse_pytorch(predict.detach(), y_tgt, self.target_smask)
                    mae, _ = masked_mae_pytorch(predict.detach(), y_tgt, self.target_smask)
                    epoch_rmse.append(rmse)
                    epoch_mae.append(mae)
                else:
                    for i in range(self.out_channels):
                        rmse, _ = masked_rmse_pytorch(predict[:, i:i+1, :, :], y_tgt[:, i:i+1, :, :], self.target_smask)
                        mae, _ = masked_mae_pytorch(predict[:, i:i+1, :, :], y_tgt[:, i:i+1, :, :], self.target_smask)
                        epoch_rmse[i].append(rmse)
                        epoch_mae[i].append(mae)
                if i % 50 == 0:
                    print("[%.2fs]Epoch %d, Iter %d, Loss Target %.4f, Loss Consistency %.4f" %\
                         (time.time() - start_time, ep, i, loss_target.item(), loss_cons.item()))
            if self.out_channels == 1:
                print("[%.2fs]Epoch %d, Train Loss Target %.4f, Train Loss Consistency %.4f, RMSE %.4f, MAE %.4f" % \
                    (time.time() - start_time, ep, np.mean(epoch_target_loss), np.mean(epoch_cons_loss), np.mean(epoch_rmse) * (self.target_datamax - self.target_datamin), np.mean(epoch_mae) * (self.target_datamax - self.target_datamin)))
                logging.info("Epoch %d, Train Loss Target %.4f, Train Loss Consistency %.4f, RMSE %.4f, MAE %.4f" % \
                    (ep, np.mean(epoch_target_loss), np.mean(epoch_cons_loss), np.mean(epoch_rmse) * (self.target_datamax - self.target_datamin), np.mean(epoch_mae) * (self.target_datamax - self.target_datamin)))
                epoch_rmse = []
                epoch_mae = []
            else:
                original_rmse = []
                original_mae = []
                for i in range(self.out_channels):
                    original_rmse.append(np.mean(epoch_rmse[i]) * (self.target_datamax[i] - self.target_datamin[i]))
                    original_mae.append(np.mean(epoch_mae[i]) * (self.target_datamax[i] - self.target_datamin[i]))
                    epoch_rmse[i] = []
                    epoch_mae[i] = []
                print("[%.2fs]Epoch %d, Train Loss Target %.4f, Train Loss Consistency %.4f, RMSE" % \
                    (time.time() - start_time, ep, np.mean(epoch_target_loss), np.mean(epoch_cons_loss)), original_rmse,  "MAE", original_mae)
                logging.info("Epoch %d, Train Loss Target %.4f, Train Loss Consistency %.4f, RMSE" % \
                    (ep, np.mean(epoch_target_loss), np.mean(epoch_cons_loss)) + str(original_rmse) +  ", MAE " + str(original_mae))
            epoch_cons_loss = []
            epoch_target_loss = []
            self.eval()
            if self.out_channels == 1:
                rmse, mae = self.evaluate("Validation", val_loader)
                if rmse < self.best_rmse:
                    self.best_rmse = rmse
                    logging.info("Saving best model")
                    print("Saving best model...")
                    self.save_model("best")
            else:
                rmse, mae = self.evaluate_multichannel("Validation", val_loader)
                if np.sum(rmse) < self.best_rmse:
                    self.best_rmse = np.sum(rmse)
                    print("Saving best model...")
                    logging.info("Saving best model...")
                    self.save_model("best")
    
    def evaluate_multichannel(self, mode, loader):
        """
        do evaluation when out_channel is not 1
        """
        sum_sq_error = np.zeros(self.out_channels)
        sum_abs_error = np.zeros(self.out_channels)
        sum_valid_points = 0
        for i, tup in enumerate(loader):
            if self.use_ext:
                X, y, ext = tup
            else:
                X, y = tup
            if self.gpu_available:
                X = X.to(self.gpu)
                y = y.to(self.gpu)
                if self.use_ext:
                    ext = ext.to(self.gpu)
            if self.use_ext:
                ypred = self.predict((X, ext))
            else:
                ypred = self.predict(X)
            for j in range(self.out_channels):
                rmse, valid_points = masked_rmse_pytorch(ypred[:, j:j+1, :, :], y[:, j:j+1, :, :], self.target_smask)
                mae, _ = masked_mae_pytorch(ypred[:, j:j+1, :, :], y[:, j:j+1, :, :], self.target_smask)
                sum_sq_error[j] += valid_points * (rmse ** 2)
                sum_abs_error[j] += valid_points * (mae)
            sum_valid_points += valid_points
        mse = sum_sq_error / sum_valid_points
        mae = sum_abs_error / sum_valid_points
        print("%s evaluation: rmse" % mode, (mse**0.5) * (self.target_datamax - self.target_datamin), "mae", mae * (self.target_datamax - self.target_datamin))
        logging.info("%s evaluation: rmse " % mode + str((mse**0.5) * (self.target_datamax - self.target_datamin)) + ", mae" + str(mae * (self.target_datamax - self.target_datamin)))
        return mse ** 0.5, mae

    

class RegionTrans_ConvLSTM(RegionTrans_base):
    """
    Implements RegionTrans, IJCAI 2019
    Switch the backbone to ConvLSTM
    Note that this module has no batch normalization, and therefore the fix_bn is not necessary. 
    """
    def __init__(self, source_name, target_name, num_feat, num_lag, out_channels, num_epochs, learning_rate, \
        source_smask, target_smask, target_datamin, target_datamax, use_ext, ext_dim, loss_w, matching_dict_path,
        lstm_hidden = 64):

        super(RegionTrans_ConvLSTM, self).__init__(source_name, target_name, num_feat, num_lag, out_channels, num_epochs,\
            learning_rate, source_smask, target_smask, target_datamin, target_datamax, use_ext, ext_dim, loss_w, matching_dict_path, lstm_hidden)

        self.save_path = "../saved_models/RegionTrans_ConvLSTM/%s_to_%s/%s" % (self.source_name, self.target_name, datetime.datetime.now().strftime("%m%d%H%M%S"))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.loss_w = loss_w

        logging.basicConfig(filename = self.save_path + "/log", \
            format = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s", 
            level = logging.INFO, 
            filemode = 'w')
        
        
        self.convlstm = convlstm.ConvLSTM(input_dim = self.num_feat, hidden_dim = self.lstm_hidden, kernel_size = [(5, 5), (5, 5)], num_layers = 2, 
            batch_first = True, bias = True, return_all_layers = False)

        if self.use_ext: 
            self.ext_fc = nn.Linear(self.ext_dim, 16) 
            self.conv1 = nn.Conv2d(self.lstm_hidden + 16, self.lstm_hidden, kernel_size = 3, padding = 1, stride = 1)
        else:
            self.conv1 = nn.Conv2d(self.lstm_hidden, self.lstm_hidden, kernel_size = 3, padding = 1, stride = 1)
        self.conv2 = nn.Conv2d(self.lstm_hidden, self.out_channels, kernel_size = 3, padding = 1, stride = 1)

    
    def forward_region(self, X):
        if self.gpu_available:
            X = X.to(self.gpu)
        region_out = self.convlstm(X)[0][0][:, -1, :, :, :]
        return region_out
    
    def forward_from_region(self, region, ext = None):
        if self.use_ext:
            if self.gpu_available:
                ext = ext.to(self.gpu)
            ext = ext[:, -1, :]
            ext = self.ext_fc(ext)
            ext = ext.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, region.shape[-2], region.shape[-1])
            combined = torch.cat([region, ext], dim = 1)
        else:
            combined = region
        
        hidden = self.conv1(combined)
        hidden = F.relu(hidden)
        out = self.conv2(hidden)
        return torch.sigmoid(out)
    
    def forward(self, X):
        if self.use_ext:
            X, ext = X
            if self.gpu_available:
                X = X.to(self.gpu)
                ext = ext.to(self.gpu)
        else:
            if self.gpu_available:
                X = X.to(self.gpu)
        region = self.forward_region(X)
        if self.use_ext:
            out = self.forward_from_region(region, ext)
        else:
            out = self.forward_from_region(region)
        return out
    
    def train_model(self, train_loader, val_loader):
        """
        trainloader and validloader are different
        trainloader contains source, while validloader does not
        At present, we first support single task transfer
        """
        optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
        start_time = time.time()

        for ep in range(self.num_epochs):
            self.train()
            epoch_target_loss = []
            epoch_cons_loss = []
            if self.out_channels == 1:
                epoch_rmse = []
                epoch_mae = []
            else:
                raise NotImplementedError("Multi-task not implemented yet")
            for i, tup in enumerate(train_loader):
                if self.use_ext:
                    (X_src, ext_src, X_tgt, y_tgt, ext_tgt) = tup
                else:
                    (X_src, X_tgt, y_tgt) = tup

                if self.gpu_available:
                    X_src = X_src.to(self.gpu)
                    X_tgt = X_tgt.to(self.gpu)
                    y_tgt = y_tgt.to(self.gpu)
                    if self.use_ext:
                        ext_tgt = ext_tgt.to(self.gpu)
                        ext_src = ext_src.to(self.gpu)
                batch_size = X_tgt.size(0)
                region_rep_src = self.forward_region(X_src)
                region_rep_tgt = self.forward_region(X_tgt)
                # both region reps should have shape [B, hid, W, H]
            
                if self.use_ext:
                    predict = self.forward_from_region(region_rep_tgt, ext_tgt)
                else:
                    predict = self.forward_from_region(region_rep_tgt)
                loss_target = ((predict - y_tgt) ** 2).mean()

                # build consistency loss
                valid_target = region_rep_tgt.permute((0, 2, 3, 1)).contiguous().view(batch_size, -1, region_rep_tgt.shape[1])[:, self.target_smask.view(-1).bool(), :]
                match_source = region_rep_src.permute((0, 2, 3, 1)).contiguous().view(batch_size, -1, region_rep_src.shape[1])[:, self.matching_indices, :]
                loss_cons = (((valid_target - match_source) ** 2).mean(2).mean(0) * self.matching_weight).mean()

                loss = (1 - self.loss_w) * loss_target + self.loss_w * loss_cons

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_target_loss.append(loss_target.item())
                epoch_cons_loss.append(loss_cons.item())
                if self.out_channels == 1:
                    rmse, _ = masked_rmse_pytorch(predict.detach(), y_tgt, self.target_smask)
                    mae, _ = masked_mae_pytorch(predict.detach(), y_tgt, self.target_smask)
                    epoch_rmse.append(rmse)
                    epoch_mae.append(mae)
                else:
                    raise NotImplementedError("Multi-task Regiontrans_v2 has not been implemented")

                if i % 50 == 0:
                    print("[%.2fs]Epoch %d, Iter %d, Loss Target %.4f, Loss Consistency %.4f" %\
                         (time.time() - start_time, ep, i, loss_target.item(), loss_cons.item()))
            if self.out_channels == 1:
                print("[%.2fs]Epoch %d, Train Loss Target %.4f, Train Loss Consistency %.4f, RMSE %.4f, MAE %.4f" % \
                    (time.time() - start_time, ep, np.mean(epoch_target_loss), np.mean(epoch_cons_loss), np.mean(epoch_rmse) * (self.target_datamax - self.target_datamin), np.mean(epoch_mae) * (self.target_datamax - self.target_datamin)))
                logging.info("Epoch %d, Train Loss Target %.4f, Train Loss Consistency %.4f, RMSE %.4f, MAE %.4f" % \
                    (ep, np.mean(epoch_target_loss), np.mean(epoch_cons_loss), np.mean(epoch_rmse) * (self.target_datamax - self.target_datamin), np.mean(epoch_mae) * (self.target_datamax - self.target_datamin)))
                epoch_rmse = []
                epoch_mae = []
            else:
                raise NotImplementedError("Multi-task RegionTrans_v2 has not been implemented")
            epoch_cons_loss = []
            epoch_target_loss = []
            self.eval()
            if self.out_channels == 1:
                rmse, mae = self.evaluate("Validation", val_loader)
                if rmse < self.best_rmse:
                    self.best_rmse = rmse
                    logging.info("Saving best model...")
                    print("Saving best model...")
                    self.save_model('best')
            else:
                raise NotImplementedError("Multi-task RegionTrans_v2 has not been implemented")
                   
