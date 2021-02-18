import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import spmm

def normalize(A , symmetric=True):
    d = A.sum(1)
    if symmetric:
        #D = D^-1/2
        D = torch.diag(torch.pow(d , -0.5))
        return D.mm(A).mm(D)
    else :
        # D=D^-1
        D =torch.diag(torch.pow(d,-1))
        return D.mm(A)

class HA():
    def __init__(self,out_dim=None,mode=1):
        self.out_dim = out_dim
        self.mode = mode
    def predict(self,X):
        if(self.out_dim):
            return np.stack([X.mean(self.mode)]*self.out_dim,self.mode)
        else:
            return X.mean(self.mode,keepdims=True)
        

class GLR(nn.Module):
    def __init__(self,in_dim,out_dim,ylength):
        super(GLR,self).__init__()
        self.layer = nn.Linear(in_dim,out_dim)
        self.ylength = ylength
        
    def forward(self,X,weather=None):
        H = X.permute(0,2,1,3).contiguous()
        H = H.view(X.size(0),X.size(2),-1)
        H = self.layer(H)
        H = H.view(X.size(0),X.size(2),self.ylength,-1).permute(0,2,1,3).contiguous()
        return H


class GCN(nn.Module):
    def __init__(self,in_dim,out_dim,euc,poi,road,ylength,n_layers=3,hid_dim=64):
        super(GCN,self).__init__()
        self.A = [normalize((euc>0).float()),normalize((road>0).float()),normalize((poi>0).float())]
        print(self.A)
        layer_list = []
        ln_list = []
        res_list = []
        self.n_layers=n_layers
        self.ylength=ylength
        
        for k in range(0,len(self.A)):
            layer_list.append(nn.ModuleList([]))
            ln_list.append(nn.ModuleList([]))
            res_list.append(nn.ModuleList([]))
            for i in range(0,n_layers):
                if(i==0):
                    layer_list[k].append(nn.Linear(in_dim,hid_dim))
                    res_list[k].append(nn.Linear(in_dim,hid_dim))
                    ln_list[k].append(nn.LayerNorm([in_dim]))
                elif(i==n_layers-1):
                    layer_list[k].append(nn.Linear(hid_dim,out_dim))
                    res_list[k].append(nn.Linear(hid_dim,out_dim))
                    ln_list[k].append(nn.LayerNorm([hid_dim]))
                else:
                    layer_list[k].append(nn.Linear(hid_dim,hid_dim))
                    res_list[k].append(nn.Linear(hid_dim,hid_dim))
                    ln_list[k].append(nn.LayerNorm([hid_dim]))
                
        self.layer = nn.ModuleList(layer_list)
        self.res = nn.ModuleList(res_list)
        self.LN = nn.ModuleList(ln_list)
        self.act = nn.LeakyReLU(0.2)
    def forward(self,X,weather=None):
        H_list = []
        for k in range(0,len(self.A)):
            A = self.A[k]            
            H = X.permute(0,2,1,3).contiguous()
            H = H.view(X.size(0),X.size(2),-1)
            for i in range(0,self.n_layers):
                #H = self.LN[i](H)
                H_prime = self.res[k][i](H)
                H = torch.matmul(H.permute(0,2,1),A).permute(0,2,1)
                H = self.layer[k][i](H) #(b,f,n)
                H = self.act(H+H_prime)
            H = H.view(X.size(0),X.size(2),self.ylength,-1).permute(0,2,1,3).contiguous()
            H_list.append(H)
        return H_list[0]+H_list[1]+H_list[2]


        
class attn(nn.Module):
    def __init__(self,in_dim,out_dim,A,alpha=0.2,device='cuda:0'):
        super(attn,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.W = nn.Linear(in_dim,out_dim)
        self.a = nn.Linear(2*out_dim,1)
        self.r = nn.Linear(in_dim,out_dim)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.ln = nn.LayerNorm(out_dim)
        self.A = A
    def forward(self,X,weather=None):
        #X (B,N,F_in)
        B = X.size(0)
        N = X.size(1)
        edge = self.A.to_sparse()

        edge=edge.indices()

        Wh = self.leakyrelu(self.W(X)) #(B,N,F_out)
        res = self.r(X)
        w1 =Wh[:,edge[0,:],:]
        w2 = Wh[:,edge[1,:],:]
        edge_h = torch.cat((w1,w2),dim=-1) #(B,e,2d)

        edge_e = torch.exp(-self.leakyrelu(self.a(edge_h).squeeze()))

        edge_e = [torch.sparse_coo_tensor(edge,edge_e[b,:],(N,N)) for b in range(0,B)]

        e_rowsum = [torch.sparse.mm(edge_e[b],torch.ones(size=(N,1)).to(self.device)) for b in range(0,B)]
        
        h_prime = [torch.sparse.mm(edge_e[b],Wh[b,:,:]).div(e_rowsum[b]+torch.Tensor([9e-15]).to(self.device)) for b in range(0,B)]
        h_prime = torch.stack(h_prime,dim=0)
        
        h_prime = self.leakyrelu(h_prime+res)
        #h_prime = self.ln(h_prime)
        return h_prime

    def dense_forward(self,X,weather=None):
        #X (B,N,F_in)
        B = X.size(0)
        N = X.size(1)
        Wh = self.W(X) #(B,N,F_out)
        a1 = Wh.repeat_interleave(N,dim=1)
        a2 = Wh.repeat(1,N,1)
        combined_a = torch.cat([a1,a2],dim=-1).view(-1,N,N,2*self.out_dim)
        e = self.leakyrelu(self.a(combined_a).squeeze(-1))

        A_r = self.A.unsqueeze(0).repeat(B,1,1)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(A_r>0,e,zero_vec)
        attention = F.softmax(attention,dim=-1)

        h_prime = torch.bmm(attention,Wh)
        return self.ln(h_prime)


class GAT(nn.Module):
    def __init__(self,in_dim,out_dim,euc,poi,road,ylength,n_layers=3,hid_dim=64,device='cuda:0'):
        super(GAT,self).__init__()
        self.A = [(euc>0).float(),(road>0).float(),(poi>0).float()]
        layer_list = []
        
        self.n_layers=n_layers
        self.ylength=ylength
        
        for k in range(0,len(self.A)):
            layer_list.append(nn.ModuleList([]))
            for i in range(0,n_layers):
                if(i==0):
                    layer_list[k].append(attn(in_dim,hid_dim,self.A[k],device=device))
                elif(i==n_layers-1):
                    layer_list[k].append(attn(hid_dim,out_dim,self.A[k],device=device))
                else:
                    layer_list[k].append(attn(hid_dim,hid_dim,self.A[k],device=device))

        self.layer = nn.ModuleList(layer_list)

    def forward(self,X,weather=None):
        H_list = []
        for k in range(0,len(self.A)):
            H=X.permute(0,2,1,3).contiguous()
            H = H.view(X.size(0),X.size(2),-1)
            for i in range(0,self.n_layers):
                H = self.layer[k][i](H)
            H = H.view(X.size(0),X.size(2),self.ylength,-1).permute(0,2,1,3).contiguous()
            H_list.append(H)
        return H_list[0]+H_list[1]+H_list[2]

        
