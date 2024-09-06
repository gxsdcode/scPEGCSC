import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from torch.autograd import Variable
import h5py
import numpy as np
from evaluation import eva
from GCE import GAE
from preprocess import read_dataset, normalize, get_adj
import scanpy as sc
from utils import thrC, post_proC, spectralclustering
import pandas as pd
import csv


class Deep_Sparse_Subspace_Clustering(nn.Module):

    def __init__(self, enc_1, enc_2, dec_1, dec_2,
                 inputsize, l_z, pre_lr,
                 adata, pre_epoches, adj):
        super(Deep_Sparse_Subspace_Clustering, self).__init__()
        self.enc_1 = enc_1
        self.enc_2 = enc_2
        self.dec_1 = dec_1
        self.dec_2 = dec_2
        self.inputsize = inputsize
        self.l_z = l_z
        self.pre_lr = pre_lr
        self.adata = adata
        self.pre_epoches = pre_epoches
        self.adj = adj
        self.model1 = GAE(adj=self.adj, input_dim=self.inputsize, hidden1_dim=self.enc_1, hidden2_dim=self.enc_2,
                          latent_dim=self.l_z)

        weights = self._initialize_weights()
        self.Coef = weights['Coef']
        self.Coef1 = weights['Coef']
        print("Coef1:", self.Coef1)

        self.norm = self.adj.shape[0] * self.adj.shape[0] / float(
            (self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) * 2)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['Coef'] = Parameter(
            1.0e-2 * (torch.ones(size=(len(self.adata.X), len(self.adata.X)))) - 1.0 * torch.eye(len(self.adata.X)))
        return all_weights

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train1(self):
        log_interval = 1
        optimizer = Adam(self.parameters(), lr=self.pre_lr)
        for epoch in range(1, self.pre_epoches + 1):
            x_tensor = Variable(torch.Tensor(adata.X))
            adj_res1, z1 = self.model1(x_tensor)
            z_c1 = torch.matmul(self.Coef1, z1)
            adj_res2 = torch.sigmoid(torch.matmul(z_c1, z_c1.t()))
            loss_reconst1 = torch.sum(torch.pow((self.adj - adj_res1), 2))
            loss_reconst2 = torch.sum(torch.pow((self.adj - adj_res2), 2))
            print("loss_adj1:", loss_reconst1)
            print("loss_adj2:", loss_reconst2)
            loss_reg1 = torch.sum(torch.pow(self.Coef1, 2))
            print("loss_reg1:", loss_reg1)
            loss_selfexpress1 = torch.sum(torch.pow((z1 - z_c1), 2))
            print("loss_self1:", loss_selfexpress1)
            # loss1 = 0.5 * loss_reconst1 + loss_selfexpress1 + loss_reg1
            loss2 = 0.2 * loss_reconst2 + 0.1 * loss_selfexpress1 + 1.0 * loss_reg1
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
            if epoch % log_interval == 0:
                print('Train1 Epoch: {} ''\tLoss: {:.6f}'.format(epoch, loss2.item()))
            if epoch == self.pre_epoches:
                print('training1 completed')
        return self.Coef1.detach().numpy()


data_mat = h5py.File('Human1.h5')
x = np.array(data_mat['X'])

x = x.T
y = np.array(data_mat['Y'])
data_mat.close()

adata = sc.AnnData(x)
adata.obs['Group'] = y

adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)

adata = normalize(adata,
                  size_factors=True,
                  normalize_input=True,
                  logtrans_input=True,
                  select_hvg=True)
x_sd = adata.X.std(0)
x_sd_median = np.median(x_sd)
print("median of gene sd: %.5f" % x_sd_median)
sd = 2.5
adj, adj_n = get_adj(adata.X)

adj = torch.Tensor(adj)
adj_n = torch.Tensor(adj_n)

inputsize = adata.X.shape[1]
net = Deep_Sparse_Subspace_Clustering(enc_1=256, enc_2=32, dec_1=32, dec_2=256, inputsize=inputsize,
                                      l_z=16, pre_lr=0.02, adata=adata, pre_epoches=200, adj=adj_n)

n_clusters = len(np.unique(y))
C1 = net.train1()
C1 = np.abs(C1)
C3 = (C1 + C1.T)
C4 = np.matmul(C3, C3)
C42 = C4 + C3
pred_label42 = spectralclustering(C42, n_clusters)
pred_label42 = pred_label42.astype(np.int64)
y = y.astype(np.int64)
eva(y, pred_label42)
