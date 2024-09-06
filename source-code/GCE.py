import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


class GAE(nn.Module):
    def __init__(self, adj, input_dim, hidden1_dim, hidden2_dim, latent_dim):
        super(GAE, self).__init__()
        self.adj = adj
        self.input_size = input_dim
        self.hidden1 = hidden1_dim
        self.hidden2 = hidden2_dim
        self.latent = latent_dim

        self.base_gcn1 = GraphConvSparse(self.input_size, self.hidden1, self.adj)
        self.base_gcn2 = GraphConvSparse(self.hidden1, self.hidden2, self.adj)
        self.gcn_mean = GraphConvSparse(self.hidden2, self.latent, self.adj, activation=lambda x: x)

    def encode(self, X):
        hidden1 = self.base_gcn1(X)
        hidden2 = self.base_gcn2(hidden1)
        mean = self.gcn_mean(hidden2)

        return mean

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred, Z
