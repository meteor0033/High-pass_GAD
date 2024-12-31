import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import math
import dgl
import sympy
import scipy
import numpy as np
from torch import nn
from torch.nn import init
#注意这里导入的几个模型
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm, ChebConv, GATConv, HeteroGraphConv

"""
     High-pass_GCN
     Paper: High-pass Graph convolution network for Graph anomaly detection
     Modified from https://github.com/squareRoot3/Rethinking-Anomaly-Detection
"""

class ChebConvGAD(nn.Module):
    ## in_feats 特征点维度；h_feats：隐层维度；num_classes：节点分类数（nomal，anomaly）
    def __init__(self, in_feats, h_feats, num_classes, graph,k=3, batch=False):
        super(ChebConvGAD, self).__init__()
        self.g = graph
        self.lambda_max = dgl.laplacian_lambda_max(self.g)
        torch.manual_seed(1234567)
        self.conv1 = ChebConv(h_feats,h_feats,k)
        self.conv2 = ChebConv(h_feats,h_feats,k)
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats, h_feats)
        self.linear3_A = nn.Linear(h_feats*2, h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()


    def forward(self, in_feat,dataset):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([2*len(in_feat), 0])

        if dataset == 'amazon':
            h0 = self.conv1(self.g,h,self.lambda_max)
            #h0 = self.conv1(self.g,h0,self.lambda_max)
            h1 = self.conv2(self.g,h,self.lambda_max)
            #h1 = self.conv2(self.g,h1,self.lambda_max)
            h0 = self.act(h0)
            h = self.act(h1)
            h_final = torch.cat([h0, h1], -1)
            h = self.linear3_A(h_final)
            h = self.act(h)
            h = self.linear4(h)

        else: 
            h = self.conv1(self.g,h,self.lambda_max)
            h = self.conv2(self.g,h,self.lambda_max)
            h = self.linear3(h)
            h = self.act(h)
            h = self.linear4(h)
        return h

    def testlarge(self, g, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
            
        # ChebConv    
        h = self.conv1(self.g,h,self.lambda_max)
        h = self.conv2(self.g,h,self.lambda_max)    
        
        h = self.linear3(h)
        h = self.act(h)
        h = self.linear4(h)
        return h

    def batch(self, blocks, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)

        
        # ChebConv    
        h = self.conv1(self.g,h,self.lambda_max)
        h = h.relu()
        h = F.dropout(h, p = 0.5, training = self.training)
        h = self.conv2(self.g,h,self.lambda_max)
        

        h = self.linear3(h)
        h = self.act(h)
        h = self.linear4(h)
        return h


# heterogeneous graph
class ChebConvGAD_Hetero(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph,k=2, args = None):
        super(ChebConvGAD_Hetero, self).__init__()
        self.args = args
        self.g = graph        
        self.h_feats = h_feats
        torch.manual_seed(1234567)
        # 拆成三个和到一起
        self.conv1 = ChebConv(h_feats,h_feats,k)
        self.conv2 = ChebConv(h_feats,h_feats,k)
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear1_sg = nn.Linear(h_feats, h_feats)
        self.linear2_sg = nn.Linear(h_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats*2, h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.LeakyReLU()


        

    def forward(self, in_feat, dataset):
        h = self.linear(in_feat)
        #self.act = nn.LeakyReLU()
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_all = []

        for relation in self.g.canonical_etypes:
            ## ChebConv 
            h_final = torch.zeros([len(in_feat), 0])
            if dataset == 'amazon':
                h0 = self.conv1(self.g[relation], h, dgl.laplacian_lambda_max(self.g[relation]))
                h_final = torch.cat([h_final, h0], -1)
                h1 = self.conv2(self.g[relation],h0,dgl.laplacian_lambda_max(self.g[relation]))
                h_final = torch.cat([h_final, h1], -1)
            else:   
                g_sub = self.g[relation]
                g_sub_degree = g_sub.in_degrees()
                g_sub_degree_mask = (g_sub_degree == 0)
                g_sub_nodes = g_sub.nodes()
                sg = g_sub.subgraph(g_sub_nodes[g_sub_degree_mask == False])
                
                h_final1 = torch.zeros([len(in_feat[g_sub_degree_mask == False]), 0])
                h_final2 = torch.zeros([len(in_feat[g_sub_degree_mask]), 0])
                
                h01 = self.conv1(sg, h[g_sub_degree_mask == False], dgl.laplacian_lambda_max(sg))
                h_final1 = torch.cat([h_final1, h01], -1)
                h11 = self.conv2(sg,h01,dgl.laplacian_lambda_max(sg))
                h_final1 = torch.cat([h_final1, h11], -1)
                
                h02 = self.linear1_sg(h[g_sub_degree_mask])
                h_final2 = torch.cat([h_final2, h02], -1)
                h12 = self.linear2_sg(h[g_sub_degree_mask])
                h_final2 = torch.cat([h_final2, h12], -1)
                h_final = torch.cat([h_final1, h_final2], 0)
                
            h = self.linear3(h_final)
            h_all.append(h)

        h_all = torch.stack(h_all).sum(0)
        #self.act = nn.LeakyReLU()
        h_all = self.act(h_all)
        h_all = self.linear4(h_all)
        
        return h_all
