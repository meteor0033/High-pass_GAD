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
     Paper: 
     Modified from https://github.com/squareRoot3/Rethinking-Anomaly-Detection
"""

class ChebConvGAD(nn.Module):
    ## in_feats 特征点维度；h_feats：隐层维度；num_classes：节点分类数（nomal，anomaly）
    def __init__(self, in_feats, h_feats, num_classes, graph,k=2, batch=False):
        super(ChebConvGAD, self).__init__()
        self.g = graph
        torch.manual_seed(1234567)
        self.conv1 = ChebConv(h_feats,h_feats,k)
        self.conv2 = ChebConv(h_feats,h_feats,k)
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats, h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        
        h = self.conv1(self.g,h,dgl.laplacian_lambda_max(self.g))
        #print("ChebConvGAD_conv1_h:", h.shape)
        h = self.conv2(self.g,h,dgl.laplacian_lambda_max(self.g))
        
        #BWGNN
        #h_final = torch.zeros([len(in_feat), 0])
        #for conv in self.conv:
        #    h0 = conv(self.g, h)
        #    h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h)
        h = self.act(h)
        h = self.linear4(h)
        return h

    def testlarge(self, g, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        #BWGNN
        #h_final = torch.zeros([len(in_feat), 0])
        #for conv in self.conv:
        #    h0 = conv(g, h)
        #    h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
            
        # ChebConv    
        h = self.conv1(self.g,h,dgl.laplacian_lambda_max(self.g))
        h = self.conv2(self.g,h,dgl.laplacian_lambda_max(self.g))    
        
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
        h = self.conv1(self.g,h,dgl.laplacian_lambda_max(self.g))
        h = h.relu()
        h = F.dropout(h, p = 0.5, training = self.training)
        h = self.conv2(self.g,h,dgl.laplacian_lambda_max(self.g))
        
        #BWGNN
        #h_final = torch.zeros([len(in_feat),0])
        #for conv in self.conv:
        #    h0 = conv(blocks[0], h)
        #    h_final = torch.cat([h_final, h0], -1)
        #    # print(h_final.shape)
        h = self.linear3(h)
        h = self.act(h)
        h = self.linear4(h)
        return h


# heterogeneous graph
class ChebConvGAD_Hetero(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph,k=2):
        super(ChebConvGAD_Hetero, self).__init__()
        self.g = graph        
        self.h_feats = h_feats
        
        self.g_0 = self.g[graph.canonical_etypes[0]]
        self.g_0_degree = self.g_0.in_degrees()
        self.g_0_degree_mask = (self.g_0_degree == 0)
        self.g_0_nodes = self.g_0.nodes()
        self.sg0 = self.g_0 .subgraph(self.g_0_nodes[self.g_0_degree_mask == False])
        self.g_0_features_mask = (self.g_0_degree == 0)
        
        
        self.g_1 = self.g[graph.canonical_etypes[1]]
        self.g_1_degree = self.g_1.in_degrees()
        self.g_1_degree_mask = (self.g_1_degree == 0)
        self.g_1_nodes = self.g_1.nodes()
        self.sg1 = self.g_1.subgraph(self.g_1_nodes[self.g_1_degree_mask == False])
        self.g_1_features_mask = (self.g_1_degree == 0)
        
        
        self.g_2 = self.g[graph.canonical_etypes[2]]
        self.g_2_degree = self.g_2.in_degrees()
        self.g_2_degree_mask = (self.g_2_degree == 0)
        self.g_2_nodes = self.g_2.nodes()
        self.sg2 = self.g_2.subgraph(self.g_2_nodes[self.g_2_degree_mask == False])
        self.g_2_features_mask = (self.g_2_degree == 0)        
        
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
        #self.linear5 = nn.Linear(h_feats, num_classes)
        self.act = nn.LeakyReLU()


        

    def forward(self, in_feat):
        h = self.linear(in_feat)
        #self.act = nn.LeakyReLU()
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_all = []
        
        #sg0
        h_final = torch.zeros([len(in_feat), 0])
        h_final1 = torch.zeros([len(in_feat[self.g_0_degree_mask == False]), 0])
        h_final2 = torch.zeros([len(in_feat[self.g_0_degree_mask]), 0])
        
        #g
        h01 = self.conv1(self.sg0, h[self.g_0_degree_mask == False], dgl.laplacian_lambda_max(self.sg0))
        h_final1 = torch.cat([h_final1, h01], -1)
        h11 = self.conv2(self.sg0,h01,dgl.laplacian_lambda_max(self.sg0))
        h_final1 = torch.cat([h_final1, h11], -1)
        #独立节点
        h02 = self.linear1_sg(h[self.g_0_degree_mask])
        h_final2 = torch.cat([h_final2, h02], -1)
        h12 = self.linear2_sg(h02)
        h_final2 = torch.cat([h_final2, h12], -1)
        
        h_final = torch.cat([h_final1, h_final2], 0)
        h = self.linear3(h_final)
        h_all.append(h)
        
        
        #sg1
        h_final = torch.zeros([len(in_feat), 0])
        h_final1 = torch.zeros([len(in_feat[self.g_1_degree_mask == False]), 0])
        h_final2 = torch.zeros([len(in_feat[self.g_1_degree_mask]), 0])
                                
        #g
        h01 = self.conv1(self.sg1, h[self.g_1_degree_mask == False], dgl.laplacian_lambda_max(self.sg1))
        h_final1 = torch.cat([h_final1, h01], -1)
        h11 = self.conv2(self.sg1,h01,dgl.laplacian_lambda_max(self.sg1))
        h_final1 = torch.cat([h_final1, h11], -1)
        #独立节点
        h02 = self.linear1_sg(h[self.g_1_degree_mask])
        h_final2 = torch.cat([h_final2, h02], -1)
        h12 = self.linear2_sg(h02)
        h_final2 = torch.cat([h_final2, h12], -1)
        
        h_final = torch.cat([h_final1, h_final2], 0)
        h = self.linear3(h_final)
        h_all.append(h) 
        
        #sg2
        h_final = torch.zeros([len(in_feat), 0])
        h_final1 = torch.zeros([len(in_feat[self.g_2_degree_mask == False]), 0])
        h_final2 = torch.zeros([len(in_feat[self.g_2_degree_mask]), 0])
                                
        #g
        h01 = self.conv1(self.sg2, h[self.g_2_degree_mask == False], dgl.laplacian_lambda_max(self.sg2))
        h_final1 = torch.cat([h_final1, h01], -1)
        h11 = self.conv2(self.sg2, h01, dgl.laplacian_lambda_max(self.sg2))
        h_final1 = torch.cat([h_final1, h11], -1)
        #独立节点
        h02 = self.linear1_sg(h[self.g_2_degree_mask])
        h_final2 = torch.cat([h_final2, h02], -1)
        h12 = self.linear2_sg(h02)
        h_final2 = torch.cat([h_final2, h12], -1)
        
        h_final = torch.cat([h_final1, h_final2], 0)
        h = self.linear3(h_final)
        h_all.append(h)

        h_all = torch.stack(h_all).sum(0)
        #self.act = nn.LeakyReLU()
        h_all = self.act(h_all)
        h_all = self.linear4(h_all)
        #h_all = self.act(h_all)
        #h_all = self.linear5(h_all)
        return h_all
