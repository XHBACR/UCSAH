import torch
import math
import torch.nn as nn
from layer import TransformerBlock, SemanticAttention
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GCNConv
from utils import is_edge_in_edge_index

class PretrainModel(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.config = config

        self.Linear1 = nn.Linear(input_dim, self.config.hidden_dim)
        self.encoder = TransformerBlock(hops=config.hops, 
                        input_dim=input_dim, 
                        n_layers=config.n_layers,
                        num_heads=config.n_heads,
                        hidden_dim=config.hidden_dim,
                        dropout_rate=config.dropout,
                        attention_dropout_rate=config.attention_dropout)
        if config.readout == "sum":
            self.readout = global_add_pool
        elif config.readout == "mean":
            self.readout = global_mean_pool
        elif config.readout == "max":
            self.readout = global_max_pool
        else:
            raise ValueError("Invalid pooling type.")
        
        self.marginloss = nn.MarginRankingLoss(0.5)
        self.SemanticAttention= SemanticAttention(config.hidden_dim*2)

    def trainModel(self, x , adj_, minus_adj , metapath_num): #[1000, 2, 5, 1232] , [2, 1000, 1000], [2, 1000, 1000] , 2
        shuf_index = torch.randperm(x.shape[0])
        TotalLoss = 0.0
        low_level_emb=[]
        high_level_emb=[]

        #1. 不同metapath下的contrastiive loss计算=======================================================================
        for i in range(0, metapath_num):
            node_tensor, neighbor_tensor = self.encoder(x[:,i,:,:]) # [1000, 1, 512] , [1000, 4, 512]  <- [1000, 5, 1232]
            neighbor_tensor = self.readout(neighbor_tensor, torch.tensor([0]).to(self.config.device)) #[1000, 1, 512] <- [1000, 4, 512] 
            node_tensor=node_tensor.squeeze() #[1000, 512]
            neighbor_tensor=neighbor_tensor.squeeze() #[1000, 512]
            
            node_tensor_shuf = node_tensor[shuf_index]  #[1000, 512]
            neighbor_tensor_shuf = neighbor_tensor[shuf_index] #[1000, 512]

            #aa表示两个未打乱的矩阵相乘,代表同节点的 节点和社区emb相乘. 算法2中
            logits_aa = torch.sigmoid(torch.sum(node_tensor * neighbor_tensor, dim = -1)) #[1000]
            logits_bb = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor_shuf, dim = -1))#[1000]
            logits_ab = torch.sigmoid(torch.sum(node_tensor * neighbor_tensor_shuf, dim = -1))#[1000]
            logits_ba = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor, dim = -1))#[1000]

            # ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
            ones = torch.ones(logits_aa.size(0)).cpu() #[2708]
            TotalLoss += self.marginloss(logits_aa, logits_ba, ones) #公式5
            TotalLoss += self.marginloss(logits_bb, logits_ab, ones)
            
            pairwise_similary = torch.mm(node_tensor, node_tensor.t())
            link_loss = minus_adj[i].multiply(pairwise_similary)-adj_[i].multiply(pairwise_similary) #公式6
            # link_loss = torch.abs(torch.sum(link_loss))/(adj_[i].shape[0])
            link_loss = torch.sum(link_loss)/(adj_[i].shape[0]*adj_[i].shape[0])

            # TotalLoss += 0.001*link_loss
            TotalLoss += self.config.alpha*link_loss  #公式7

            low_level_emb.append(node_tensor)
            high_level_emb.append(neighbor_tensor)
        #========================================================================================================


        #2. 聚合不同semantic的emb,并计算contrastive loss================================================================
        low_level_emb = torch.stack(low_level_emb, dim=1) #[1000, 2, 512]
        high_level_emb = torch.stack(high_level_emb, dim=1) #[1000, 2, 512]
        node_emb, community_emb = self.SemanticAttention(low_level_emb, high_level_emb) #[1000, 512] , [1000, 512]


        node_tensor_shuf = node_emb[shuf_index]  #[1000, 512]
        neighbor_tensor_shuf = community_emb[shuf_index] #[1000, 512]

        #aa表示两个未打乱的矩阵相乘,代表同节点的 节点和社区emb相乘. 算法2中
        logits_aa = torch.sigmoid(torch.sum(node_tensor * neighbor_tensor, dim = -1)) #[1000]
        logits_bb = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor_shuf, dim = -1))#[1000]
        logits_ab = torch.sigmoid(torch.sum(node_tensor * neighbor_tensor_shuf, dim = -1))#[1000]
        logits_ba = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor, dim = -1))#[1000]

        # ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
        ones = torch.ones(logits_aa.size(0)).cpu() #[2708]
        TotalLoss += self.marginloss(logits_aa, logits_ba, ones) #公式5
        TotalLoss += self.marginloss(logits_bb, logits_ab, ones)
        
        pairwise_similary = torch.mm(node_tensor, node_tensor.t())
        link_loss = minus_adj[i].multiply(pairwise_similary)-adj_[i].multiply(pairwise_similary) #公式6
        # link_loss = torch.abs(torch.sum(link_loss))/(adj_[i].shape[0])
        link_loss = torch.sum(link_loss)/(adj_[i].shape[0]*adj_[i].shape[0])

        # TotalLoss += 0.001*link_loss
        TotalLoss += self.config.alpha*link_loss  #公式7
        #========================================================================================================



        return community_emb, TotalLoss

    def forward(self, x,  metapath_num): #   , [2708, 6, 1436]
        shuf_index = torch.randperm(x.shape[0])
        TotalLoss = 0.0
        low_level_emb=[]
        high_level_emb=[]

        #1. 不同metapath下的contrastiive loss计算=======================================================================
        for i in range(0, metapath_num):
            node_tensor, neighbor_tensor = self.encoder(x[:,i,:,:]) # [1000, 1, 512] , [1000, 4, 512]  <- [1000, 5, 1232]
            neighbor_tensor = self.readout(neighbor_tensor, torch.tensor([0]).to(self.config.device)) #[1000, 1, 512] <- [1000, 4, 512] 
            node_tensor=node_tensor.squeeze() #[1000, 512]
            neighbor_tensor=neighbor_tensor.squeeze() #[1000, 512]
            
            low_level_emb.append(node_tensor)
            high_level_emb.append(neighbor_tensor)
        #========================================================================================================


        #2. 聚合不同semantic的emb,并计算contrastive loss================================================================
        low_level_emb = torch.stack(low_level_emb, dim=1) #[1000, 2, 512]
        high_level_emb = torch.stack(high_level_emb, dim=1) #[1000, 2, 512]
        node_emb, community_emb = self.SemanticAttention(low_level_emb, high_level_emb) #[1000, 512] , [1000, 512]
        
        return community_emb

       

