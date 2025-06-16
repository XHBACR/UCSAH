import torch
import math
import torch.nn as nn
from layer import *
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GCNConv


class PretrainModel(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.config = config

        self.Linear1 = nn.Linear(input_dim, self.config.hidden_dim)
        self.encoder = Zoom_aware_Transformer(hops=config.hops, 
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
        
        
        self.SemanticAttention= SemanticAttention(config.hidden_dim*2)
        self.classification_layer = NodeClassificationLayer(config.hidden_dim, config.class_num)

        self.marginloss = nn.MarginRankingLoss(0.5)
        self.classificationloss = nn.CrossEntropyLoss()


    def trainModel(self, x , adj_, minus_adj , metapath_num, class_label): 
        shuf_index = torch.randperm(x.shape[0])
        sem_con_loss = 0.0
        glo_con_loss = 0.0
        cls_loss = 0.0
        low_level_emb=[]
        high_level_emb=[]

        for i in range(0, metapath_num):
            node_tensor, neighbor_tensor = self.encoder(x[:,i,:,:])
            neighbor_tensor = self.readout(neighbor_tensor, torch.tensor([0]).to(self.config.device)) 
            node_tensor=node_tensor.squeeze() 
            neighbor_tensor=neighbor_tensor.squeeze() 
            node_tensor_shuf = node_tensor[shuf_index] 
            neighbor_tensor_shuf = neighbor_tensor[shuf_index] 
            logits_aa = torch.sigmoid(torch.sum(node_tensor * neighbor_tensor, dim = -1)) 
            logits_bb = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor_shuf, dim = -1))
            logits_ab = torch.sigmoid(torch.sum(node_tensor * neighbor_tensor_shuf, dim = -1))
            logits_ba = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor, dim = -1))
            # ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
            ones = torch.ones(logits_aa.size(0)).cpu() 
            sem_con_loss += self.marginloss(logits_aa, logits_ba, ones) 
            sem_con_loss += self.marginloss(logits_bb, logits_ab, ones)
            low_level_emb.append(node_tensor)
            high_level_emb.append(neighbor_tensor)
        low_level_emb = torch.stack(low_level_emb, dim=1) 
        high_level_emb = torch.stack(high_level_emb, dim=1) 
        node_emb, community_emb = self.SemanticAttention(low_level_emb, high_level_emb) 
        node_tensor_shuf = node_emb[shuf_index]  
        neighbor_tensor_shuf = community_emb[shuf_index] 
        logits_aa = torch.sigmoid(torch.sum(node_emb * community_emb, dim = -1)) 
        logits_bb = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor_shuf, dim = -1))
        logits_ab = torch.sigmoid(torch.sum(node_emb * neighbor_tensor_shuf, dim = -1))
        logits_ba = torch.sigmoid(torch.sum(node_tensor_shuf * community_emb, dim = -1))
        # ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
        ones = torch.ones(logits_aa.size(0)).cpu() 
        glo_con_loss += self.marginloss(logits_aa, logits_ba, ones) 
        glo_con_loss += self.marginloss(logits_bb, logits_ab, ones)
        con_loss= 0.7*sem_con_loss + glo_con_loss
        
        class_prediction = self.classification_layer(node_emb) 
        cls_loss=self.classificationloss(class_prediction, class_label)
        
        total_loss=con_loss + cls_loss
        return community_emb, class_prediction, total_loss

    def forward(self, x,  metapath_num): 
        shuf_index = torch.randperm(x.shape[0])
        TotalLoss = 0.0
        low_level_emb=[]
        high_level_emb=[]
        for i in range(0, metapath_num):
            node_tensor, neighbor_tensor = self.encoder(x[:,i,:,:]) 
            neighbor_tensor = self.readout(neighbor_tensor, torch.tensor([0]).to(self.config.device))
            node_tensor=node_tensor.squeeze() 
            neighbor_tensor=neighbor_tensor.squeeze()
            
            low_level_emb.append(node_tensor)
            high_level_emb.append(neighbor_tensor)
        low_level_emb = torch.stack(low_level_emb, dim=1) 
        high_level_emb = torch.stack(high_level_emb, dim=1) 
        node_emb, community_emb = self.SemanticAttention(low_level_emb, high_level_emb) 
        class_prediction = self.classification_layer(node_emb)
        return community_emb, class_prediction

       

