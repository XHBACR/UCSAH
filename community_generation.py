import torch
import networkx as nx
from tqdm import tqdm
from data_loader import get_dataset

import time
import utils
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from early_stop import EarlyStopping, Stop_args
from model import PretrainModel
from lr import PolynomialDecayLR
import os.path
import torch.utils.data as Data
from utils import *
from accuracy_globalsearch import epoch_evaluate , evaluate , cosin_similarity , GlobalSearch,GlobalSearch_test

if __name__ == "__main__":

    args = parse_args()
    adjs = torch.load(f'./dataset/IMDB/imdb_adjs.pt') #(2, 4780, 4780) 稀疏tensor
    features = torch.load(f'./dataset/IMDB/imdb_node_features.pt') #(2, 4780, 1232) 稀疏tensor
    node_class = torch.load(f'./dataset/IMDB/imdb_node_class.pt') #(4780, 3)
    home_adj = torch.load(f'./dataset/IMDB/imdb_con_adj.pt')
    query_node=torch.load(f'./dataset/IMDB/query_node.pt')
    G = nx.from_numpy_array(home_adj.to_dense().numpy())
        
    processed_features=[]
    for i in range(0,args.metapath_num):
        processed_features.append(utils.re_features(adjs[i].float(), features[i].to_dense(), args.hops))  #[4780, 5, 1232]
        print("1")
    processed_features = torch.stack(processed_features, dim=1) #[4780, 2, 5, 1232]

    adj_batch, minus_adj_batch = [], []
    for i in range(0,args.metapath_num):
        adj_, minus_adj_ = transform_sp_csr_to_coo(transform_coo_to_csr(adjs[i]), args.batch_size, features.shape[1])
        adj_batch.append(adj_)
        minus_adj_batch.append(minus_adj_)

    data_loader = Data.DataLoader(processed_features, batch_size=args.batch_size, shuffle = False)

    model = PretrainModel(input_dim=processed_features.shape[3], config=args).to(args.device)
    model.load_state_dict(torch.load(args.save_path + args.model_name + '.pth'))


    all_community_emb=[]
    for index, item in enumerate(data_loader):

        nodes_features = item.to(args.device) #[1000, 2, 5, 1232]
        
        adj_set, minus_adj_set=[] , []
        for i in range(0,args.metapath_num): #[0,2)
            adj_set.append(adj_batch[i][index])  # append[1000,1000]
            minus_adj_set.append(minus_adj_batch[i][index])
        adj_set = torch.stack(adj_set, dim=0) #[2, 1000, 1000]
        minus_adj_set = torch.stack(minus_adj_set, dim=0) #[2, 1000, 1000]

        community_emb = model(nodes_features,  args.metapath_num) # [1000, 512] , []    <- [1000, 2, 5, 1232]

        all_community_emb.append(community_emb)
        # loss_train_b.append(loss_train.item())
    all_community_emb = torch.cat(all_community_emb, dim=0) #[4780, 512]

    query_score = cosin_similarity(all_community_emb[query_node[:50]], all_community_emb) #[600, 4780]
    y_pred = torch.zeros_like(query_score) #[600, 4780]
    for i in tqdm(range(query_score.shape[0])): #range(0, 600)
        selected_candidates = GlobalSearch_test([query_node[i].tolist()], query_score[i].tolist(),G )  #算法5
        for j in range(len(selected_candidates)):
            y_pred[i][selected_candidates[j]] = 1 #将预测的社区以独热编码的形式标记出来
        print("1")

    true_community = y_pred.to_sparse()

    # # 保存稀疏张量
    # torch.save(true_community, 'true_community_of_query.pt')
    # torch.save(query_node, 'query_node.pt')

    print("done")













# # 假设 adj_sum 是形状为 [4780, 4780] 的邻接矩阵
# # 假设 node_class 是形状为 [4780, 3] 的节点类别矩阵
# adjs = torch.load(f'./dataset/IMDB/imdb_adjs.pt')  # (2, 4780, 4780) 稀疏tensor
# features = torch.load(f'./dataset/IMDB/imdb_node_features.pt')  # (2, 4780, 1232) 稀疏tensor
# node_class = torch.load(f'./dataset/IMDB/imdb_node_class.pt')  # (4780, 3)

# node_num=node_class.shape[0]
# # 将 adjs 张量的两个矩阵相加
# adj_sum = adjs[0] + adjs[1]
# adj_sum = adj_sum.to_dense()
# # 将结果中大于1的值设置为1
# adj_sum = torch.where(adj_sum > 1, torch.tensor(1), adj_sum)

# # 将邻接矩阵转换为 NetworkX 图
# G = nx.from_numpy_array(adj_sum.numpy())

# # 初始化社区矩阵
# community_matrix = torch.zeros((node_num, node_num))

# # 遍历每个节点
# for node in tqdm(range(node_num), desc="Processing nodes"):
#     # 捕获2跳邻居，得到一个2跳之内的子图
#     subgraph_nodes = nx.single_source_shortest_path_length(G, node, cutoff=2).keys()
#     subgraph = G.subgraph(subgraph_nodes).copy()
#         # 获取目标节点的类别
#     target_class_indices = node_class[node].nonzero(as_tuple=True)[0]
    
#     if len(target_class_indices) == 0:
#         # 如果目标节点没有类别，跳过筛选类别的步骤
#         connected_subgraph = nx.node_connected_component(subgraph, node)
#     else:
#         # 获取目标节点的类别
#         target_class = target_class_indices.item()
        
#         # 剔除与目标节点类别不同的节点
#         nodes_to_remove = [n for n in subgraph.nodes if node_class[n, target_class] == 0]
#         subgraph.remove_nodes_from(nodes_to_remove)
        
#         # 只保留与目标节点直接或间接相连的部分
#         if node in subgraph:
#             connected_subgraph = nx.node_connected_component(subgraph, node)
#     # print("1")
#     for neighbor in connected_subgraph:
#         community_matrix[node, neighbor] = 1
#     # print("1")
# # print(community_matrix)
# # 将 community_matrix 转换为稀疏张量
# community_matrix_sparse = community_matrix.to_sparse()

# # 保存稀疏张量
# torch.save(community_matrix_sparse, 'community_matrix_sparse.pt')