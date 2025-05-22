import torch
import networkx as nx
from tqdm import tqdm
from data_loader import get_dataset
import matplotlib.pyplot as plt

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
    true_community = torch.load(f'./dataset/{args.dataset}/true_community_of_query.pt')
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


    # "3059"节点是第12个, query_node[12]
    a_query_node_id = query_node[12].tolist() 
    a_ture_com = torch.nonzero(true_community.to_dense()[12], as_tuple=True)[0].tolist()

    query_score = cosin_similarity(all_community_emb[a_query_node_id].unsqueeze(0), all_community_emb).tolist()[0] #[1, 4780]

    #>0.45
    nodeList_big= [1540, 1029, 522, 523, 4620, 3597, 3089, 4114, 1043, 1556, 4626, 2584, 1052, 1564, 1054, 543, 1568, 3615, 1058, 1575, 4143, 2612, 2624, 65, 1606, 582, 4678, 73, 585, 1097, 589, 595, 608, 1121, 616, 1134, 1649, 1650, 119, 637, 2689, 1160, 1673, 657, 2194, 157, 1702, 3242, 1710, 1723, 700, 2751, 706, 1731, 709, 3782, 2757, 3277, 717, 718, 4306, 212, 3284, 215, 221, 225, 1763, 740, 231, 1769, 750, 755, 2818, 3331, 778, 267, 1296, 273, 4370, 1811, 284, 2847, 1316, 3882, 301, 302, 2357, 828, 316, 1345, 323, 1354, 4426, 3917, 337, 853, 857, 1374, 1891, 355, 1899, 1906, 1396, 373, 391, 908, 397, 399, 3474, 917, 925, 928, 936, 428, 2478, 947, 437, 3000, 3005, 1983, 965, 454, 3527, 2506, 974, 464, 467, 3542, 3036, 477, 4062, 1005, 498, 3059, 1016]
    
    
    #> 0.5
    nodeList_1= [1540, 523, 4620, 3089, 4626, 1043, 2584, 1052, 1564, 1054, 3615, 1568, 1058, 1575, 1606, 4678, 73, 585, 595, 608, 1121, 1636, 616, 1134, 637, 2689, 1160, 1673, 657, 157, 1702, 1723, 706, 1219, 709, 3277, 717, 4306, 225, 1763, 231, 1769, 750, 755, 2818, 3331, 284, 1316, 3882, 301, 2357, 828, 1345, 1354, 337, 857, 1891, 1899, 1396, 373, 2951, 908, 397, 399, 925, 936, 428, 437, 1983, 965, 454, 2506, 467, 4062, 3563, 1005, 3059, 1016]
    #> 0.6
    nodeList_2= [2818, 1540, 397, 2584, 1052, 284, 925, 1058, 1316, 828, 706, 965, 454, 4678, 2506, 3277, 467, 1636, 1899, 3059, 1396, 1016]
    #> 0.9
    nodeList_4= [706, 1058, 1316, 965, 2818, 828, 3059, 467, 1396, 2584, 284, 925]


    intersection = list(set(nodeList_2) & set(nodeList_4))



    print("1")



    # 收集二跳邻居内得分大于 0.x 的节点 index
    high_score_nodes = [node for node in range(0,len(query_score)) if query_score[node] > 0.45]

    # 根据这些节点构建子图
    high_score_subgraph = G.subgraph(high_score_nodes).copy()

    # 从查询节点出发，选择与查询节点连接的部分
    connected_nodes = nx.node_connected_component(high_score_subgraph, a_query_node_id)
    connected_nodes_list = list(connected_nodes)


    # 使用原始子图 high_score_subgraph 获取节点的度数，并去掉度小于 2 的节点
    nodes_to_remove = [node for node in connected_nodes if high_score_subgraph.degree(node) < 4]#相当于小于2,有自环
    high_score_subgraph.remove_nodes_from(nodes_to_remove)

    # 更新 connected_subgraph
    connected_subgraph = high_score_subgraph.subgraph(connected_nodes).copy()
    result_nodes = list(connected_subgraph)


    print("1")

    
    high_score_subgraph = G.subgraph(result_nodes).copy()

    # 去掉子图中的自环
    high_score_subgraph.remove_edges_from(nx.selfloop_edges(high_score_subgraph))

    # 设置节点颜色，默认颜色为蓝色，查询节点为红色
    node_colors = ['red' if node == a_query_node_id else 'blue' for node in high_score_subgraph.nodes]

    # 可视化子图
    plt.figure(figsize=(10, 10))  # 设置图形大小
    nx.draw(
        high_score_subgraph,
        with_labels=True,  # 不显示节点 ID
        node_size=50,
        font_size=8,
        node_color=node_colors  # 设置节点颜色
    )
    plt.title("High Score Subgraph")
    plt.show()



    # subgraph_nodes = list(nx.single_source_shortest_path_length(G, query_index[0], cutoff=2).keys())
    # selected_candidates = GlobalSearch_test([a_query_node_id.tolist()], query_score[i].tolist(),G )  #算法5


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