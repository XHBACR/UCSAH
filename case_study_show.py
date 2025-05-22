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

import matplotlib.pyplot as plt
from matplotlib import rcParams




if __name__ == "__main__":

    args = parse_args()
    # true_community = torch.load(f'./dataset/{args.dataset}/true_community_of_query.pt')
    home_adj = torch.load(f'./dataset/IMDB/imdb_con_adj.pt').to_dense()
    home_adj.fill_diagonal_(0)

    G = nx.from_numpy_array(home_adj.numpy())
    # ture_com = torch.nonzero(true_community.to_dense()[12], as_tuple=True)[0].tolist()

    true_com=[73, 157, 284, 301, 428, 437, 454, 467, 595, 616, 637, 706, 717, 750, 755, 828, 857, 925, 936, 965, 1005, 1016, 1043, 1052, 1054, 1058, 1121, 1160, 1316, 1345, 1354, 1396, 1540, 1568, 1575, 1673, 1763, 1769, 1899, 1983, 2357, 2506, 2584, 2689, 2818, 3059, 3882, 4306]
    #查询节点"3059"
    query_id= 3059

    #>0.45
    nodeList_big= [1106, 1219, 1636, 3563, 2951, 1540, 1029, 522, 523, 4620, 3597, 3089, 4114, 1043, 1556, 4626, 2584, 1052, 1564, 1054, 543, 1568, 3615, 1058, 1575, 4143, 2612, 2624, 65, 1606, 582, 4678, 73, 585, 1097, 589, 595, 608, 1121, 616, 1134, 1649, 1650, 119, 637, 2689, 1160, 1673, 657, 2194, 157, 1702, 3242, 1710, 1723, 700, 2751, 706, 1731, 709, 3782, 2757, 3277, 717, 718, 4306, 212, 3284, 215, 221, 225, 1763, 740, 231, 1769, 750, 755, 2818, 3331, 778, 267, 1296, 273, 4370, 1811, 284, 2847, 1316, 3882, 301, 302, 2357, 828, 316, 1345, 323, 1354, 4426, 3917, 337, 853, 857, 1374, 1891, 355, 1899, 1906, 1396, 373, 391, 908, 397, 399, 3474, 917, 925, 928, 936, 428, 2478, 947, 437, 3000, 3005, 1983, 965, 454, 3527, 2506, 974, 464, 467, 3542, 3036, 477, 4062, 1005, 498, 3059, 1016]
    
    
    #> 0.5
    nodeList_1= [1540, 523, 4626, 1043, 2584, 1052, 1564, 1054, 1568, 1058, 1575, 73, 585, 595, 608, 1121, 1636, 616, 1134, 637, 2689, 1160, 1673, 657, 157, 1702, 1723, 706, 1219, 709, 3277, 717, 4306, 225, 1763, 231, 1769, 750, 755, 2818, 3331, 284, 1316, 3882, 301, 2357, 828, 1345, 1354, 337, 857, 1891, 1899, 1396, 373, 2951, 908, 397, 925, 936, 428, 437, 1983, 965, 454, 2506, 467, 1005, 3059, 1016]
    #> 0.6
    nodeList_2= [2818, 1540, 397, 2584, 1052, 284, 925, 1058, 1316, 828, 706, 965, 454, 2506, 3277, 467, 1636, 1899, 3059, 1396, 1016]
    #> 0.9
    nodeList_3= [706, 1058, 1316, 965, 2818, 828, 3059, 467, 1396, 2584, 284, 925]


    intersection = list(set(nodeList_big) & set(true_com))
    not_in_big = list(set(true_com) - set(nodeList_big))


    graph = G.subgraph(nodeList_big).copy()


    print("1")


    # target_nodes = true_com
    target_nodes = nodeList_3
    # 设置节点颜色
    node_colors = [
        '#bb603b' if node == query_id else ('#3c71b7' if node in target_nodes else 'white')  # query_id 为红色，nodeList_1 为蓝色，其他无颜色
        for node in graph.nodes
    ]
    
    # 设置节点边框颜色
    node_edge_colors = [
        'black' if node in target_nodes else 'lightgray'  # nodeList_1 中的节点边框为黑色，其他节点为蓝色
        for node in graph.nodes
    ]

    # 设置边颜色为浅灰色
    edge_colors = 'lightgray'

    # 使用 spring_layout 布局
    pos = nx.spring_layout(graph, k=0.6, iterations=50, seed=51)  # seed 固定随机数种子

    # 设置字体为新罗马字体
    rcParams['font.family'] = 'Times New Roman'

    # 可视化子图
    plt.figure(figsize=(5, 4))  # 设置图形大小
    nx.draw(
        graph,
        pos=pos,  # 使用 spring_layout 的位置
        with_labels=False,  # 不显示节点标签
        # with_labels=True, 
        node_size=200,  # 调整节点大小
        font_size=8,
        node_color=node_colors,  # 设置节点填充颜色
        edge_color=edge_colors,  # 设置边颜色
        edgecolors=node_edge_colors  # 设置节点边框颜色
    )


    # 添加标题到图下方
    plt.gcf().text(0.5, 0, "(d) With threshold 0.9", fontsize=24, ha='center', fontfamily='Times New Roman')

    # 保存为 PDF 文件，并裁剪白边
    plt.savefig("case_x.pdf", format="pdf", bbox_inches="tight", pad_inches=0)  # pad_inches 控制边距
    plt.show()

    # subgraph_nodes = list(nx.single_source_shortest_path_length(G, query_index[0], cutoff=2).keys())
    # selected_candidates = GlobalSearch_test([a_query_node_id.tolist()], query_score[i].tolist(),G )  #算法5


    print("done")





