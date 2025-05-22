import torch
from utils import f1_score_calculation, load_query_n_gt, cosin_similarity, get_gt_legnth, evaluation
import argparse
import numpy as np
from tqdm import tqdm
from numpy import *
import time
import networkx as nx

def search_parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()
    # main parameters
    parser.add_argument('--dataset', type=str, default='cora', help='dataset name')
    parser.add_argument('--embedding_tensor_name', type=str, help='embedding tensor name')
    parser.add_argument('--EmbeddingPath', type=str, default='./pretrain_result/', help='embedding path')
    parser.add_argument('--topk', type=int, default=400, help='the number of nodes selected.')

    return parser.parse_args()

def evaluate(embedding_tensor, args): # [4780, 512]

    query=torch.load(f'./dataset/IMDB/imdb_test_node_id.pt') #[200]
    labels=torch.load(f'./dataset/IMDB/imdb_test_community.pt') #[200, 4780]
    adj=torch.load(f'./dataset/IMDB/imdb_con_adj.pt') #[4780, 4780]
    # start = time.time()
    query_feature = embedding_tensor[query] # [200, 512]  (query_num, embedding_dim)
    G = nx.from_numpy_array(adj.to_dense().numpy())

    #算法3 
    similarity_score = cosin_similarity(query_feature, embedding_tensor) # [200, 4780]  <- [200, 512], [4780, 512]
    similarity_score = torch.nn.functional.normalize(similarity_score, dim=1, p=1) # [200, 4780]

    # print("query_score.shape: ", query_score.shape)

    y_pred = torch.zeros_like(similarity_score) #[200, 4780]
    for i in tqdm(range(similarity_score.shape[0])): #range(0, 200)
        selected_candidates = CommunitySearch(query[i].tolist(), similarity_score[i].tolist() , G)  #算法5
        for j in range(len(selected_candidates)):
            y_pred[i][selected_candidates[j]] = 1 #将预测的社区以独热编码的形式标记出来
        
    # end = time.time()
    # print("The global search using time: {:.4f}".format(end-start)) 
    # print("The global search using time (one query): {:.4f}".format((end-start)/query_feature.shape[0])) 
    f1_score = f1_score_calculation(y_pred.int(), labels.int())

    # print("F1 score by maximum weight gain: {:.4f}".format(f1_score))

    nmi, ari, jac = evaluation(y_pred.int(), labels.int())
    
    # print("NMI score by maximum weight gain: {:.4f}".format(nmi))
    # print("ARI score by maximum weight gain: {:.4f}".format(ari))
    # print("JAC score by maximum weight gain: {:.4f}".format(jac))

    print("F1:{:.4f} ".format(f1_score)+" NMI:{:.4f} ".format(nmi)+" ARI:{:.4f} ".format(ari)+" JAC:{:.4f} ".format(jac))
    return  f1_score


def epoch_evaluate(embedding_tensor,query, labels ,G,args): # [4780, 512]

    node_num_in_community = labels.sum(dim=1).int().tolist() #[200]
    # start = time.time()
    query_feature = embedding_tensor[query] # [200, 512]  (query_num, embedding_dim)
    

    #算法3 
    query_score = cosin_similarity(query_feature, embedding_tensor) # [200, 4780]  <- [200, 512], [4780, 512]
    # query_score = torch.nn.functional.normalize(query_score, dim=1, p=1) # [200, 4780]

    # print("query_score.shape: ", query_score.shape)

    y_pred = torch.zeros_like(query_score) #[200, 4780]
    for i in tqdm(range(query_score.shape[0])): #range(0, 200)
        selected_candidates = GlobalSearch_test([query[i].tolist()], query_score[i].tolist(),G)  #算法5
        for j in range(len(selected_candidates)):
            y_pred[i][selected_candidates[j]] = 1 #将预测的社区以独热编码的形式标记出来
        
    # end = time.time()
    # print("The global search using time: {:.4f}".format(end-start)) 
    # print("The global search using time (one query): {:.4f}".format((end-start)/query_feature.shape[0])) 
    f1_score = f1_score_calculation(y_pred.int(), labels.int())

    # print("F1 score by maximum weight gain: {:.4f}".format(f1_score))

    nmi, ari, jac = evaluation(y_pred.int(), labels.int())
    
    # print("NMI score by maximum weight gain: {:.4f}".format(nmi))
    # print("ARI score by maximum weight gain: {:.4f}".format(ari))
    # print("JAC score by maximum weight gain: {:.4f}".format(jac))

    print("F1:{:.4f} ".format(f1_score)+" NMI:{:.4f} ".format(nmi)+" ARI:{:.4f} ".format(ari)+" JAC:{:.4f} ".format(jac))
    return  f1_score

def subgraph_density_controled(candidate_score, graph_score): #计算ESG  7页左上角
    
    weight_gain = (sum(candidate_score)-sum(graph_score)*(len(candidate_score)**1)/(len(graph_score)**1))/(len(candidate_score)**0.68)
    return weight_gain

# def community_score(S, G, graph_score):
#     """
#     计算当前社区 S 的社区得分，综合考虑特征相似度和结构内聚性。
#     """
#     if len(S) < 2:
#         return 0  # 至少要有两个节点才能计算社区结构

#     # 计算特征相似性得分（社区内所有节点的平均相似度）
#     similarity_score = np.mean([graph_score[v] for v in S])

#     # 计算结构内聚性（边密度 = 实际边数 / 最大可能边数）
#     subgraph = G.subgraph(S)
#     edges_within = subgraph.number_of_edges()
#     max_possible_edges = len(S) * (len(S) - 1) / 2  # 完全图的最大边数
#     cohesion_score = edges_within / max_possible_edges if max_possible_edges > 0 else 0

#     # 线性加权求和，两者都很重要
#     alpha = 0.2  # 调节特征相似性和结构内聚性的权重
#     return alpha * similarity_score + (1 - alpha) * cohesion_score

# def CommunitySearch(query_index, graph_score, G):
#     """
#     从查询节点 query_index 出发，搜索特征相似且内聚性高的社区。
#     """
#     # 初始化社区，只包含查询节点
#     community = {query_index}
#     best_score = community_score(community, G, graph_score)
    
#     # 获取查询节点的邻居
#     neighbors = list(G.neighbors(query_index))
    
#     # 按特征相似度降序排列邻居
#     neighbors.sort(key=lambda v: graph_score[v], reverse=True)

#     for v in neighbors:
#         # 仅考虑与查询节点相邻的节点，逐步扩展
#         community.add(v)
#         new_score = community_score(community, G, graph_score)

#         if new_score > best_score:
#             best_score = new_score  # 更新最佳得分
#         else:
#             community.remove(v)  # 如果得分下降，则回退

#     return list(community)

def CommunitySearch(query_index, graph_score, G):  # [1,] , [4780,]  算法5
    candidates = query_index
    selected_candidate = candidates

    graph_score = np.array(graph_score)  # (4780,)
    max2min_index = np.argsort(-graph_score)  # 从大到小排序,返回索引

    # 根据查询节点在 G 中生成3跳子图
    subgraph_nodes = nx.single_source_shortest_path_length(G, query_index[0], cutoff=2).keys()
    subgraph = G.subgraph(subgraph_nodes).copy()

    # 计算子图内的相似度得分，并计算平均值
    subgraph_scores = graph_score[list(subgraph.nodes)]
    mean_score = np.mean(subgraph_scores)

    # 删除子图内相似度得分小于平均值的节点
    nodes_to_remove = [n for n in subgraph.nodes if graph_score[n] < mean_score]
    subgraph.remove_nodes_from(nodes_to_remove)

    # 选取从 query_index 出发，连接的更小的子图
    if query_index[0] in subgraph:
        connected_subgraph = nx.node_connected_component(subgraph, query_index[0])
        selected_candidate = list(connected_subgraph)

    # return selected_candidate  # 选择 score 大的节点,直到分割点
    return list(subgraph_nodes)

def GlobalSearch(query_index, graph_score):  # [1,] , [4780,]  算法5

    candidates = query_index
    selected_candidate = candidates

    graph_score=np.array(graph_score) #(2708,)
    max2min_index = np.argsort(-graph_score) #从大到小排序,返回索引
    
    startpoint = 0
    endpoint = int(0.50*len(max2min_index)) #1354
    if endpoint >= 10000:
        endpoint = 10000
    
    while True: #找到一个最佳分割点
        candidates_half = query_index+[max2min_index[i] for i in range(0, int((startpoint+endpoint)/2))]  #len() =680  加上前一半分数大的节点
        candidate_score_half = [graph_score[i] for i in candidates_half] #记录这些节点的score
        candidates_density_half = subgraph_density_controled(candidate_score_half, graph_score)

        candidates = query_index+[max2min_index[i] for i in range(0, endpoint)] #len() =1357
        candidate_score = [graph_score[i] for i in candidates]
        candidates_density = subgraph_density_controled(candidate_score, graph_score)

        if candidates_density>= candidates_density_half: #类似折半查找
            startpoint = int((startpoint+endpoint)/2)
            endpoint = endpoint
        else:
            startpoint = startpoint
            endpoint = int((startpoint+endpoint)/2)
        
        if startpoint == endpoint or startpoint+1 == endpoint:
            break

    selected_candidate = query_index+[max2min_index[i] for i in range(0, startpoint)] 
    
    return selected_candidate #选择score大的节点,直到分割点

def GlobalSearch_test(query_index, graph_score , G):  # [1,] , [4780,]  算法5
    # 根据查询节点在 G 中生成 2 跳邻居
    # subgraph_nodes = list(nx.single_source_shortest_path_length(G, query_index[0], cutoff=2).keys())

    # 收集二跳邻居内得分大于 0.3 的节点 index
    high_score_nodes = [node for node in range(0,len(graph_score)) if graph_score[node] > 0.5]

    # 根据这些节点构建子图
    high_score_subgraph = G.subgraph(high_score_nodes).copy()

    # 从查询节点出发，选择与查询节点连接的部分
    if query_index[0] in high_score_subgraph:
        connected_subgraph = nx.node_connected_component(high_score_subgraph, query_index[0])
        result_nodes = list(connected_subgraph)
    else:
        result_nodes = query_index

    # 返回最终的结果
    return result_nodes


    # # 根据查询节点在 G 中生成 2 跳邻居
    # subgraph_nodes = list(nx.single_source_shortest_path_length(G, query_index[0], cutoff=2).keys())


    # # 获取得分最高的 30 个节点
    # sorted_nodes = sorted(subgraph_nodes, key=lambda x: graph_score[x], reverse=True)
    # top_30_nodes = sorted_nodes[:30]

    # # 构建包含得分最高 30 个节点的子图
    # top_30_subgraph = G.subgraph(top_30_nodes).copy()

    # # 判断是否与查询节点连接
    # if query_index[0] in top_30_subgraph:
    #     connected_subgraph = nx.node_connected_component(top_30_subgraph, query_index[0])
    #     result_nodes = list(connected_subgraph)
    # else:
    #     result_nodes = query_index

    # # 返回最终的结果
    # return result_nodes






    # selected_graph_score = np.array([graph_score[i] for i in subgraph_nodes])
    # max2min_index = np.argsort(-selected_graph_score)  # 从大到小排序,返回索引
    # selected_node= [subgraph_nodes[i] for i in max2min_index[:int(select_number)]]

    # subgraph = G.subgraph(selected_node).copy()
    # connected_subgraph = nx.node_connected_component(subgraph, query_index[0])
    # return list(connected_subgraph) 


    # graph_score=np.array(graph_score) #(2708,)
    # max2min_index = np.argsort(-graph_score) #从大到小排序,返回索引
    
    # startpoint = 0
    # endpoint = int(0.50*len(max2min_index)) #1354
    # if endpoint >= 10000:
    #     endpoint = 10000
    
    # while True: #找到一个最佳分割点
    #     candidates_half = query_index+[max2min_index[i] for i in range(0, int((startpoint+endpoint)/2))]  #len() =680  加上前一半分数大的节点
    #     candidate_score_half = [graph_score[i] for i in candidates_half] #记录这些节点的score
    #     candidates_density_half = subgraph_density_controled(candidate_score_half, graph_score)

    #     candidates = query_index+[max2min_index[i] for i in range(0, endpoint)] #len() =1357
    #     candidate_score = [graph_score[i] for i in candidates]
    #     candidates_density = subgraph_density_controled(candidate_score, graph_score)

    #     if candidates_density>= candidates_density_half: #类似折半查找
    #         startpoint = int((startpoint+endpoint)/2)
    #         endpoint = endpoint
    #     else:
    #         startpoint = startpoint
    #         endpoint = int((startpoint+endpoint)/2)
        
    #     if startpoint == endpoint or startpoint+1 == endpoint:
    #         break

    # selected_candidate = query_index +[max2min_index[i] for i in range(0, startpoint)] 
    

    # subgraph = G.subgraph(selected_candidate).copy()
    # connected_subgraph = nx.node_connected_component(subgraph, query_index[0])

    # return list(connected_subgraph)  # 返回包含查询节点的连通部分

    # return selected_candidate #选择score大的节点,直到分割点

if __name__ == "__main__":
    args = search_parse_args()
    print(args)


    # 设置 embedding_tensor_name 的默认值
    if args.embedding_tensor_name is None:
        args.embedding_tensor_name = args.dataset
    

    embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.embedding_tensor_name + '.npy')) #[2708, 1024]
    f1_result = epoch_evaluate(embedding_tensor, args)




    # # [100, 2708], [100, 2708]   load queries and labels  
    # query, labels = load_query_n_gt("./dataset/", args.dataset, embedding_tensor.shape[0])

    # start = time.time()
    # query_feature = torch.mm(query, embedding_tensor) # [100, 1024]  (query_num, embedding_dim)
    
    # query_num = torch.sum(query, dim=1) #[100]  记录每次查询节点集里面有几个节点
    # query_feature = torch.div(query_feature, query_num.view(-1, 1))  # [100, 1024]   类似归一化, 因为前面存在加好几个节点的emb
    
    # # cosine similarity
    # print(query_feature.shape, embedding_tensor.shape)
    
    # #算法3 
    # query_score = cosin_similarity(query_feature, embedding_tensor) # [100, 2708]  <- [100, 1024], [2708, 1024]
    # query_score = torch.nn.functional.normalize(query_score, dim=1, p=1) # [100, 2708])

    # print("query_score.shape: ", query_score.shape)

    # y_pred = torch.zeros_like(query_score) #[100, 2708]
    # for i in tqdm(range(query_score.shape[0])): #range(0, 100)
    #     query_index = (torch.nonzero(query[i]).squeeze()).reshape(-1) #torch.Size([3])

    #     selected_candidates = GlobalSearch(query_index.tolist(), query_score[i].tolist())  #算法5
    #     for j in range(len(selected_candidates)):
    #         y_pred[i][selected_candidates[j]] = 1 #将预测的社区以独热编码的形式标记出来
        
    # end = time.time()
    # print("The global search using time: {:.4f}".format(end-start)) 
    # print("The global search using time (one query): {:.4f}".format((end-start)/query_feature.shape[0])) 
    # f1_score = f1_score_calculation(y_pred.int(), labels.int())

    # print("F1 score by maximum weight gain: {:.4f}".format(f1_score))

    # nmi, ari, jac = evaluation(y_pred.int(), labels.int())
    
    # print("NMI score by maximum weight gain: {:.4f}".format(nmi))
    # print("ARI score by maximum weight gain: {:.4f}".format(ari))
    # print("JAC score by maximum weight gain: {:.4f}".format(jac))
