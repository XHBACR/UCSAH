import torch
from utils import f1_score_calculation, cosin_similarity, evaluation
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

# def testPhase(community_embedding, class_prediction, args): 

#     query=torch.load(f'./dataset/IMDB/imdb_test_node_id.pt') 
#     labels=torch.load(f'./dataset/IMDB/imdb_test_community.pt') 
#     adj=torch.load(f'./dataset/IMDB/imdb_con_adj.pt') 
#     # start = time.time()
#     query_feature = community_embedding[query] 
#     G = nx.from_numpy_array(adj.to_dense().numpy())

#     query_score = cosin_similarity(query_feature, community_embedding) 

#     # print("query_score.shape: ", query_score.shape)

#     y_pred = torch.zeros_like(query_score) 
#     for i in tqdm(range(query_score.shape[0])): 
#         selected_candidates = McCommunitySearch([query[i].tolist()],class_prediction, query_score[i].tolist(),G)  
#         for j in range(len(selected_candidates)):
#             y_pred[i][selected_candidates[j]] = 1 

#     # end = time.time()
#     # print("The search using time: {:.4f}".format(end-start)) 
#     f1_score = f1_score_calculation(y_pred.int(), labels.int())

#     # print("F1 score by maximum weight gain: {:.4f}".format(f1_score))

#     nmi, ari, jac = evaluation(y_pred.int(), labels.int())
    
#     # print("NMI score by maximum weight gain: {:.4f}".format(nmi))
#     # print("ARI score by maximum weight gain: {:.4f}".format(ari))
#     # print("JAC score by maximum weight gain: {:.4f}".format(jac))

#     print("F1:{:.4f} ".format(f1_score)+" NMI:{:.4f} ".format(nmi)+" ARI:{:.4f} ".format(ari)+" JAC:{:.4f} ".format(jac))
#     return  f1_score


def epoch_evaluate(community_embedding, class_prediction, query, labels ,G,args): 

    node_num_in_community = labels.sum(dim=1).int().tolist() 
    # start = time.time()
    query_feature = community_embedding[query] 
    query_score = cosin_similarity(query_feature, community_embedding) 

    y_pred = torch.zeros_like(query_score)
    for i in tqdm(range(query_score.shape[0])): 
        selected_candidates = McCommunitySearch([query[i].tolist()],class_prediction, query_score[i].tolist(),G)  
        for j in range(len(selected_candidates)):
            y_pred[i][selected_candidates[j]] = 1
        
    # end = time.time()
    # print("The search using time: {:.4f}".format(end-start)) 
    f1_score = f1_score_calculation(y_pred.int(), labels.int())
    return  f1_score

def subgraph_density_controled(candidate_score, graph_score): 
    
    weight_gain = (sum(candidate_score)-sum(graph_score)*(len(candidate_score)**1)/(len(graph_score)**1))/(len(candidate_score)**0.68)
    return weight_gain


# Multi-Constrained Community Search
def McCommunitySearch(query_index, class_prediction, graph_score , G): 
    filtered_nodes = [node for node in range(0,len(graph_score)) if graph_score[node] > 0.5]
    
    top_values = torch.topk(class_prediction[query_index], k=2).values  
    max_value, second_max_value = top_values[0][0], top_values[0][1] 
    if max_value - second_max_value > 0.3:
        query_class = torch.argmax(class_prediction[query_index], dim=1) 
        filtered_nodes = [
            node for node in filtered_nodes
            if torch.argmax(class_prediction[node]) == query_class
        ]

    high_score_subgraph = G.subgraph(filtered_nodes).copy()

    if query_index[0] in high_score_subgraph:
        connected_subgraph = nx.node_connected_component(high_score_subgraph, query_index[0])
        result_nodes = list(connected_subgraph)
    else:
        result_nodes = query_index

    return result_nodes




if __name__ == "__main__":
    args = search_parse_args()
    print(args)


