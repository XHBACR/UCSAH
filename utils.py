import argparse
import torch
import scipy.sparse as sp
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import dgl
from sklearn.metrics import f1_score
import scipy.sparse as sp
import numpy as np
import networkx as nx
from numpy import *
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, jaccard_score


import time


# Training settings
def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='imdb', help='dataset name')
    # parser.add_argument('--device', type=int, default=1, help='Device cuda id')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=666, 
                        help='Random seed.')
    parser.add_argument('--metapath_num', type=int, default=2, 
                        help='numbers of metapath.')
    # model parameters
    parser.add_argument('--hops', type=int, default=2,
                        help='Hop of neighbors to be calculated')
    parser.add_argument('--pe_dim', type=int, default=5,
                        help='position embedding size')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer size')
    parser.add_argument('--ffn_dim', type=int, default=64,
                        help='FFN layer size')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')
    parser.add_argument('--readout', type=str, default="mean")
    parser.add_argument('--alpha', type=float, default=0.1, 
                        help='the value the balance the loss.')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=4780,
                        help='Batch size')
    parser.add_argument('--group_epoch_gap', type=int, default=20,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--tot_updates',  type=int, default=1000,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=400,
                        help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.001, 
                        help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001, 
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--patience', type=int, default=50, 
                        help='Patience for early stopping')
    
    # model saving
    parser.add_argument('--save_path', type=str, default='./model/',
                        help='The path for the model to save')
    parser.add_argument('--model_name', type=str, default='imdb',
                        help='The name for the model to save')
    parser.add_argument('--embedding_path', type=str, default='./pretrain_result/',
                        help='The path for the embedding to save')
    


    parser.add_argument('--embedding_tensor_name', type=str, help='embedding tensor name')
    parser.add_argument('--EmbeddingPath', type=str, default='./pretrain_result/', help='embedding path')
    parser.add_argument('--topk', type=int, default=400, help='the number of nodes selected.')

    return parser.parse_args()


def laplacian_positional_encoding(adj, pos_enc_dim):
    """
    Graph positional encoding with Laplacian eigenvectors.
    
    Parameters:
    adj (scipy.sparse.csr_matrix): Sparse adjacency matrix.
    pos_enc_dim (int): Dimension of the positional encoding.
    
    Returns:
    torch.Tensor: Laplacian positional encoding.
    """

    # Laplacian
    N = sp.diags(1.0 / (adj.sum(axis=1).A1 + 1e-10) ** 0.5, dtype=float)  # Degree matrix inverse square root
    L = sp.eye(adj.shape[0]) - N @ adj @ N  # Normalized Laplacian

    # Eigenvectors with scipy
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)  # Compute eigenvalues and eigenvectors
    EigVec = EigVec[:, EigVal.argsort()]  # Sort eigenvectors by eigenvalues
    lap_pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()  # Select the first pos_enc_dim eigenvectors

    return lap_pos_enc  # Return the positional encoding as a torch tensor


def re_features(adj, features, K):#[2708, 2708] , [2708, 1436] , 5
    #传播之后的特征矩阵,size= (N, 1, K+1, d )  torch.Size([2708, 1, 6, 1436])
    nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])

    for i in range(features.shape[0]):

        nodes_features[i, 0, 0, :] = features[i] #torch.Size([2708, 1, 6, 1436])

    x = features + torch.zeros_like(features) #[2708, 1436]
    
    for i in range(K):  #k=5 所有节点构建k个特征,代表k条邻居内的特征

        x = torch.matmul(adj, x)

        for index in range(features.shape[0]):

            nodes_features[index, 0, i + 1, :] = x[index]  #[2708, 1, 6, 1436]  

    nodes_features = nodes_features.squeeze()


    return nodes_features #[2708, 6, 1436]

def conductance_hop(adj, max_khop): #[2708, 2708] ,  5
    adj = adj.to(dtype=torch.float)
    adj_current_hop = adj

    results = torch.zeros((max_khop+1, adj.shape[0])) #[6, 2708]
    for hop in range(max_khop+1):
        adj_current_hop = torch.matmul(adj_current_hop, adj) #[2708, 2708]
        degree = torch.sum(adj_current_hop, dim=0) #[2708]
        adj_current_hop_sign = torch.sign(adj_current_hop) #[2708, 2708]
        degree_1 = torch.sum(adj_current_hop_sign, dim=0) #[2708] 计算符号矩阵的度
        results[hop] = (degree-degree_1).to_dense().reshape(1, -1) # 计算导电性并存储在结果矩阵中
        hop += 1
    results = results.T #[2708, 6]
    max_indices = torch.argmax(results, dim=1) #[2708] 找到每个节点的最大导电性索引
    
    for i in range(results.shape[0]): #range(0, 2708)
        for j in range(results.shape[1]):    #range(0, 6)    
            if j>max_indices[i] and max_indices[i] != 0:
                results[i][j] = 0  # 如果当前跳数大于最大导电性索引且最大导电性索引不为0，则将结果置为0
            else:
                results[i][j] = 1 # 否则将结果置为1
    if hop==1:
        results==torch.ones((max_khop+1, adj.shape[0])) # 如果只有一跳，则将结果矩阵置为全1
    return results #[2708, 6]  是一个矩阵，用于存储每个节点在不同跳数（hop）下的导电性（conductance）值, 根据最大导电性跳数，将 results 矩阵二值化（0或1），用于进一步的图分析或特征处理。


def f1_score_calculation(y_pred, y_true):  # [100, 2708], [100, 2708]
    F1_scores = []
    for i in range(y_pred.shape[0]):  # 遍历每一条
        y_pred_i = y_pred[i].reshape(1, -1)
        y_true_i = y_true[i].reshape(1, -1)
        pre = torch.sum(torch.multiply(y_pred_i, y_true_i)) / (torch.sum(y_pred_i) + 1E-9)
        rec = torch.sum(torch.multiply(y_pred_i, y_true_i)) / (torch.sum(y_true_i) + 1E-9)
        F1 = 2 * pre * rec / (pre + rec + 1E-9)
        F1_scores.append(F1.item())  # 将每条的 F1 分数添加到列表中
    return sum(F1_scores) / len(F1_scores)  # 返回平均 F1 分数

def f1_score_calculation_test (y_pred, y_true): #[100, 2708], [100, 2708]
    y_pred = y_pred.reshape(1, -1)
    y_true = y_true.reshape(1, -1)
    pre = torch.sum(torch.multiply(y_pred, y_true))/(torch.sum(y_pred)+1E-9)
    rec = torch.sum(torch.multiply(y_pred, y_true))/(torch.sum(y_true)+1E-9)
    F1 = 2 * pre * rec / (pre + rec+1E-9)
    # print("recall: ", rec, "pre: ", pre)
    return F1


def evaluation(comm_find, comm):
    
    comm_find = comm_find.reshape(-1)
    comm = comm.reshape(-1)
    
    return normalized_mutual_info_score(comm, comm_find), adjusted_rand_score(comm, comm_find), jaccard_score(comm, comm_find)



def NMI_score(comm_find, comm):

    score = normalized_mutual_info_score(comm, comm_find)
    #print("q, nmi:", score)
    return score

def ARI_score(comm_find, comm):

    score = adjusted_rand_score(comm, comm_find)
    #print("q, ari:", score)

    return score

def JAC_score(comm_find, comm):

    score = jaccard_score(comm, comm_find)
    #print("q, jac:", score)
    return score

def load_query_n_gt(path, dataset, vec_length):
    # load query and ground truth
    query = []
    file_query = open(path + dataset + '/' + dataset + ".query", 'r')
    for line in file_query:
        vec = [0 for i in range(vec_length)]  # [2708]
        line = line.strip()
        line = line.split(" ")
        for i in line:  #本次查询节点(该line的)独热编码
            vec[int(i)] = 1  # [2708]
        query.append(vec)

    gt = []
    file_gt = open(path + dataset + '/' + dataset + ".gt", 'r')
    for line in file_gt:
        vec = [0 for i in range(vec_length)]
        line = line.strip()
        line = line.split(" ")
        
        for i in line:
            vec[int(i)] = 1
        gt.append(vec)
    
    return torch.Tensor(query), torch.Tensor(gt)

def get_gt_legnth(path, dataset):
    gt_legnth = []
    file_gt = open(path + dataset + '/' + dataset + ".gt", 'r')
    for line in file_gt:
        line = line.strip()
        line = line.split(" ")
        gt_legnth.append(len(line))
    
    return torch.Tensor(gt_legnth)

def cosin_similarity(query_tensor, emb_tensor):# [100, 1024], [2708, 1024]
    # similarity = torch.stack([torch.cosine_similarity(query_tensor[i], emb_tensor, dim=1) for i in range(len(query_tensor))], 0)
    similarity = torch.stack([torch.cosine_similarity(query_tensor[i].reshape(1, -1), emb_tensor, dim=1) for i in range(len(query_tensor))], 0)
    # print(similarity.shape)
    return similarity
    
def dot_similarity(query_tensor, emb_tensor):
    similarity = torch.mm(query_tensor, emb_tensor.t()) # (query_num, node_num)
    similarity = torch.nn.Softmax(dim=1)(similarity)
    return similarity


def transform_coo_to_csr(adj): #[2708, 2708]
    row=adj._indices()[0] #[10556]
    col=adj._indices()[1] #[10556]
    data=adj._values() #[10556]
    shape=adj.size()  #[2708, 2708]
    adj=sp.csr_matrix((data, (row, col)), shape=shape)
    return adj #<2708x2708 sparse matrix of type '<class 'numpy.int64'>'
	            # with 10556 stored elements in Compressed Sparse Row format>

def transform_csr_to_coo(adj, size=None):
    adj = adj.tocoo()
    adj = torch.sparse.LongTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                              torch.LongTensor(adj.data.astype(np.int32)),
                              torch.Size([size, size]))
    return adj

def transform_sp_csr_to_coo(adj, batch_size, node_num): #2708x2708 , 2708 , 2708
    # chunks
    node_index = [i for i in range(node_num)]# [0-2708]
    divide_index = [node_index[i:i+batch_size] for i in range(0, len(node_index), batch_size)] #按照batchsize划分节点索引 , 比如每组1000个节点

    # adj of each chunks, in the format of sp_csr
    print("start mini batch: adj of each chunks")
    adj_sp_csr = [adj[divide_index[i]][:, divide_index[i]] for i in range(len(divide_index))]# 按划分好的节点索引划分邻接矩阵
    print("start mini batch: minus adj of each chunks")
    minus_adj_sp_csr = [sp.csr_matrix(torch.ones(item.shape))-item for item in adj_sp_csr] #1减去邻接矩阵

    # adj_tensor_coo = [transform_csr_to_coo(item).to_dense() for item in adj_sp_csr]
    # minus_adj_tensor_coo = [transform_csr_to_coo(item).to_dense() for item in minus_adj_sp_csr]
    print("start mini batch: back to torch coo adj")
    adj_tensor_coo = [transform_csr_to_coo(adj_sp_csr[i], len(divide_index[i])).to_dense() for i in range(len(divide_index))]
    print("start mini batch: back to torch coo minus adj")
    minus_adj_tensor_coo = [transform_csr_to_coo(minus_adj_sp_csr[i], len(divide_index[i])).to_dense() for i in range(len(divide_index))]

    return adj_tensor_coo, minus_adj_tensor_coo  #都转化为密集矩阵了(不是靠索引定位的)


# transform coo to edge index in pytorch geometric 
def transform_coo_to_edge_index(adj):
    adj = adj.coalesce()
    edge_index = adj.indices().detach().long()
    return edge_index

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def torch_adj_to_scipy(adj):

    shape = adj.shape
    coords = adj.coalesce().indices()
    values = adj.coalesce().values()

    scipy_sparse = sp.coo_matrix((values.cpu().numpy(), (coords[0].cpu().numpy(), coords[1].cpu().numpy())), shape=shape)

    return scipy_sparse

# determine one edge in edge_index or not of torch geometric
def is_edge_in_edge_index(edge_index, source, target):
    mask = (edge_index[0] == source) & (edge_index[1] == target)
    return mask.any()

def construct_pseudo_assignment(cluster_ids_x):
    pseudo_assignment = torch.zeros(cluster_ids_x.shape[0], int(cluster_ids_x.max()+1))

    for i in range(cluster_ids_x.shape[0]):
        pseudo_assignment[i][int(cluster_ids_x[i])] = 1
    
    return pseudo_assignment

def pq_computation(similarity):
    q = torch.nn.functional.normalize(similarity, dim=1, p=1)
    p_temp = torch.mul(q, q)
    q_colsum = torch.sum(q, axis=0)
    p_temp = torch.div(p_temp,q_colsum)
    p = torch.nn.functional.normalize(p_temp, dim=1, p=1)
    return q, p

def coo_matrix_to_nx_graph(matrix):
    # Create an empty NetworkX graph
    graph = nx.Graph()

    # Get the number of nodes in the COO matrix
    num_nodes = matrix.shape[0]

    # Convert the COO matrix to a dense matrix
    dense_matrix = matrix.to_dense()

    # Iterate over the non-zero entries in the dense matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            if dense_matrix[i][j] != 0:
                # Add an edge to the NetworkX graph
                graph.add_edge(i, j)
                graph.add_edge(j, i)

    return graph

def coo_matrix_to_nx_graph_efficient(adj_matrix):
    # 创建一个无向图对象
    graph = nx.Graph()

    # 获取 COO 矩阵的行和列索引以及权重值
    adj_matrix = adj_matrix.coalesce()
    rows = adj_matrix.indices()[0]
    cols = adj_matrix.indices()[1]

    # 添加节点和边到图中
    for i in range(len(rows)):
        graph.add_edge(int(rows[i]), int(cols[i]))
        graph.add_edge(int(cols[i]), int(rows[i]))

    return graph

def obtain_adj_from_nx(graph):
    return np.array(nx.adjacency_matrix(graph, nodelist=[i for i in range(max(graph.nodes)+1)]).todense())

def find_all_neighbors_bynx(query, Graph):
    
    nodes = Graph.nodes()

    neighbors = []
    for i in range(len(query)):
        if query[i] not in nodes:
            continue
        for j in Graph.neighbors(query[i]):
            if j not in query:
                neighbors.append(j)
    return neighbors

def MaxMinNormalization(x, Min, Max):
    
    x = np.array(x)
    x_max = np.max(x)
    x_min = np.min(x)

    x = [(item-x_min)*(Max-Min)/(x_max - x_min) + Min for item in x]

    return x

