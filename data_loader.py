import utils
import dgl
import torch
import scipy.sparse as sp

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import  AmazonCoBuyPhotoDataset,CoauthorCSDataset,CoauthorPhysicsDataset



def get_dataset(dataset, pe_dim):
    if dataset in {"pubmed", "photo", "cs", "cora", "physics","citeseer"}:
        if dataset in {"photo", "cs"}:
            file_path = "dataset/"+dataset+"_dgl.pt"
        else:
            file_path = "dataset/"+dataset+"_pyg.pt"
        # file_path = "dataset/"+dataset+".pt"
        data_list = torch.load(file_path)
        
        adj = data_list[0] #邻接矩阵 [2708, 2708]
        
        features = data_list[1] #特征矩阵 [2708, 1433]
        
        if dataset == "pubmed":
            graph = PubmedGraphDataset()[0]
        elif dataset == "photo":
            graph = AmazonCoBuyPhotoDataset()[0]
        elif dataset == "cs":
            graph = CoauthorCSDataset()[0]
        elif dataset == "physics":
            graph = CoauthorPhysicsDataset()[0]
        elif dataset == "cora":
            graph = CoraGraphDataset()[0]
        elif dataset == "citeseer":
            graph = CiteseerGraphDataset()[0]

        # print(graph)
        graph = dgl.to_bidirected(graph) #转换为无向图
        
        lpe = utils.laplacian_positional_encoding(graph, pe_dim) #拉普拉斯位置编码向量 torch.Size([2708, 3])
     
        features = torch.cat((features, lpe), dim=1) #[2708, 1436]  1433+3
    
    
    elif dataset in {"texas", "cornell", "wisconsin", "dblp", "reddit"}:
        file_path = "dataset/"+dataset+"_pyg.pt"

        data_list = torch.load(file_path)
       
        adj = data_list[0]
        
        features = data_list[1]
        
        adj_scipy = utils.torch_adj_to_scipy(adj)
        graph = dgl.from_scipy(adj_scipy)
        lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
        features = torch.cat((features, lpe), dim=1)
    
    

    print(type(adj), type(features))
    
    return adj.cpu().type(torch.LongTensor), features.long()




