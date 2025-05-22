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
from accuracy_globalsearch import epoch_evaluate , evaluate 



if __name__ == "__main__":

    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(args.seed)

    adjs = torch.load(f'./dataset/{args.dataset}/{args.dataset}_adjs.pt') #(2, 4780, 4780) 稀疏tensor
    features = torch.load(f'./dataset/{args.dataset}/{args.dataset}_node_features.pt') #(2, 4780, 1232) 稀疏tensor
    node_class = torch.load(f'./dataset/{args.dataset}/{args.dataset}_node_class.pt') #(4780, 3)
    query_node=torch.load(f'./dataset/{args.dataset}/query_node.pt')
    true_community = torch.load(f'./dataset/{args.dataset}/true_community_of_query.pt')
    home_adj = torch.load(f'./dataset/IMDB/imdb_con_adj.pt')
    G = nx.from_numpy_array(home_adj.to_dense().numpy())
    test_query_node=query_node[:200] #num
    test_true_community = true_community.to_dense()[:200] #num
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
    print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
                    optimizer,
                    warmup_updates=args.warmup_updates,
                    tot_updates=args.tot_updates,
                    lr=args.peak_lr,
                    end_lr=args.end_lr,
                    power=1.0)
    print("starting training...")
    # model train
    model.train()

    t_start = time.time()

    # loss_train_b = []
    for epoch in range(args.epochs):
        epoch_loss = 0
        all_community_emb=[]
        for index, item in enumerate(data_loader):

            nodes_features = item.to(args.device) #[1000, 2, 5, 1232]
            
            adj_set, minus_adj_set=[] , []
            for i in range(0,args.metapath_num): #[0,2)
                adj_set.append(adj_batch[i][index])  # append[1000,1000]
                minus_adj_set.append(minus_adj_batch[i][index])
            adj_set = torch.stack(adj_set, dim=0) #[2, 1000, 1000]
            minus_adj_set = torch.stack(minus_adj_set, dim=0) #[2, 1000, 1000]

            optimizer.zero_grad()
            community_emb, loss_train = model.trainModel(nodes_features,  adj_set, minus_adj_set, args.metapath_num) # [1000, 512] , []    <- [1000, 2, 5, 1232]

            # print(node_tensor.shape, neighbor_tensor.shape, adj_.shape, minus_adj.shape)
            # loss_train = model.contrastive_link_loss(node_tensor, neighbor_tensor, adj_set, minus_adj_set) #算法2
            epoch_loss += loss_train.item()

            loss_train.backward()
            optimizer.step()
            lr_scheduler.step()
            all_community_emb.append(community_emb)
            # loss_train_b.append(loss_train.item())
        all_community_emb = torch.cat(all_community_emb, dim=0) #[4780, 512]
        # result_f1= evaluate(all_community_emb, args) #[4780, 512]
        result_f1= epoch_evaluate(all_community_emb,test_query_node,test_true_community, G,args)

        # result_f1= epoch_evaluate(all_community_emb, args) #[4780, 512]
        print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.4f}'.format(epoch_loss))
        # print("1")
        # torch.save(model.state_dict(), args.save_path + args.model_name + '.pth')

    # model.eval()
    # all_community_emb = torch.cat(all_community_emb, dim=0) #[4780, 512]
    # result_f1= epoch_evaluate(all_community_emb,test_query_node,test_true_community, G,args)
    
    print("1")





#==============================================================================


    

    print("stop")









#==============================================================================
    print("load data")  
    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    adj, features = get_dataset(args.dataset, args.pe_dim) #[2708, 2708] , [2708, 1436]
    

    start_feature_processing = time.time()
    processed_features = utils.re_features(adj, features, args.hops)  #[2708, 6, 1436]   return (N, hops+1, d)
    if processed_features.shape[0] < 10000:
        indicator = utils.conductance_hop(adj, args.hops) # [2708, 6] 3.2 Augmented Subgraph Sampler   return (N, hops+1)
        indicator = indicator.unsqueeze(2).repeat(1, 1, features.shape[1]) #[2708, 6, 1436]
        processed_features = processed_features*indicator  # [2708, 6, 1436]  将处理后的特征矩阵与导电性矩阵逐元素相乘，得到最终的特征矩阵。
    t_feature_precessing = time.time() - start_feature_processing
    print("feature process time: {:.4f}s".format(t_feature_precessing))

    start = time.time()
    print("starting transformer to coo")
    adj = transform_coo_to_csr(adj) # 2708x2708  transform coo(tensor) to csr to support slicing operation
    print("start mini batch processing")

    # [[2708, 2708]] , [[2708, 2708]]  划分好的密集tensor矩阵, 该例子中batch_size=node_num了,所以只划分出来一个batch
    adj_batch, minus_adj_batch = transform_sp_csr_to_coo(adj, args.batch_size, features.shape[0]) # transform to coo to support tensor operation
    print(len(adj_batch[0]), len(minus_adj_batch[0]))
    print("adj process time: {:.4f}s".format(time.time() - start))
    
    #                               [2708, 6, 1436]
    data_loader = Data.DataLoader(processed_features, batch_size=args.batch_size, shuffle = False)

    # model configuration
    model = PretrainModel(input_dim=processed_features.shape[2], config=args).to(args.device)

    print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
                    optimizer,
                    warmup_updates=args.warmup_updates,
                    tot_updates=args.tot_updates,
                    lr=args.peak_lr,
                    end_lr=args.end_lr,
                    power=1.0)
    
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(model, **stopping_args)

    print("starting training...")
    # model train
    model.train()

    t_start = time.time()

    loss_train_b = []
    for epoch in range(args.epochs):
        for index, item in enumerate(data_loader):

            start_index = index*args.batch_size
            nodes_features = item.to(args.device) #[2708, 6, 1436]
            adj_ = adj_batch[index].to(args.device)  #[2708, 2708]
            minus_adj = minus_adj_batch[index].to(args.device) #[2708, 2708]
            # print(nodes_features.shape)
            optimizer.zero_grad()
            node_tensor, neighbor_tensor = model(nodes_features) # [2708, 512], [2708, 512] 算法1的输出

            # print(node_tensor.shape, neighbor_tensor.shape, adj_.shape, minus_adj.shape)
            loss_train = model.contrastive_link_loss(node_tensor, neighbor_tensor, adj_, minus_adj) #算法2
            loss_train.backward()
            optimizer.step()
            lr_scheduler.step()
            loss_train_b.append(loss_train.item())
            # break
        embedding_tensor = torch.from_numpy(np.concatenate((node_tensor.cpu().detach().numpy(), neighbor_tensor.cpu().detach().numpy()), axis=1))
        result_f1= epoch_evaluate(embedding_tensor, args)


        if early_stopping.simple_check(loss_train_b):
            break

        print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()))
        # 'loss_train: {:.4f}'.format(np.mean(np.array(loss_train_b)))
    
    print("Optimization Finished!")
    print("Train time: {:.4f}s".format(time.time() - t_start + t_feature_precessing))

    # model save
    print("Start Save Model...")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    if not os.path.exists(args.embedding_path):
        os.makedirs(args.embedding_path)

    torch.save(model.state_dict(), args.save_path + args.model_name + '.pth')
    
    # obtain all the node embedding from the learned model
    model.eval()
    node_embedding = []
    for _, item in enumerate(data_loader):
        nodes_features = item.to(args.device)
        node_tensor, neighbor_tensor = model(nodes_features)
        if len(node_embedding) == 0:
            node_embedding = np.concatenate((node_tensor.cpu().detach().numpy(), neighbor_tensor.cpu().detach().numpy()), axis=1)
            # node_embedding = node_tensor.cpu().detach().numpy()
        else:
            new_node_embedding = np.concatenate((node_tensor.cpu().detach().numpy(), neighbor_tensor.cpu().detach().numpy()), axis=1)
            # new_node_embedding = node_tensor.cpu().detach().numpy()
            node_embedding = np.concatenate((node_embedding, new_node_embedding), axis=0)
    
    np.save(args.embedding_path + args.model_name + '.npy', node_embedding)







    print("Finish pretrain process!")
    


