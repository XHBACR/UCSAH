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
from evaluation import epoch_evaluate  



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





    print("Finish pretrain process!")
    


