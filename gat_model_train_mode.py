import math
import time
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
#import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
#########################
from sklearn.model_selection import StratifiedKFold
import random
#import calculate_avg_acc_of_cross_validation_test
from sklearn import metrics
from scipy import stats
import torch.nn.functional as F
from torch.nn import Parameter
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden = 32
dropout = 0.5
lr = 0.01 
weight_decay = 1e-4
fastmode = 'store_true'


def encode_onehot(labels):
    #classes=set(labels)
    classes=sorted(list(set(labels)),reverse=True)
    classes_dict={c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot,classes_dict

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(mlp_or_not,graph,node_file,input_sample):
    ##### Load input sample info ######
    f=open(input_sample,'r')
    line=f.readline()
    train_id={}
    idx_train=[]
    idx_test=[]
    lid=0
    c=0
    tid2name={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()    
        if ele[-1]=='train':
            train_id['S'+ele[1]]=''
            if int(ele[1])>lid:
                lid=int(ele[1])
            idx_train.append(c)
        else:
            idx_test.append(c)
        tid2name[c].append(ele[2])
        c+=1





    print('Loading {} dataset...'.format(graph+' plus '+node_file))
    idx_features_labels = np.genfromtxt("{}".format(node_file),dtype=np.dtype(str))
    #print(idx_features_labels)
    features=idx_features_labels[:, 1:-1]
    features=features.astype(float)
    
    a=np.array(idx_features_labels[:, 1:-1])
    a=a.astype(float)
    #a=stats.zscore(a,axis=1,ddof=1)
    '''
    features=[]
    for s in a:
        mean=s.mean()
        std=s.std()
        features.append((s-mean)/std)
    '''
    features=np.array(features)
    #print(features)
    #exit()
    #features=a
    
    #print(features)
    #exit()
    
    #features = sp.csr_matrix(features, dtype=np.float32)
    #features = normalize(features)
    #features = torch.FloatTensor(np.array(features.todense()))
    #print(idx_features_labels[:, -1])
    #exit()
    labels,classes_dict = encode_onehot(idx_features_labels[:, -1])
    #features_train
    #labels_train
    #print(idx_features_labels[:, -1])
    #print(len(labels))
    #labels = torch.LongTensor(np.where(labels)[1])
    f1=features[idx_train]
    f2=features[idx_test]
    l1=labels[idx_train]
    l2=labels[idx_test]
    features=np.concatenate((f1, f2), axis=0)
    labels=np.concatenate((l1, l2), axis=0)
    #print(features)
    #print(labels)
    #exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}".format(graph),dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #### identity matrix
    if mlp_or_not=='mlp':
        adj=sp.identity(len(labels)).toarray()
        adj=sp.csr_matrix(adj)
    else:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    #print(adj.shape)
    #print(adj)
    #exit()
    
    total_num=len(labels)
    #tnum=int(3*(total_num/4))

    #idx_train = range(tnum)
    #idx_val = range(tnum,total_num)
    #print(idx_train)
    #print(idx_val)
    #exit()
    idx_test = range(len(idx_train), len(labels))
    #print(len(idx_train))
    #print(idx_test)
    #exit()

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    features_train=features[:len(idx_train)]
    labels_train=labels[:len(idx_train)]

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    #idx_train = torch.LongTensor(idx_train)
    #idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    #print(labels)
    return adj, features, labels, features_train,labels_train, idx_test,idx_train,classes_dict,tid2name

#adj, features, labels, idx_train, idx_val, idx_test=load_data()
#splits=KFold(n_splits=10,shuffle=True,random_state=1234)

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = Parameter(torch.FloatTensor(in_features, out_features))
        # 分别计算节点自身的注意力权重和邻居的注意力权重
        self.a_self = Parameter(torch.FloatTensor(out_features, 1))
        self.a_neigh = Parameter(torch.FloatTensor(out_features, 1))
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # 添加批归一化层
        self.bn = nn.BatchNorm1d(out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_neigh.data, gain=1.414)

    def forward(self, input, adj):
        # 确保输入维度正确
        if input.dim() == 1:
            input = input.unsqueeze(1)
        
        h = torch.mm(input, self.W)
        N = h.size(0)

        # 改进的注意力机制计算方式，分别计算自身和邻居的重要性
        a_self_input = torch.matmul(h, self.a_self)
        a_neigh_input = torch.matmul(h, self.a_neigh)
        
        # 注意力系数计算
        e = a_self_input + a_neigh_input.T
        e = self.leakyrelu(e)

        # 处理稀疏邻接矩阵
        if adj.is_sparse:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj
            
        # 创建注意力掩码，保留自连接
        zero_vec = -9e15 * torch.ones_like(e)
        # 允许自连接，这样每个节点至少关注自己
        attention = torch.where(adj_dense > 0, e, zero_vec)
        
        # 数值稳定性增强：限制注意力权重的范围
        attention = torch.clamp(attention, -10, 10)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, h)
        # 添加批归一化
        h_prime = self.bn(h_prime)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads=4):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nheads = nheads
        self.nclass = nclass
        
        # 添加输入层的批归一化
        self.input_bn = nn.BatchNorm1d(nfeat)
        
        # 增加网络深度，添加中间层
        self.attentions1 = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, concat=True) 
                                         for _ in range(nheads)])
        
        self.attentions2 = nn.ModuleList([GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, concat=True) 
                                         for _ in range(nheads)])
        
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, concat=False)
        
        # 添加残差连接的权重
        self.residual_weight = nn.Parameter(torch.FloatTensor(1))
        # 初始化残差权重
        nn.init.constant_(self.residual_weight, 0.5)

    def forward(self, x, adj):
        # 确保输入维度正确
        if x.dim() == 1:
            x = x.unsqueeze(1)
            
        # 输入批归一化
        x = self.input_bn(x)
            
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 第一层多头注意力
        x1 = torch.cat([att(x, adj) for att in self.attentions1], dim=1)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        
        # 第二层多头注意力（增加网络深度）
        x2 = torch.cat([att(x1, adj) for att in self.attentions2], dim=1)
        x2 = F.dropout(x2, self.dropout, training=self.training)
        
        # 添加残差连接
        if x1.size(1) == x2.size(1):
            x2 = self.residual_weight * x1 + (1 - self.residual_weight) * x2
        
        # 输出层注意力
        x = F.elu(self.out_att(x2, adj))
        
        # 添加数值稳定性检查
        x = torch.clamp(x, -10, 10)
        
        return F.log_softmax(x, dim=1)

def accuracy(output,labels):
    preds=output.max(1)[1].type_as(labels)
    correct=preds.eq(labels).double()
    correct=correct.sum()
    return correct/len(labels)

def AUC(output,labels):
    #print(output.data.numpy())
    output=torch.exp(output)
    a=output.data.numpy()
    
    # 检查是否有NaN或无穷大值
    if np.isnan(a).any() or np.isinf(a).any():
        print("Warning: NaN or Inf values found in output")
        # 将NaN替换为0，将Inf替换为有限值
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    
    preds=a[:,1]
    #exit()
    #preds=output.max(1)[0].data.numpy()
    #print(preds,output.max(1)[1])
    #print(labels)
    #exit()
    #preds=output.max(1)[1].type_as(labels)
    #print(np.array(preds),np.array(labels))
    #exit()
    try:
        fpr,tpr,thresholds=metrics.roc_curve(np.array(labels),np.array(preds))
        auc=metrics.auc(fpr,tpr)
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        print(f"Labels: {np.array(labels)}")
        print(f"Predictions: {np.array(preds)}")
        auc = 0.5  # 返回随机猜测的AUC值
    #print(fpr,tpr)
    #exit()
    return auc

def train_fs(epoch,idx_train_in,idx_val_in,model,optimizer,features,adj,labels,o,max_val_auc,rdir,fold,classes_dict,tid2name,record):
    #model.to(device).super().reset_parameters()
    #model = GCN(nfeat=features.shape[1], nhid=hidden, nclass=labels.max().item() + 1, dropout=dropout)
    #optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
    t=time.time()
    model.train()
    optimizer.zero_grad()
    output=model(features,adj)
    
    # 添加梯度裁剪防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    loss_train=torch.nn.functional.nll_loss(output[idx_train_in], labels[idx_train_in])
    acc_train = accuracy(output[idx_train_in], labels[idx_train_in])
    auc_train=AUC(output[idx_train_in], labels[idx_train_in])
    loss_train.backward()
    optimizer.step()

    #if not fastmode:
    model.eval()
    output=model(features,adj)
    loss_val = torch.nn.functional.nll_loss(output[idx_val_in], labels[idx_val_in])
    acc_val = accuracy(output[idx_val_in], labels[idx_val_in])
    auc_val = AUC(output[idx_val_in], labels[idx_val_in])
    print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.4f}'.format(loss_train.item()),'acc_train: {:.4f}'.format(acc_train.item()),'loss_val: {:.4f}'.format(loss_val.item()),'acc_val: {:.4f}'.format(acc_val.item()),'time: {:.4f}s'.format(time.time() - t),'AUC_train: {:.4f}'.format(auc_train.item()),'AUC_val: {:.4f}'.format(auc_val.item()))
    o.write('Epoch: {:04d}'.format(epoch+1)+' loss_train: {:.4f}'.format(loss_train.item())+' acc_train: {:.4f}'.format(acc_train.item())+' loss_val: {:.4f}'.format(loss_val.item())+' acc_val: {:.4f}'.format(acc_val.item())+' time: {:.4f}s'.format(time.time() - t)+' AUC_train: {:.4f}'.format(auc_train.item())+' AUC_val: {:.4f}'.format(auc_val.item())+'\n')
    
    return auc_train,torch.exp(output).data.numpy()

def train(epoch,idx_train_in,idx_val_in,model,optimizer,features,adj,labels,o,max_val_auc,rdir,fold,classes_dict,tid2name,record):
    #model.to(device).super().reset_parameters()
    #model = GCN(nfeat=features.shape[1], nhid=hidden, nclass=labels.max().item() + 1, dropout=dropout)
    #optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
    t=time.time()
    model.train()
    optimizer.zero_grad()
    output=model(features,adj)
    
    # 添加梯度裁剪防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    loss_train=torch.nn.functional.nll_loss(output[idx_train_in], labels[idx_train_in])
    acc_train = accuracy(output[idx_train_in], labels[idx_train_in])
    auc_train=AUC(output[idx_train_in], labels[idx_train_in])
    loss_train.backward()
    optimizer.step()

    #if not fastmode:
    model.eval()
    output=model(features,adj)
    loss_val = torch.nn.functional.nll_loss(output[idx_val_in], labels[idx_val_in])
    acc_val = accuracy(output[idx_val_in], labels[idx_val_in])
    auc_val = AUC(output[idx_val_in], labels[idx_val_in])
    print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.4f}'.format(loss_train.item()),'acc_train: {:.4f}'.format(acc_train.item()),'loss_val: {:.4f}'.format(loss_val.item()),'acc_val: {:.4f}'.format(acc_val.item()),'time: {:.4f}s'.format(time.time() - t),'AUC_train: {:.4f}'.format(auc_train.item()),'AUC_val: {:.4f}'.format(auc_val.item()))
    o.write('Epoch: {:04d}'.format(epoch+1)+' loss_train: {:.4f}'.format(loss_train.item())+' acc_train: {:.4f}'.format(acc_train.item())+' loss_val: {:.4f}'.format(loss_val.item())+' acc_val: {:.4f}'.format(acc_val.item())+' time: {:.4f}s'.format(time.time() - t)+' AUC_train: {:.4f}'.format(auc_train.item())+' AUC_val: {:.4f}'.format(auc_val.item())+'\n')
    if auc_val>max_val_auc and record==1:
        o3=open(rdir+'/sample_prob_fold'+str(fold)+'_val.txt','w+')
        output_res=torch.exp(output[idx_val_in])
        output_res=output_res.data.numpy()
        c=0
        dt={}
        for n in classes_dict:
            if int(classes_dict[n][0])==1:
                dt[0]=n
            else:
                dt[1]=n
        for a in output_res:
            nt=labels[idx_val_in[c]].data.numpy()
            o3.write(tid2name[int(idx_val_in[c])]+'\t'+str(a[0])+'\t'+str(a[1])+'\t'+str(labels[idx_val_in[c]].data.numpy())+'\t'+str(dt[int(nt)])+'\n')
            c+=1
    return auc_val



'''
def test():
    model.eval()
    output=model(features,adj)
    loss_test = torch.nn.functional.nll_loss(output[idx_test], labels[idx_test])
    preds=output[idx_test].max(1)[1].type_as(labels[idx_test])
    print(preds,labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:","loss= {:.4f}".format(loss_test.item()),"accuracy= {:.4f}".format(acc_test.item()))

def test_pred():
    model.eval()
    output=model(features,adj)
    preds=output[idx_test].max(1)[1].type_as(labels[idx_test])
    print(preds)
'''

#optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
'''
if device:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
'''

def test(model,idx_test,features,adj,labels,o,max_test_auc,rdir,fn,classes_dict,tid2name,record):
    model.eval()
    output=model(features,adj)
    loss_test=torch.nn.functional.nll_loss(output[idx_test], labels[idx_test])
    preds=output[idx_test].max(1)[1].type_as(labels[idx_test])
    #print(preds,labels[idx_test])
    #exit()
    acc_test=accuracy(output[idx_test],labels[idx_test])
    auc_test=AUC(output[idx_test], labels[idx_test])
    print(" | Test set results:","loss={:.4f}".format(loss_test.item()),"accuracy={:.4f}".format(acc_test.item()),"AUC={:.4f}".format(auc_test.item()))
    o.write(" | Test set results:"+"loss={:.4f}".format(loss_test.item())+" accuracy: {:.4f}".format(acc_test.item())+" AUC: {:.4f}".format(auc_test.item())+'\n')
    if auc_test>max_test_auc and record==1:
        o3=open(rdir+'/sample_prob_fold'+str(fn)+'_test.txt','w+')
        output_res=torch.exp(output[idx_test])
        output_res=output_res.data.numpy()
        c=0
        dt={}
        for n in classes_dict:
            if int(classes_dict[n][0])==1:
                dt[0]=n
            else:
                dt[1]=n
        for a in output_res:
            nt=labels[idx_test[c]].data.numpy()
            o3.write(tid2name[int(idx_test[c])]+'\t'+str(a[0])+'\t'+str(a[1])+'\t'+str(labels[idx_test[c]].data.numpy())+'\t'+str(dt[int(nt)])+'\n')
            c+=1
    return auc_test

def run_GCN_test(mlp_or_not,epochs,graph,node_file,outfile1,outfile2,input_sample):
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = 16
    lr = 0.01
    weight_decay = 5e-4
    fastmode = 'store_true'
    '''

    adj,features,labels,features_train,labels_train,idx_test,idx_train,classes_dict,tid2name=load_data(mlp_or_not,graph,node_file,input_sample)
    #print(adj)
    #exit()
    splits=StratifiedKFold(n_splits=10,shuffle=True,random_state=1234)

    #print(np.array(features).shape)
    #exit()
    epochs = epochs
    total_num=len(labels)
    o1=open(outfile1,'w+')
    fn=0
    for train_idx,val_idx in splits.split(np.array(features_train),np.array(labels_train)):
        #print('Fold {}'.format(fold+1))
        o1.write('Fold {}'.format(fn+1)+'\n')
        #print(train_idx,val_idx)
        #print(train_idx,val_idx)
        #exit()

        # 将原来的GCN模型替换为GAT模型
        model = GAT(nfeat=features.shape[1],  # 修复特征维度
                   nhid=32, 
                   nclass=labels.max().item() + 1, 
                   dropout=0.5,
                   nheads=4)  # 可以调整头数，常用值为1, 4, 8

# 优化器保持不变
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(epochs):
            train(epoch,train_idx,val_idx,model,optimizer,features,adj,labels,o1,0,'',0,classes_dict,tid2name,0)
        test(model,idx_test,features,adj,labels,o1,0,'',0,classes_dict,tid2name,0)
        fn+=1
        #test_pred()
    o1.close()
    #o2=open(outfile2,'w+')
    #calculate_avg_acc_of_cross_validation_test.cal_acc_cv(outfile1,outfile2)

####### Species style
#run_GCN_test('gcn',500,'Graph_File_test_last_raw_Sp/sp_pca_knn_graph_final_trans_USA.txt','Node_File/species_node_feature.txt','Res_record_Sp/r1_USA_sp_lasso_gcn.txt','Res_record_Sp/r2_USA_sp_lasso_gcn.txt','sample_USA_new.txt')

#run_GCN_test('gcn',500,'Graph_File_test_last_raw_Sp/sp_pca_knn_graph_final_trans_AUS.txt','Node_File/species_node_feature.txt','Res_record_Sp/r1_AUS_sp_lasso_gcn.txt','Res_record_Sp/r2_AUS_sp_lasso_gcn.txt','sample_AUS_new.txt')
#run_GCN_test('gcn',500,'Graph_File_test_last_raw_Sp/sp_pca_knn_graph_final_trans_China.txt','Node_File/species_node_feature.txt','Res_record_Sp/r1_China_sp_lasso_gcn.txt','Res_record_Sp/r2_China_sp_lasso_gcn.txt','sample_China_new.txt')
#run_GCN_test('gcn',500,'Graph_File_test_last_raw_Sp/sp_pca_knn_graph_final_trans_Denmark.txt','Node_File/species_node_feature.txt','Res_record_Sp/r1_Denmark_sp_lasso_gcn.txt','Res_record_Sp/r2_Denmark_sp_lasso_gcn.txt','sample_Denmark_new.txt')
#run_GCN_test('gcn',500,'Graph_File_test_last_raw_Sp/sp_pca_knn_graph_final_trans_French.txt','Node_File/species_node_feature.txt','Res_record_Sp/r1_French_sp_lasso_gcn.txt','Res_record_Sp/r2_French_sp_lasso_gcn.txt','sample_French_new.txt')

### eggNOG style
#run_GCN_test('gcn',500,'Graph_File_test_last_raw/eggNOG_pca_knn_graph_final_trans_Denmark.txt','Node_File/species_node_feature.txt','Res_record_retest_ECE/r1_Denmark_eggNOG_lasso_raw.txt','Res_record_retest_ECE/r2_Denmark_eggNOG_lasso_raw.txt','sample_Denmark_new.txt')