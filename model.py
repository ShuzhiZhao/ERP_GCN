#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

###define model for training
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import spmm
from torch_scatter import scatter_add
##from torch_geometric.utils import add_self_loops
from utils import add_self_loops


class FCN(nn.Module):
###fully-connected
    def __init__(self, input_dim, output_dim,hidden_dim=128, dropout=0.2):
        super(FCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Define the output layer
        self.linear = nn.Linear(self.input_dim, hidden_dim)

        self.hidden2label = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim // 4, self.output_dim),
        )

    def forward(self, inputs, adj=None):
        if len(inputs.size())>2:
            ###average across all channels
            inputs = torch.mean(inputs, dim=-1)

        x = F.relu(self.linear(inputs))
        x = F.dropout(x, training=self.training,p=self.dropout)
        y_pred = self.hidden2label(x)
        return y_pred


class GraphConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    math:
      Z = D(-1/2)AstarD(-1/2)X; Astar=I+A
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size, *i.e.* number of hops :math:`K`.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """


    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        ####initialize all layer parameters
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self,  x, edge_index, edge_weight=None):
        batch, num_nodes = x.size(0), x.size(1)
        ##first adjust the adj matrix with diag elements
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, 1, num_nodes)
        row, col = edge_index
        
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)
        
        ###degree matrix
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        # Compute normalized and rescaled Laplacian.
        deg = deg.pow(-0.5)
        deg[torch.isinf(deg)] = 0
        lap = deg[row] * edge_weight * deg[col]
        ###Rescale the Laplacian eigenvalues in [-1, 1]
        #fill_value = 0.05  ##-0.5
        #edge_index, lap = add_self_loops(edge_index, lap, fill_value, num_nodes)

#         print('GC ',x.size(),'\n',self.weight.shape)
        x = torch.matmul(x, self.weight)
        out = spmm(edge_index, lap, num_nodes, x.permute(1, 2, 0).contiguous().view((num_nodes, -1))).view((num_nodes, -1, batch)).permute(2, 0,1)  # spmm(edge_index, lap, num_nodes, x)

        if self.bias is not None:
            out = out + self.bias

        return out
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
class ChebConv(nn.Module):
    """The chebyshev spectral graph convolutional operator
    .. math::
        \mathbf{Z}^{(0)} &= \mathbf{X}
        \mathbf{Z}^{(1)} &= \mathbf{\hat{L}} \cdot \mathbf{X}
        \mathbf{Z}^{(k)} &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{Z}^{(k-1)} - \mathbf{Z}^{(k-2)}
    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size, *i.e.* number of hops :math:`K`.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, K, bias=True):
        super(ChebConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(K+1, in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        ####initialize all layer parameters
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        # edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        # print(x.size(), edge_index.size())
        row, col = edge_index
        batch, num_nodes, num_edges, K = x.size(0), x.size(1), row.size(0), self.weight.size(0)
            
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)
        
        ###degree matrix
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        # Compute normalized and rescaled Laplacian.
        deg = deg.pow(-0.5)
        deg[torch.isinf(deg)] = 0
        lap = -deg[row] * edge_weight * deg[col]
        ###Rescale the Laplacian eigenvalues in [-1, 1]
        ##rescale: 2L/lmax-I; lmax=1.0
        fill_value = -0.05  ##-0.5
        edge_index, lap = add_self_loops(edge_index, lap, fill_value, num_nodes)
        lap *= 2

        ########################################
        # Perform filter operation recurrently.
        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])
        if K > 1:
            Tx_1 = spmm(edge_index, lap, num_nodes, x.permute(1, 2, 0).contiguous().view((num_nodes, -1))).view((num_nodes, -1, batch)).permute(2, 0,1)  # spmm(edge_index, lap, num_nodes, x)
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, K):
            Tx_2 = 2 * spmm(edge_index, lap, num_nodes, x.permute(1, 2, 0).contiguous().view((num_nodes, -1))).view((num_nodes, -1, batch)).permute(2,0,1) - Tx_0
            # 2 * spmm(edge_index, lap, num_nodes, Tx_1) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.weight.size(0)-1)

    
class ChebNet(nn.Module):
    def __init__(self, nfeat, nfilters, nclass, K=2, nodes=360, nhid=128, gcn_layer=2, dropout=0, gcn_flag=False):
        super(ChebNet, self).__init__()
        self.gcn_layer = gcn_layer

        ####feature extracter
        self.graph_features = nn.ModuleList()
        if gcn_flag is True:
            print('Using GCN Layers instead')
            self.graph_features.append(GraphConv(nfeat, nfilters))
        else:
            self.graph_features.append(ChebConv(nfeat, nfilters, K))
        for i in range(gcn_layer):
            if gcn_flag is True:
                self.graph_features.append(GraphConv(nfilters, nfilters))
            else:
                self.graph_features.append(ChebConv(nfilters, nfilters, K))


        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity(dropout)

        # Define the output layer
        self.graph_nodes = nodes
        self.hidden_size = self.graph_nodes
        self.pool = nn.AdaptiveMaxPool2d((self.hidden_size,1))

        self.linear = nn.Linear(self.hidden_size, nclass)
        self.hidden2label = nn.Sequential(
            nn.Linear(self.hidden_size, nhid),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(nhid, nhid // 4),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(nhid // 4, nclass),
        )

    def forward(self, inputs, adj_mat):        
        edge_index = adj_mat._indices()
        edge_weight = adj_mat._values()
        batch = inputs.size(0)
        ###gcn layer
        x = inputs
        for layer in self.graph_features:
            x = F.relu(layer(x, edge_index, edge_weight))
            x = self.dropout(x)
        x = self.pool(x)
        ###linear dense layer
        # y_pred = self.linear(x.view(batch,-1))
        y_pred = self.hidden2label(x.view(batch, -1))
        return y_pred


####useful functions regarding to model training
########a few useful functions
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

##training the model
def train(model, adj_mat, device,train_loader,optimizer,loss_func, epoch):
    model.train()

    acc = 0.
    train_loss = 0.
    total = 0
    t0 = time.time()
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
#         print('inputs ',data.size(),'\nadj_mat ',adj_mat.size())
        out = model(data,adj_mat)
        loss = loss_func(out,target)
        pred = F.log_softmax(out, dim=1).argmax(dim=1)

        total += target.size(0)
        train_loss += loss.sum().item()
        acc += pred.eq(target.view_as(pred)).sum().item()
        
        loss.backward()
        optimizer.step()
        
    print("\nEpoch {}: \nTime Usage:{:4f} | Training Loss {:4f} | Acc {:4f}".format(epoch,time.time()-t0,train_loss/total,acc/total))
    return train_loss/total,acc/total

def test(model, adj_mat, device, test_loader,loss_func):
    model.eval()
    test_loss=0.
    test_acc = 0.
    total = 0
    ##no gradient desend for testing
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            ## saliency maps
            saliency = saliency_Maps(data,target)
            saliRank = saliency_Rank(saliency) ## donate of channel
            out = model(data,adj_mat)
            
            loss = loss_func(out,target)
            test_loss += loss.sum().item()
            pred = F.log_softmax(out, dim=1).argmax(dim=1)
            #pred = out.argmax(dim=1,keepdim=True) # get the index of the max log-probability
            total += target.size(0)
            test_acc += pred.eq(target.view_as(pred)).sum().item()            
            ## Network
            network = adj_network(saliency)
    
    test_loss /= total
    test_acc /= total
    print('Test Loss {:4f} | Acc {:4f}'.format(test_loss,test_acc))
    return test_loss,test_acc, network

def plot_Network(model_history):
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt
    G=nx.Graph()
    cm = plt.cm.get_cmap('RdYlBu_r')
    cm1 = plt.cm.get_cmap('Greys')
    ## row, col and value type np.array 
    rr,cc,vv = [],[],[]
    for jj in range(10):
        for ii in model_history['test_cow_']:
            rr.append(ii[jj]+1) 
        for ii in model_history['test_col_']:
            cc.append(ii[jj]+1) 
        for ii in model_history['test_value_']:
            vv.append(ii[jj]+1) 
    row = np.array(rr)
    col = np.array(cc)
    value = np.array(vv)
#     print(row,col,value)
    for j in range(len(row)):
        G.add_weighted_edges_from([(row[j],col[j],value[j])])
    print(G.edges())
    pos = [(-2.92,1.08),(-2.92,1.08),(-1.98,1.04),(-1.,1.02),(0,1),(1,1.02),(1.98,1.04),(2.92,1.08),(-3.,0),(-2,0),(-1,0),(0,0),(1,0),(2,0),(3,0),(-2.92,-1.08),(-1.98,-1.04),(-1.,-1.02),(0,-1),(1,-1.02),(1.98,-1.04),(2.92,-1.08),(-1.3,3.8),(0,4),(1.3,3.8),(-2.4,3.15),(-1.5,3),(0,3),(1.5,3),(2.4,3.15),(-3.3,2.37),(-2.6,2.25),(-1.7,2.15),(-0.85,2.07),(0,2),(0.85,2.07),(1.7,2.15),(2.6,2.25),(3.3,2.37),(-3.85,1.13),(3.85,1.13),(-4,0),(4,0),(-5,0),(5,0),(-3.85,-1.13),(3.85,-1.13),(-3.3,-2.37),(-2.6,-2.25),(-1.7,-2.15),(-0.85,-2.07),(0,-2),(0.85,-2.07),(1.7,-2.15),(2.6,-2.25),(3.3,-2.37),(-2.4,-3.15),(-1.5,-3),(0,-3),(1.5,-3),(2.4,-3.15),(-1.3,-3.8),(0,-4),(1.3,-3.8),(0,-5)]
    de = nx.draw_networkx_edges(G,pos,alpha=0.8,edge_color=[float(v['weight']*1) for (r,c,v) in G.edges(data=True)],edge_cmap=cm1,width=4)##edge color
    dn = nx.draw_networkx_nodes(G,pos,node_color=[1*jj[1] for jj in G.degree()],cmap=cm)
    nx.draw_networkx_labels(G,pos)
    plt.colorbar(de)
    plt.colorbar(dn)
    # plt.savefig('a.png')
    plt.show()

def adj_network(saliency):
    network = [[],[],[]]
    rankNum = 10
    for ii in range(rankNum):
        max_ = torch.max(saliency,1)
        row = max_.indices.numpy().tolist()[0]
        col = max_.indices.numpy().tolist()[1]
        value = max_.values.numpy().tolist()[0]+max_.values.numpy().tolist()[1]
        network[0].append(row)
        network[1].append(col)
        network[2].append(value) 
        max_ = torch.max(saliency,1)
        for jj in range(max_.values.size()[0]):
            saliency[jj][max_.indices.numpy().tolist()[jj]] = 0
    return network
    
def saliency_Rank(saliency):
    saliRank = {} # channel and donate
    rankNum = 10
    for ii in range(rankNum):
        max_ = torch.max(saliency,1)
        for jj in range(max_.values.size()[0]):
            saliRank[max_.indices.numpy().tolist()[jj]] = max_.values.numpy().tolist()[jj]
            saliency[jj][max_.indices.numpy().tolist()[jj]] = 0     
#     print(saliRank)
    return saliRank

def saliency_Maps(data,target): 
    import torch.nn as nn
    entroy = nn.CrossEntropyLoss()
    ## get the average (20,360) of (20,360,17)
    data1 = torch.randn(data.size()[0], 1)
    for ii in range(data.size()[1]):
        data1 = torch.cat([data1,torch.mean(data[:,ii,:],1,True)],dim=1)
    data1 = data1[:,1:]
    saliency = None
    ## get gradient of data
    data1.requires_grad_()
#     data1 = torch.autograd.Variable(data1,requires_grad=True)
    ## get true class
    with torch.enable_grad():
        loss = entroy(data1,target)
    loss.backward()
#     logits = model.forward(data)
#     logits = logits.gather(1,target.view(-1,1)).squeeze()
#     # torch.FloatTensor([1. for i in range(len(target))])
#     logits.backward(torch.FloatTensor([1. for i in range(len(target))])) # element 0 of tensors does not require grad and does not have a grad_fn    
    saliency = abs(data1.grad.data)
#     print(saliency.size(),saliency)
    
    return saliency

def TPR_FPR(target,pred):
    tpr,fpr = 0,0
    TP1,FN1,FP1,TN1=0,0,0,0
    cla1 = np.unique(pred)
    cla2 = np.unique(target)
    claSum = np.unique(np.hstack((cla1,cla2)))
    cc = claSum[0]
    for ii in range(len(target)):
            if target[ii]==cc and pred[ii]==cc:
                TP1+=1
            elif target[ii]!=cc and pred[ii]==cc:
                FP1+=1
            elif target[ii]==cc and pred[ii]!=cc:
                FN1+=1
            elif target[ii]!=cc and pred[ii]!=cc:
                TN1+=1
    tpr = TP1/(TP1+FN1)
    fpr = FP1/(TN1+FP1)
    print(tpr,fpr)
    return tpr,fpr

def plot_history(model_history):
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(model_history['train_acc'], color='r')
    plt.plot(model_history['test_acc'], color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Prediction Accuracy')
    plt.legend(['Training', 'Validation'])

    plt.subplot(122)
    plt.plot(model_history['train_loss'], color='r')
    plt.plot(model_history['test_loss'], color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Function')
    plt.legend(['Training', 'Validation'])
    plt.show()

def model_fit_evaluate(model,adj_mat,device,train_loader,test_loader,optimizer,loss_func,num_epochs=100):
    best_acc = 0 
    model_history={}
    model_history['train_loss']=[];
    model_history['train_acc']=[];
    model_history['test_loss']=[];
    model_history['test_acc']=[]; 
    model_history['test_cow_']=[];
    model_history['test_col_']=[];
    model_history['test_value_']=[]; 
    for epoch in range(num_epochs):
        train_loss,train_acc =train(model,adj_mat, device, train_loader, optimizer,loss_func, epoch)
        model_history['train_loss'].append(train_loss)
        model_history['train_acc'].append(train_acc)

        test_loss,test_acc,network = test(model,adj_mat, device, test_loader,loss_func)
        model_history['test_loss'].append(test_loss)
        model_history['test_acc'].append(test_acc)        
        model_network(network,model_history)
        if test_acc > best_acc:
            best_acc = test_acc
            print("Model updated: Best-Acc = {:4f}".format(best_acc))

    print("best testing accuarcy:",best_acc)
    plot_history(model_history)
    plot_Network(model_history)

def model_network(network,model_history):
    model_history['test_cow_'].append(network[0])
    model_history['test_col_'].append(network[1])
    model_history['test_value_'].append(network[2]) 