# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:22:11 2018

@author: Muhammad Dawood
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.preprocessing import StandardScaler

def xentropy_cost(x_target, x_pred):
 assert x_target.size() == x_pred.size(), "size fail ! "+str(x_target.size()) + " " + str(x_pred.size())
 logged_x_pred = torch.log(x_pred)
 cost_value = -torch.sum(x_target * logged_x_pred)
 return cost_value


# experiment 1: noiseless labels as privileged info
#def synthetic_01(a,n):
#    x  = np.random.randn(n,a.size)
 #   e  = (np.random.randn(n))[:,np.newaxis]
  #  xs = np.dot(x,a)[:,np.newaxis]
   # y  = ((xs+e) > 0).ravel()
 #   return (xs,x,y)


    
def softmax(w, t = 1.0):
    e = np.exp(w / t)
    return e/np.sum(e,1)[:,np.newaxis]

class Net(nn.Module):
    def __init__(self,d,q):
        super(Net, self).__init__()
#        q = 2
#        self.hidden1 = nn.Linear(d,1)
        self.out = nn.Linear(d,q)

    def forward(self,x):
#        x = self.hidden1(x)
        x = torch.sigmoid(self.out(x))
        return x,x    


def fitModel(model,optimizer,criterion,epochs,x,target):
    for epoch in range(epochs):
            # Forward pass: Compute predicted y by passing x to the model
        y_pred,_ = model(x)
        # Compute and print loss
        loss = criterion(y_pred, target)
        #print(epoch, loss.data[0])
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model
def do_exp(x_tr,xs_tr,y_tr,x_te,xs_te,y_te,chang):
    t = 0.1
    l=0.01
    l_r=0.001
    epochs=4000
    criterion = torch.nn.BCELoss()
    # scale stuff
    s_x   = StandardScaler().fit(x_tr)
    s_s   = StandardScaler().fit(xs_tr)
    x_tr  = s_x.transform(x_tr)
    x_te  = s_x.transform(x_te)
    xs_tr = s_s.transform(xs_tr)
    #xs_te = s_s.transform(xs_te)
    """
    Training of privilage space model
    """
    xs_tr = Variable(torch.from_numpy(xs_tr)).type(torch.FloatTensor)
    y_tr = Variable(torch.from_numpy(y_tr)).type(torch.FloatTensor)
    mlp_priv = Net(xs_tr.shape[1],1)
    optimizer = optim.RMSprop(mlp_priv.parameters())
    mlp_priv=fitModel(mlp_priv,optimizer,criterion,epochs,xs_tr[0:chang],y_tr[0:chang])
    xs_te = Variable(torch.from_numpy(xs_te)).type(torch.FloatTensor)
    _,p_tr=mlp_priv(xs_tr[0:chang])
    p_tr = p_tr.detach()
    """
    Training of regular MLP
    """
    x_tr = Variable(torch.from_numpy(x_tr)).type(torch.FloatTensor)
    mlp_reg = Net(x_tr.shape[1],1)
    optimizer = optim.RMSprop(mlp_reg.parameters())
    mlp_reg=fitModel(mlp_reg,optimizer,criterion,epochs,x_tr,y_tr)
    x_te = Variable(torch.from_numpy(x_te)).type(torch.FloatTensor)
    output,_=mlp_reg(x_te)
    pred = output > 0.5
    pred=pred.numpy().flatten()
    res_reg=np.mean(pred==y_te)
    out=output.detach().numpy().flatten()
    auc_reg=roc_auc_score(y_te,out)

#    softened=soften.detach()
#    p_tr=softened.numpy()
#    import pdb; pdb.set_trace()
#    #p_tr=softmax(softened,t)
#    p_tr=Variable(torch.from_numpy(p_tr)).type(torch.FloatTensor)
    
    ### freezing layers
    for param in mlp_priv.parameters():
        param.requires_grad =False
    
    """
    LUPI Combination of two model
    """
    mlp_dist = Net(x_tr.shape[1],1)
    optimizer = optim.RMSprop(mlp_dist.parameters())
    criterion = torch.nn.BCELoss()
    # Training loop
    for epoch in range(epochs):
            # Forward pass: Compute predicted y by passing x to the model
        y_pred,_ = mlp_dist(x_tr)
        # Compute and print loss
     #   loss1 = (1-l)*criterion(y_pred, y_tr)
     #   loss2=t*t*l*criterion(y_pred[0:849], p_tr[0:849])
      #  loss=loss1+loss2
        loss =l*(torch.exp(-t*criterion(p_tr[0:chang] , y_tr[0:chang]))*criterion(y_pred[0:chang] , p_tr[0:chang])) + (1-l)*criterion(y_pred,y_tr)
       # loss = criterion(y_pred,y_tr) + torch.exp(-t*criterion(p_tr[0:chang],y_tr[0:chang]))*(criterion(y_pred[0:chang],p_tr[0:chang]) - criterion(y_pred,y_tr))
        #print(epoch, loss.data[0])
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    output,_=mlp_dist(x_te)
    pred = output > 0.5
    pred=pred.numpy().flatten()
    res_dis=np.mean(pred==y_te)
    out=output.detach().numpy().flatten()
    auc_dis=roc_auc_score(y_te,out)
    print("reg_acc",res_reg,"dis_acc:",res_dis,"reg_auc:",auc_reg,"dis_auc:",auc_dis)
   # return np.array([res_priv,res_reg,res_dis,auc_priv,auc_reg,auc_dis])


# experiment hyper-parameters
d      = 50
n_tr   = 200
n_te   = 800
n_epochs = 7
eid    = 0

def did(aa,bb,cc):
    return (aa,bb,cc)
np.random.seed(0)
csv_data = pd.read_csv('data.csv')
tox_data=pd.read_csv('tox21.csv')
# do all four experiments
print("\nDistillation Pytorch Regression  Using BCE Loss method")
n=888
for i in range(0,12):
    tmp1=pd.concat((csv_data.iloc[:,i:i+1],csv_data.iloc[:,12:]),axis=1)
    datanow=tmp1.dropna(axis=0)
    data=np.array(datanow)
    tmp2 = pd.concat((tox_data.iloc[:, i:i + 1], tox_data.iloc[:, 12:]), axis=1)
    predict = tmp2.dropna(axis=0)
    test_data = np.array(predict)


    X_gene=data[:,1:962]
    chang=np.shape(X_gene)[0]
    print(chang)
   # print(X_gene.shape)
    X_structure=data[:,962:]
    Y=data[:,0]
    XX_gene=np.zeros((4000,961))
    XX_structure=test_data[0:4000,1:]
    YY=test_data[0:4000,0]
    X_gene=np.vstack((X_gene,XX_gene))
    X_structure=np.vstack((X_structure,XX_structure))
    Y=np.hstack((Y,YY))
    xs_tr,x_tr,y_tr,xs_te,x_te,y_te=(None,None,None,None,None,None)
    eid += 1
    R = np.zeros((n_epochs,6))
    xs_tr=X_gene
    x_tr=X_structure
    y_tr=Y

    x_te=test_data[4001:,1:]
    y_te=test_data[4001:,0]
    xs_te=x_te
    y_tr=y_tr.reshape(-1,1)
    y_te=y_te.reshape(-1,1)
    do_exp(x_tr,xs_tr,y_tr*1.0,x_te,xs_te,y_te*1.0,chang)

    print(i)