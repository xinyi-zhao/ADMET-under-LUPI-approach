import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dgl import model_zoo

from utilis import Meter, EarlyStopping, collate_molgraphs, set_random_seed, \
    load_dataset_for_classification, load_model

l=0.1
t=10
dict={}
class model2(nn.Module):
    def __init__(self,n_task,n_features):
        super(model2,self).__init__()
        self.n_features=n_features
        self.n_task=n_task
        self.layer=nn.Sequential(
            nn.Linear(n_features,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,12),
            nn.Softmax(dim=0)
        )
    def forward(self,features):
        return self.layer(features)
uu=[]
def get_gene(smiles,args):
    gene=[]
    not_missing=[]
    for smile in smiles:
        if(smile in dict):
            gene.append(np.array(dict[smile]))
            not_missing.append(np.ones((12)))
        else:
            gene.append(np.zeros((961)))
            not_missing.append(np.zeros((12)))
    gene=np.array(gene,dtype='float32')
    not_missing=np.array(not_missing,dtype='float32')
    #print(gene)
    #print(not_missing)
    Gene=torch.from_numpy(gene).cuda()
    Not_missing=torch.from_numpy(not_missing).cuda()
    return Gene,Not_missing
def run_a_train_epoch(args, epoch, model,model_2,data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    tt=0
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        atom_feats = bg.ndata.pop(args['atom_data_field'])
        atom_feats, labels, masks = atom_feats.to(args['device']), \
                                    labels.to(args['device']), \
                                    masks.to(args['device'])
        logits = model(bg, atom_feats)
        # Mask non-existing labels

        star_features,not_missing=get_gene(smiles,args)
        y_star=model_2(star_features)
        lupi_loss=(loss_criterion(logits,y_star)*(not_missing!=0).float()).mean()


        loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
        tmp=torch.exp(-t*((loss_criterion(y_star,labels)*((masks!=0)*(not_missing!=0)).float()).mean()))*((loss_criterion(logits,y_star)*(not_missing!=0).float()).mean()-
                                                                                                    (loss_criterion(logits, labels) * (masks != 0).float()).mean())
       # loss=(1-l)*loss+t*l*t*lupi_loss
        loss=loss+tmp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
        tt=tt+loss.item()
        train_meter.update(logits, labels, masks)
    train_score = np.mean(train_meter.compute_metric(args['metric_name']))
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric_name'], train_score))
    print(tt)
    uu.append(tt)

def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            atom_feats = bg.ndata.pop(args['atom_data_field'])
            atom_feats, labels = atom_feats.to(args['device']), labels.to(args['device'])
            logits = model(bg, atom_feats)
            eval_meter.update(logits, labels, masks)
    return np.mean(eval_meter.compute_metric(args['metric_name']))

def main(args):
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    set_random_seed()

    # Interchangeable with other datasets
    dataset, train_set, val_set, test_set = load_dataset_for_classification(args)
    print(dataset)
    train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                              collate_fn=collate_molgraphs)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs)
    test_loader = DataLoader(test_set, batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)
    model_2=model2(n_task=12,n_features=961)
    if args['pre_trained']:
        args['num_epochs'] = 0
        model = model_zoo.chem.load_pretrained(args['exp'])
    else:
        args['n_tasks'] = dataset.n_tasks
        model = load_model(args)
        loss_criterion = BCEWithLogitsLoss(pos_weight=dataset.task_pos_weights.to(args['device']),
                                           reduction='none')
        optimizer = Adam([{'params':model.parameters()},{'params':model_2.parameters()}], lr=args['lr'])
        stopper = EarlyStopping(patience=args['patience'])
    model.to(args['device'])
    model_2=model_2.cuda()
    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, model_2,train_loader, loss_criterion, optimizer)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, val_loader)
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['metric_name'],
            val_score, args['metric_name'], stopper.best_score))
        if early_stop:
            break

    if not args['pre_trained']:
        stopper.load_checkpoint(model)
    test_score = run_an_eval_epoch(args, model, test_loader)
    print('test {} {:.4f}'.format(args['metric_name'], test_score))

if __name__ == '__main__':
    import argparse

    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description='Molecule Classification')
    parser.add_argument('-m', '--model', type=str, choices=['GCN', 'GAT'],
                        help='Model to use')
    parser.add_argument('-d', '--dataset', type=str, choices=['Tox21'],
                        help='Dataset to use')
    parser.add_argument('-p', '--pre-trained', action='store_true',
                        help='Whether to skip training and use a pre-trained model')
    args = parser.parse_args().__dict__
    args['exp'] = 'GCN_Tox21'
    args['dataset']='Tox21'
    args['model']='GCN'
    args['node_in_feats']=74
    args.update(get_exp_configure(args['exp']))

    csv_data = pd.read_excel('gene_info.xlsx')
    for index,row in csv_data.iterrows():
        tmp = list(row.values)
        tmp=np.array(tmp)
        dict[tmp[1]]=tmp[2:]

    main(args)
    print(uu)
    plt.plot(range(len(uu)), uu, label="loss")
    plt.ylim(0, 100)
    plt.legend(loc="lower right")
    plt.show()