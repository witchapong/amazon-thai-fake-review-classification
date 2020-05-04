import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import *

def emb_sz_rule(n_cat): return min(600, round(1.6 * n_cat**0.56))

def targify(targs, input_len):
    '''
    Replace None target i.e. in test dataset with zero array
    '''
    out = []
    for targ in targs:
        if targ is None:
            out.append(np.zeros(input_len))
        else:
            out.append(targ)
    return out

class UniversalDataset(Dataset):
    def __init__(self, ins:list, in_dtypes:list, targs:list, targ_dtypes:list):
    
        self.ins = ins
        self.in_dtypes = in_dtypes
        self.len = len(ins[0])
        self.targs = targify(targs, input_len=self.len)
        self.targ_dtypes = targ_dtypes

    def __len__(self): return self.len
    def __getitem__(self, idx):
        return (tuple([torch.tensor(input[idx], dtype=dtype) for input,dtype in zip(self.ins, self.in_dtypes)]),
                tuple([torch.tensor(targ[idx], dtype=dtype) for targ,dtype in zip(self.targs, self.targ_dtypes)]))

class UniversalDataset_(Dataset):
    def __init__(self, ins:list, in_dtypes:list, targs:list, targ_dtypes:list):
    
        self.ins = ins
        self.in_dtypes = in_dtypes
        self.len = len(ins[0][0])
        self.targs = targify(targs, input_len=self.len)
        self.targ_dtypes = targ_dtypes

    def __len__(self): return self.len
    def __getitem__(self, idx):
        return (tuple([[torch.tensor(input_[idx], dtype=dtype_) for input_, dtype_ in zip(input,dtype)] for input,dtype in zip(self.ins, self.in_dtypes)]),
                tuple([torch.tensor(targ[idx], dtype=dtype) for targ,dtype in zip(self.targs, self.targ_dtypes)]))

def get_basenn(static_cat_ins:list,emb_p:float,static_num_ins:int,
               fc_sizes:list,fc_ps:list,
               out_tasks:list,out_ranges:list,out_features:list):
    
    config = {'static_cat_ins': static_cat_ins,
          'emb_p': emb_p,
          'static_num_ins': static_num_ins,
          'fc_sizes': fc_sizes,
          'fc_ps': fc_ps,
          'out_tasks': out_tasks,
          'out_ranges': out_ranges,
          'out_features': out_features}
    
    return BaseNN(config)

class RangeSigmoid(nn.Module):
    def __init__(self,low,high):
        super().__init__()
        self.low = low
        self.high = high
    def forward(self,input):
        return torch.sigmoid(input)*(self.high-self.low)+self.low

def get_fc_layers(fc_sizes, ps):
    fc_layers_list = []
    for ni,nf,p in zip(fc_sizes[:-1], fc_sizes[1:], ps):
        fc_layers_list.append(nn.Linear(ni, nf))
        fc_layers_list.append(nn.ReLU(inplace=True))
        fc_layers_list.append(nn.BatchNorm1d(nf))
        fc_layers_list.append(nn.Dropout(p=p))
    return nn.Sequential(*fc_layers_list)
    
class BaseNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # embedding layers
        self.static_embs = nn.ModuleList([nn.Embedding(input_size,emb_sz_rule(input_size)) for input_size in config['static_cat_ins']])
        self.emb_drop = nn.Dropout(config['emb_p'])
        self.bn = nn.BatchNorm1d(config['static_num_ins'])
        
        # fc layers
        self.lin_in = sum([emb_sz_rule(i) for i in config['static_cat_ins']]) + config['static_num_ins']
        self.fc_sizes = [self.lin_in] + config['fc_sizes']
        self.fc_layers = get_fc_layers(self.fc_sizes, config['fc_p'])

        # output head
        out_heads = []
        for out_task, out_range, out_feature in zip(config['out_tasks'],config['out_ranges'],config['out_features']):
            if (out_task == 'regression') & (out_range is not None):
                out_heads.append(nn.Sequential(nn.Linear(in_features=self.fc_sizes[-1],out_features=out_feature),
                                                        RangeSigmoid(out_range[0],out_range[1])))
            else:
                out_heads.append(nn.Linear(in_features=self.fc_sizes[-1], out_features=out_feature))
        self.out_heads = nn.ModuleList(out_heads)
    
    def forward(self,ins):
        
        # extract input
        static_num, static_cat = ins
        static_num,static_cat = static_num.to(self.device), static_cat.to(self.device)
        
        # get static cat embedding feature
        static_cat_emb = [e(static_cat[:,i]) for i,e in enumerate(self.static_embs)]
        static_cat_emb = torch.cat(static_cat_emb,1)
        static_cat_emb = self.emb_drop(static_cat_emb)
        
        static_num = self.bn(static_num)
        
        # concat numeric & embedding feature
        lin_in = torch.cat([static_num,static_cat_emb],1)

        # forward thru fc
        fc_out = self.fc_layers(lin_in)
        
        outs = []
        for head in self.out_heads: outs.append(head(fc_out))
        
        return outs
        
class MultitasksLoss():
    
    def __init__(self, loss_funcs: Iterable, loss_weights: list=None):
        self.loss_funcs = loss_funcs
        if loss_weights is None: loss_weights = [1.]*len(self.loss_funcs)
        self.loss_weights = loss_weights

    def __call__(self,preds: Iterable,trues: Iterable):
        losses = [loss_func(pred,true) for loss_func, pred, true in zip(self.loss_funcs,preds,trues)]
        loss = losses[0]*self.loss_weights[0]
        for i in range(1,len(self.loss_funcs)): loss += losses[i]*self.loss_weights[i]
        return loss

def extract_pred_outs(dataloader: DataLoader, model: nn.Module):
    '''
    Extracting prediction from all tasks of a model. Return as list of numpy array of each prediction.
    '''
    model.eval()
    outs=[[] for _ in range(len(dataloader.dataset.targs))]
    for ins,_ in dataloader:
        out = model(ins)
        for i in range(len(dataloader.dataset.targs)): outs[i].append(out[i].cpu().detach().numpy())
            
    # concat output
    outs = [np.concatenate(o,0) for o in outs]
    return outs

def numpy_softmax(arr: np.ndarray): return np.exp(arr)/np.exp(arr).sum(1).reshape(-1,1)

def get_cnn(static_cat_ins:int, static_num_ins:int,
           seq_feat:list, seq_len:list, num_filter:list, filter_size:list,
           out_tasks:list, out_ranges:list, out_features:list,
           emb_p:float=.05,fc_sizes:list=[400,200], fc_ps:list=[.5,.25]):
    
    config = {'static_cat_ins':static_cat_ins,'emb_p':emb_p,'static_num_ins':static_num_ins,
          'seq_feat':seq_feat, 'seq_len':seq_len , 'num_filter':num_filter, 'filter_size':filter_size,
          'fc_sizes':fc_sizes,'fc_ps':fc_ps,
          'out_tasks':out_tasks,
          'out_ranges':out_ranges,
          'out_features':out_features}
    
    return CNN(config)

class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # embedding layers
        self.static_embs = nn.ModuleList([nn.Embedding(input_size,emb_sz_rule(input_size)) for input_size in config['static_cat_ins']])
        self.emb_drop = nn.Dropout(config['emb_p'])
        
        # convs layers
        self.convs = nn.ModuleList([nn.ModuleList([nn.Conv1d(1, config['num_filter'][i], (size,config['seq_feat'][i])) for size in config['filter_size'][i]]) for i in range(len(config['seq_feat']))])

        # fc layers
        self.lin_in = sum([emb_sz_rule(i) for i in config['static_cat_ins']]) + config['static_num_ins']\
        + sum([num_filter*len(filter_size) for num_filter, filter_size in zip(config['num_filter'],config['filter_size'])])
        
        self.fc_sizes = [self.lin_in] + config['fc_sizes']
        self.fc_layers = get_fc_layers(self.fc_sizes, config['fc_p'])
        
        # output head
        out_heads = []
        for out_task, out_range, out_feature in zip(config['out_tasks'],config['out_ranges'],config['out_features']):
            if (out_task == 'regression') & (out_range is not None):
                out_heads.append(nn.Sequential(nn.Linear(in_features=self.fc_sizes[-1],out_features=out_feature),
                                                        RangeSigmoid(out_range[0],out_range[1])))
            else:
                out_heads.append(nn.Linear(in_features=self.fc_sizes[-1], out_features=out_feature))
        self.out_heads = nn.ModuleList(out_heads)
    
    def forward(self,ins):
        
        # extract input
        static_num, static_cat, seq = ins
        static_num, static_cat = static_num[0], static_cat[0]
        
        # get demo embedding feature
        static_cat_emb = [e(static_cat[:,i]) for i,e in enumerate(self.static_embs)]
        static_cat_emb = torch.cat(static_cat_emb,1)
        static_cat_emb = self.emb_drop(static_cat_emb)
        
        # sequential input
        seq_outs = []
        for i,s_in in enumerate(seq):
            seq_out = s_in.view(-1,self.config['seq_len'][i],self.config['seq_feat'][i]).unsqueeze(1)
            seq_out = [torch.relu(conv(seq_out)).squeeze(-1) for conv in self.convs[i]]
            seq_out = [torch.max_pool1d(i,i.size(-1)).squeeze(-1) for i in seq_out]
            seq_out = torch.cat(seq_out,1)
            seq_outs.append(seq_out)
        
        # concat all features
        lin_in = torch.cat([static_num,static_cat_emb,*seq_outs],1)
        
        # forward thru fc
        fc_out = self.fc_layers(lin_in)
        
        outs = []
        for head in self.out_heads: outs.append(head(fc_out))
        
        return outs

def get_rnn(static_cat_ins:int, static_num_ins:int,
           seq_feat:list, seq_len:list , num_layer:list, rnn_h:list,rnn_p:list,
           out_tasks:list, out_ranges:list, out_features:list,
           emb_p:float=.05,fc_sizes:list=[400,200], fc_ps:list=[.5,.25]):
    
    config = {'static_cat_ins': static_cat_ins, 'emb_p': emb_p,'static_num_ins': static_num_ins,
          'seq_feat':seq_feat, 'seq_len':seq_len, 'num_layer':num_layer, 'rnn_h':rnn_h, 'rnn_p':rnn_p,
          'fc_sizes':fc_sizes,'fc_ps':fc_ps,
          'out_tasks':out_tasks,
          'out_ranges':out_ranges,
          'out_features':out_features}
    
    return RNNAttention(config)

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim 
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class RNNAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # embedding layers
        self.static_embs = nn.ModuleList([nn.Embedding(input_size,emb_sz_rule(input_size)) for input_size in config['static_cat_ins']])
        self.emb_drop = nn.Dropout(config['emb_p'])
        
        # rnn layers
        self.rnns = nn.ModuleList([nn.GRU(input_size=seq_feat, hidden_size=rnn_h, num_layers=num_layer,
                              bias=True, batch_first=True, dropout=rnn_p, bidirectional=True)\
               for seq_feat, rnn_h, num_layer, rnn_p in zip(config['seq_feat'],config['rnn_h'],config['num_layer'],config['rnn_p'])])
        
        # attention layer
        self.attns = nn.ModuleList([Attention(2*rnn_h,seq_len) for rnn_h, seq_len in zip(config['rnn_h'],config['seq_len'])])
        
        # fc layers
        self.lin_in = sum([emb_sz_rule(i) for i in config['static_cat_ins']]) + config['static_num_ins']\
        + sum([2*rnn_h for rnn_h in config['rnn_h']])
        
        self.fc_sizes = [self.lin_in] + config['fc_sizes']
        self.fc_layers = get_fc_layers(self.fc_sizes, config['fc_p'])
        
        # output head
        out_heads = []
        for out_task, out_range, out_feature in zip(config['out_tasks'],config['out_ranges'],config['out_features']):
            if (out_task == 'regression') & (out_range is not None):
                out_heads.append(nn.Sequential(nn.Linear(in_features=self.fc_sizes[-1],out_features=out_feature),
                                                        RangeSigmoid(out_range[0],out_range[1])))
            else:
                out_heads.append(nn.Linear(in_features=self.fc_sizes[-1], out_features=out_feature))
        self.out_heads = nn.ModuleList(out_heads)
    
    def forward(self,ins):
        
        # extract input
        static_num, static_cat, seq = ins
        static_num, static_cat = static_num[0], static_cat[0]
        
        # get demo embedding feature
        static_cat_emb = [e(static_cat[:,i]) for i,e in enumerate(self.static_embs)]
        static_cat_emb = torch.cat(static_cat_emb,1)
        static_cat_emb = self.emb_drop(static_cat_emb)
        
        # sequential input
        seq_outs = []
        for i, s_in in enumerate(seq):
            seq_out, _ = self.rnns[i](s_in.view(-1,config['seq_len'][i],config['seq_feat'][i]))
            seq_out = self.attns[i](seq_out)
            seq_outs.append(seq_out)
            
        
        # concat all features
        lin_in = torch.cat([static_num,static_cat_emb,*seq_outs],1)
        
        # forward thru fc
        fc_out = self.fc_layers(lin_in)
        
        outs = []
        for head in self.out_heads: outs.append(head(fc_out))
        
        return outs

class Conv1dLayer(nn.Module):
    def __init__(self, seq_feat, seq_len, num_filter, filter_size, stride):
        super().__init__()
        self.num_filter = num_filter
        self.seq_feat = seq_feat
        self.filter_size = filter_size
        self.seq_len = seq_len
        self.stride = stride
        self.conv1d = nn.Conv1d(1, num_filter, (filter_size,seq_feat), stride=stride)
        self.bn = nn.BatchNorm1d(num_filter)
        self.relu = nn.ReLU()
    def forward(self,input):
        out = self.conv1d(input.view(-1,self.seq_len,self.seq_feat).unsqueeze(1))
        out = self.relu(self.bn(out.squeeze(-1))).transpose(1,-1)
        return out

def get_1d_out_len(in_len, filter_size, stride): return (in_len - filter_size)//stride+1

def get_deep_cnn(static_cat_ins:int, static_num_ins:int,
           seq_feat:list, seq_len:list, num_filter:list, filter_size:list, stride:list,
           out_tasks:list, out_ranges:list, out_features:list,
           emb_p:float=.05,fc_sizes:list=[400,200], fc_ps:list=[.5,.25]):
    
    config = {'static_cat_ins':static_cat_ins,'emb_p':emb_p,'static_num_ins':static_num_ins,
                'seq_feat':seq_feat, 'seq_len':seq_len , 'num_filter':num_filter, 'filter_size':filter_size, 'stride':stride,
                'fc_sizes':fc_sizes,'fc_ps':fc_ps,
                'out_tasks':out_tasks,
                'out_ranges':out_ranges,
                'out_features':out_features}

    return DeepCNN(config)

class Conv1dCH(nn.Module):
    def __init__(self, seq_feat, seq_len, num_filters, filter_sizes, strides):
        super().__init__()
        seq_feats = [seq_feat] + num_filters[:-1]
        conv_outs = []
        for i in range(len(filter_sizes)):
            if i==0:
                conv_outs.append(get_1d_out_len(seq_len, filter_sizes[i], strides[i]))
            else:
                conv_outs.append(get_1d_out_len(conv_outs[-1], filter_sizes[i], strides[i]))
                
        seq_lens = [seq_len] + conv_outs[:-1]
        self.conv1d_layers = nn.Sequential(*[Conv1dLayer(w,h,nf,fs,s) for w,h,nf,fs,s in zip(seq_feats,seq_lens,num_filters,filter_sizes,strides)])
        self.w_pooling = nn.Conv1d(1,1,(1,conv_outs[-1]))
        
    def forward(self,input):
        return self.w_pooling(self.conv1d_layers(input).transpose(1,-1).unsqueeze(1)).squeeze()

class DeepCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # embedding layers
        self.static_embs = nn.ModuleList([nn.Embedding(input_size,emb_sz_rule(input_size)) for input_size in config['static_cat_ins']])
        self.emb_drop = nn.Dropout(config['emb_p'])
        
        # mutil-channel deep conv layers
        conv_seqs = []
        for i in range(len(config['seq_feat'])):
            conv_chs = []
            for j in range(len(config['num_filter'][i])):
                conv_chs.append(Conv1dCH(config['seq_feat'][i], config['seq_len'][i], config['num_filter'][i][j], config['filter_size'][i][j], config['stride'][i][j]))
            conv_seqs.append(nn.ModuleList(conv_chs))
        self.conv_seqs = nn.ModuleList(conv_seqs)

        # fc layers
        self.lin_in = sum([emb_sz_rule(i) for i in config['static_cat_ins']]) + config['static_num_ins']\
        + sum([config['num_filter'][i][j][-1] for j in range(len(config['num_filter'][i])) for i in range(len(config['num_filter']))])
        
        self.fc_sizes = [self.lin_in] + config['fc_sizes']
        
        self.fc_layers = get_fc_layers(self.fc_sizes, config['fc_p'])
        
        # output head
        out_heads = []
        for out_task, out_range, out_feature in zip(config['out_tasks'],config['out_ranges'],config['out_features']):
            if (out_task == 'regression') & (out_range is not None):
                out_heads.append(nn.Sequential(nn.Linear(in_features=self.fc_sizes[-1],out_features=out_feature),
                                                        RangeSigmoid(out_range[0],out_range[1])))
            else:
                out_heads.append(nn.Linear(in_features=self.fc_sizes[-1], out_features=out_feature))
        self.out_heads = nn.ModuleList(out_heads)
    
    def forward(self,ins):
        
        # extract input
        static_num, static_cat, seq = ins
        static_num, static_cat = static_num[0], static_cat[0]
        
        # get demo embedding feature
        static_cat_emb = [e(static_cat[:,i]) for i,e in enumerate(self.static_embs)]
        static_cat_emb = torch.cat(static_cat_emb,1)
        static_cat_emb = self.emb_drop(static_cat_emb)
        
        # sequential input
        seq_outs = []
        for i,s_in in enumerate(seq):
            seq_out = [conv(s_in) for conv in self.conv_seqs[i]]
            seq_out = torch.cat(seq_out,1)
            seq_outs.append(seq_out)

        # concat all features
        lin_in = torch.cat([static_num,static_cat_emb,*seq_outs],1)
        
        # forward thru fc
        fc_out = self.fc_layers(lin_in)
        
        outs = []
        for head in self.out_heads: outs.append(head(fc_out))
        
        return outs