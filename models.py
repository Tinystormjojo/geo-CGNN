import sys
import torch
import time
import json
import os
import copy
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
import torch.nn as nn
from torch.nn import ( Linear, Bilinear, Sigmoid, Softplus, ELU, ReLU, SELU,
                       CELU, BatchNorm1d, ModuleList, Sequential,Tanh )
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
import re

def get_activation(name):
    act_name = name.lower()
    m = re.match(r"(\w+)\((\d+\.\d+)\)", act_name)
    if m is not None:
        act_name, alpha = m.groups()
        alpha = float(alpha)
        print(act_name, alpha)
    else:
        alpha = 1.0
    if act_name == 'softplus':
        return Softplus()
    elif act_name == 'ssp':
        return SSP()
    elif act_name == 'elu':
        return ELU(alpha)
    elif act_name == 'relu':
        return ReLU()
    elif act_name == 'selu':
        return SELU()
    elif act_name == 'celu':
        return CELU(alpha)
    elif act_name == 'sigmoid':
        return Sigmoid()
    elif act_name == 'tanh':
        return Tanh()
    else:
        raise NameError("Not supported activation: {}".format(name))


class Metric(object):
    def __init__(self, metric_fn, name):
        self.metric_fn = metric_fn
        self.name = name
        self.total_metric = 0.0
        self.total_count = 0

    def __call__(self, predictoins, targets):
        return self.metric_fn(predictoins, targets)

    def add_batch_metric(self, predictoins, targets):
        metric_tensor = self.metric_fn(predictoins, targets)
        self.total_metric += metric_tensor.item() * targets.size(0)
        self.total_count += targets.size(0)
        return metric_tensor

    def get_total_metric(self):
        score = self.total_metric / self.total_count
        self.total_metric = 0.0
        self.total_count = 0
        return score

class History(object):
    def __init__(self,file_path="data/history.csv"):
        self.history_path = file_path
        self.file = open(file_path, 'w')
        self.first = True

    def write(self, epoch, metrics):
        if self.first:
            self.first = False
            header = ','.join([name for name, _ in metrics])
            header = 'epoch,' + header + '\n'
            self.file.write(header)
        row = ','.join(['{}'.format(metric) for _, metric in metrics])
        row = '{},'.format(epoch) + row + '\n'
        self.file.write(row)
        self.file.flush()

    def close(self):
        self.file.close()

class Checkpoint(object):
    def __init__(self, model):
        self.model = model
        self.best_metric = None
        self.best_weights = model.weights

    def check(self, metric):
        if self.best_metric is None or metric < self.best_metric:
            self.best_metric = metric
            self.best_weights = self.model.weights

    def restore(self):
        self.model.weights = self.best_weights

class Model(object):
    def __init__(self, device, model, name, optimizer, scheduler, clip_value=None,
                 metrics=[('loss', nn.MSELoss()), ('mae', nn.L1Loss())]):
        self.name=name
        self.model=model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = [Metric(metric, name) for name, metric in metrics]
        self.clip_value = clip_value
        self.device=device

    def _set_mode(self, phase):
        if phase == 'train':
            self.model.train()  # Set model to training mode
        else:
            self.model.eval()   # Set model to evaluate mode

    def _process_batch(self, input, targets, phase):
        
        nodes=input.nodes.to(self.device)
        edge_sources=input.edge_sources.to(self.device)
        edge_targets=input.edge_targets.to(self.device)
        edge_distance=input.edge_distance.to(self.device)
        graph_indices=input.graph_indices.to(self.device)
        node_counts=input.node_counts.to(self.device)
        combine_sets=input.combine_sets.to(self.device)
        plane_wave=input.plane_wave.to(self.device)
        targets = targets.to(self.device)        
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = self.model(nodes,edge_sources,edge_targets,edge_distance,graph_indices,node_counts,combine_sets,plane_wave) 
            metric_tensors = [metric.add_batch_metric(outputs, targets) for metric in self.metrics]
            if phase == 'train':
                loss = metric_tensors[0]
                loss.backward()
                if self.clip_value is not None:
                    clip_grad_value_(self.model.parameters(), self.clip_value)
                self.optimizer.step()

        return metric_tensors, outputs

    def train(self, train_dl, val_dl, num_epochs):
        since = time.time()

        dataloaders = {'train': train_dl, 'val': val_dl}
        history = History(file_path='data/history_{}.csv'.format(self.name))
        checkpoint = Checkpoint(self)
        for epoch in range(num_epochs):
            epoch_since = time.time()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1), flush=True)

            #self.scheduler.step()
            print('current lr:', self.optimizer.param_groups[0]['lr'])

            train_val_metrics = []
            for phase in ['train', 'val']:
                self._set_mode(phase)

                for input, targets in dataloaders[phase]:
                    _, outputs = self._process_batch(input, targets, phase)


                epoch_metrics = [(metric.name, metric.get_total_metric()) for metric in self.metrics]
                text = ' '.join(['{}: {:.4f}'.format(name, metric) for name, metric in epoch_metrics])
                print('{} {}'.format(phase, text))

                if phase == 'val':
                    metric = epoch_metrics[1][1]
                    checkpoint.check(metric)

                train_val_metrics += [('_'.join([phase, name]), metric) for name, metric in epoch_metrics]
            self.scheduler.step()

            history.write(epoch, train_val_metrics)
            time_elapsed = time.time() - epoch_since
            print('Elapsed time (sec.): {:.3f}'.format(time_elapsed))
            print()
        history.close()

        if num_epochs > 0:
            time_elapsed = time.time() - since
            print('Total elapsed time (sec.): {:.3f}'.format(time_elapsed))
            print('The best val metric: {:4f}'.format(checkpoint.best_metric))
            print()

            # load the best model weights
            checkpoint.restore()


    def evaluate(self, dataloader):
        self.model.eval()   # Set model to evaluate mode

        # Iterate over data.
        all_outputs = []
        all_targets = []
        all_graph_vec = []
        for input, targets in dataloader:

            nodes=input.nodes.to(self.device)
            edge_sources=input.edge_sources.to(self.device)
            edge_targets=input.edge_targets.to(self.device)
            edge_distance=input.edge_distance.to(self.device)
            graph_indices=input.graph_indices.to(self.device)
            node_counts=input.node_counts.to(self.device)
            combine_sets=input.combine_sets.to(self.device)
            plane_wave=input.plane_wave.to(self.device)
            targets = targets.to(self.device)        
            
            with torch.set_grad_enabled(False):
                outputs,graph_vec = self.model(nodes,edge_sources,edge_targets,edge_distance,graph_indices,node_counts,combine_sets,plane_wave,output_graph=True)

            outputs = outputs.to(torch.device("cpu")).numpy()
            targets = targets.to(torch.device("cpu")).numpy()
            graph_vec = graph_vec.to(torch.device("cpu")).numpy()
            all_outputs.append(outputs)
            all_targets.append(targets)
            all_graph_vec.append(graph_vec)

        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)
        all_graph_vec = np.concatenate(all_graph_vec)

        all_outputs = torch.FloatTensor(all_outputs).to(self.device)
        all_targets = torch.FloatTensor(all_targets).to(self.device)

        total_metrics = [(metric.name, metric(all_outputs, all_targets).item()) for metric in self.metrics]
        text = ' '.join(['{}: {:.4f}'.format(name, metric) for name, metric in total_metrics])
        print('test {}'.format(text))

        all_outputs = all_outputs.to(torch.device("cpu")).numpy()
        all_targets = all_targets.to(torch.device("cpu")).numpy()

        return all_outputs, all_targets, all_graph_vec

    def save(self, model_path="model"):
        model_path="model/model_{}.pth".format(self.name)
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    @property
    def weights(self):
        return copy.deepcopy(self.model.state_dict())

    @weights.setter
    def weights(self, state):
        self.model.load_state_dict(state)


def _bn_act(num_features, activation, use_batch_norm=False):
    # batch normal + activation
    if use_batch_norm:
        if activation is None:
            return BatchNorm1d(num_features)
        else:
            return Sequential(BatchNorm1d(num_features), activation)
    else:
        return activation

class NodeEmbedding(Module):
    """
    Node Embedding layer
    """
    def __init__(self, in_features, out_features, activation=Sigmoid(),
                 use_batch_norm=False, bias=False):
        super(NodeEmbedding, self).__init__()
        self.linear = Linear(in_features, out_features, bias=bias)
        self.activation = _bn_act(out_features, activation, use_batch_norm)

    def forward(self, input):
        output=self.linear(input)
        output = self.activation(output)
        return output


class OLP(Module):
    def __init__(self, in_features, out_features, activation=ELU(),
                use_batch_norm=False, bias=False):
        # One layer Perceptron
        super(OLP, self).__init__()
        self.linear = Linear(in_features, out_features, bias=bias)
        self.activation = _bn_act(out_features, activation, use_batch_norm)

    def forward(self,  input):
        z = self.linear(input)
        if self.activation:
            z = self.activation(z)
        return z

class Gated_pooling(Module):

    def __init__(self, in_features, out_features, activation=ELU(),
                 use_batch_norm=False, bias=False):
        super(Gated_pooling, self).__init__()
        self.linear1 = Linear(in_features, out_features, bias=bias)
        self.activation1 = _bn_act(out_features, activation, use_batch_norm)
        self.linear2 = Linear(in_features, out_features, bias=bias)
        self.activation2 = _bn_act(out_features, activation, use_batch_norm)

    def forward(self,  input,graph_indices,node_counts):

        z = self.activation1(self.linear1(input))*self.linear2(input)
        graphcount=len(node_counts)
        device=z.device
        blank=torch.zeros(graphcount,z.shape[1]).to(device)
        blank.index_add_(0, graph_indices, z)/node_counts.unsqueeze(1)
        #output = self.activation2(self.linear2(blank)) ################对每个图加起来
        return blank
 

class GatedGraphConvolution(Module):
    def __init__(self,n_node_feat, in_features, out_features, N_shbf ,N_srbf,n_grid_K,n_Gaussian, gate_activation=Sigmoid(),
                 use_node_batch_norm=False, use_edge_batch_norm=False,
                 bias=False, conv_type=0, MLP_activation=ELU()):
        super(GatedGraphConvolution, self).__init__()
        k1= n_Gaussian # k is the number of basis
        k2=n_grid_K**3
        self.linear1_vector = Linear(k1, out_features, bias=bias) # linear for combine sets
        self.linear1_vector_gate = Linear(k1, out_features, bias=bias) # linear for combine sets
        self.activation1_vector_gate = _bn_act(out_features, gate_activation, use_edge_batch_norm)
        self.linear2_vector = Linear(k2, out_features, bias=bias) # linear for plane waves
        self.linear2_vector_gate = Linear(k2, k2, bias=bias) # linear for plane waves
        self.activation2_vector_gate = _bn_act(k2, gate_activation, use_edge_batch_norm)

        self.linear_gate = Linear(in_features, out_features, bias=bias)
        self.activation_gate = _bn_act(out_features, gate_activation, use_edge_batch_norm)

        self.linear_MLP = Linear(in_features, out_features, bias=bias)
        self.activation_MLP = _bn_act(out_features, MLP_activation, use_edge_batch_norm)

   
    def forward(self, input,nodes, edge_sources, edge_targets, rij ,combine_sets,plane_wave,cutoff):

        ni = input[edge_sources].contiguous()
        nj = input[edge_targets].contiguous()
        rij=rij.unsqueeze(1).contiguous()
        mask=rij<cutoff
        delta= (ni-nj)/rij
        final_fe=torch.cat([ni,nj,delta],dim=1)
        del ni,nj,delta
        torch.cuda.empty_cache()
 
        e_gate = self.activation_gate(self.linear_gate(final_fe))
        e_MLP = self.activation_MLP(self.linear_MLP(final_fe))

        z1 = self.linear1_vector(combine_sets)
        gate=self.activation2_vector_gate(self.linear2_vector_gate(plane_wave))
        z2 = self.linear2_vector(plane_wave*gate)
        z =  e_gate * e_MLP * (z1+z2) * mask
        #z =  e_gate * e_MLP * mask
        del z1,z2,e_gate,e_MLP
        torch.cuda.empty_cache()
        output = input.clone()
        output.index_add_(0, edge_sources, z)
        
        return output

class geo_CGNN(nn.Module):
    def __init__(self,n_node_feat, n_hidden_feat,n_GCN_feat,conv_bias,N_block,node_activation, MLP_activation,use_node_batch_norm,use_edge_batch_norm, N_shbf ,N_srbf, cutoff,max_nei,n_MLP_LR,n_grid_K,n_Gaussian):

        super(geo_CGNN, self).__init__()
        self.cutoff=cutoff
        self.N_block=N_block
        node_activation=get_activation(node_activation)
        MLP_activation=get_activation(MLP_activation)
        self.embedding = NodeEmbedding(n_node_feat, n_hidden_feat)
        n2v_concatent_feat = n_hidden_feat*3 #ni+nj+delta

        self.conv = [GatedGraphConvolution(n_node_feat,n2v_concatent_feat, n_hidden_feat, N_shbf ,N_srbf,n_grid_K,n_Gaussian,
                gate_activation=node_activation, # sigmoid
                MLP_activation=MLP_activation, # Elu >1
                use_node_batch_norm=use_node_batch_norm,
                use_edge_batch_norm=use_edge_batch_norm,
                bias=conv_bias)]

        self.MLP_psi2n=[OLP(n_hidden_feat, n_hidden_feat, activation=MLP_activation, use_batch_norm=use_node_batch_norm, bias=conv_bias)]

        self.conv += [GatedGraphConvolution(n_node_feat,n2v_concatent_feat, n_hidden_feat, N_shbf ,N_srbf,n_grid_K,n_Gaussian,
                gate_activation=node_activation,
                MLP_activation=MLP_activation,
                use_node_batch_norm=use_node_batch_norm,
                use_edge_batch_norm=use_edge_batch_norm,
                bias=conv_bias) for _ in range(N_block-1)]
        self.conv=ModuleList(self.conv)

        self.MLP_psi2n = [OLP(n_hidden_feat, n_hidden_feat, activation=MLP_activation, use_batch_norm=use_node_batch_norm, bias=conv_bias) for _ in range(N_block)]
        self.MLP_psi2n=ModuleList(self.MLP_psi2n)
        
        # gated pooling for every block
        self.gated_pooling=[Gated_pooling(n_hidden_feat, n_GCN_feat, activation=MLP_activation ,use_batch_norm=use_node_batch_norm, bias=conv_bias) for _ in range(N_block)]
        self.gated_pooling=ModuleList(self.gated_pooling)
        
        # final linear regression
        self.linear_regression=[OLP(int(n_GCN_feat/2**(i-1)), int(n_GCN_feat/2**i) , activation=MLP_activation, use_batch_norm=use_node_batch_norm, bias=conv_bias) for i in range(1,n_MLP_LR)]
        self.linear_regression += [OLP(int(n_GCN_feat/2**(n_MLP_LR-1)), 1 , activation=None, use_batch_norm=None, bias=conv_bias)]
        self.linear_regression=ModuleList(self.linear_regression)
        '''
        # final linear regression
        self.linear_regression=[OLP(int(n_GCN_feat/i), int(n_GCN_feat/(i+1)) , activation=MLP_activation, use_batch_norm=use_node_batch_norm, bias=conv_bias) for i in range(1,n_MLP_LR)]
        self.linear_regression += [OLP(int(n_GCN_feat/n_MLP_LR), 1 , activation=None, use_batch_norm=None, bias=conv_bias)]
        self.linear_regression=ModuleList(self.linear_regression) 
        '''
    
    def forward(self,nodes,edge_sources,edge_targets,edge_distance,graph_indices,node_counts,combine_sets,plane_wave,output_graph=False):
        x = self.embedding(nodes) 
        Poolingresults=[]
        
        for i in range(self.N_block):
            x = self.conv[i](x,nodes,  edge_sources, edge_targets,edge_distance,combine_sets,plane_wave,self.cutoff)

            poo=self.gated_pooling[i](x,graph_indices,node_counts)
            Poolingresults.append(poo)
            x = self.MLP_psi2n[i](x)
        graph_vec=torch.sum(torch.stack(Poolingresults),dim=0)
        y=graph_vec
        for lr in self.linear_regression:
            y=lr(y)
        if output_graph:
            return y.squeeze(),graph_vec
        else:
            return y.squeeze()
