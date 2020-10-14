import os.path
import json
import scipy
from scipy.special import jn_zeros,jn,sph_harm
import numpy as np
import pandas as pd
import glob
import torch
from torch.utils.data import Dataset
import tqdm

def load_graph_data(file_path):
    Total={}
    for path in file_path:
        print('loading : {}'.format(path))
        try:
            graphs = np.load(path,allow_pickle=True)['graph_dict'].item()
        except UnicodeError:
            graphs = np.load(path, encoding='latin1',allow_pickle=True)['graph_dict'].item()
            graphs = { k.decode() : v for k, v in graphs.items() }
        Total={**Total,**graphs}
    print('load successed, final volume : {}'.format(len(Total)))
    return Total



def Atomgraph_collate(batch):
    nodes = []
    edge_distance=[]
    edge_targets=[]
    edge_sources = []
    graph_indices = []
    node_counts = []
    targets = []
    combine_sets =[]
    plane_wave = []
    total_count = 0

    for i, (graph, target) in enumerate(batch):

        # Numbering for each batch
        nodes.append(graph.nodes) 
        edge_distance.append(graph.distance) 
        edge_sources.append(graph.edge_sources + total_count) # source number of each edge
        edge_targets.append(graph.edge_targets + total_count) # target number of each edge
        combine_sets.append(graph.combine_sets)
        plane_wave.append(graph.plane_wave)
        node_counts.append(len(graph))
        targets.append(target)
        graph_indices += [i] * len(graph)
        total_count += len(graph)

    combine_sets=np.concatenate(combine_sets,axis=0)
    plane_wave=np.concatenate(plane_wave,axis=0)
    nodes = np.concatenate(nodes,axis=0)
    edge_distance = np.concatenate(edge_distance,axis=0)
    edge_sources = np.concatenate(edge_sources,axis=0)
    edge_targets = np.concatenate(edge_targets,axis=0)
    input = geo_CGNN_Input(nodes,edge_distance, edge_sources, edge_targets, graph_indices, node_counts,combine_sets,plane_wave)
    targets = torch.Tensor(targets)
    return input, targets

class AtomGraph(object):
    def __init__(self, graph,cutoff,N_shbf,N_srbf,n_grid_K,n_Gaussian):
        lattice, self.nodes, neighbors,volume = graph
        nei=neighbors[0] 
        distance=neighbors[1] 
        vector=neighbors[2] 
        n_nodes = len(self.nodes) 
        self.nodes = np.array(self.nodes, dtype=np.float32)
        self.edge_sources = np.concatenate([[i] * len(nei[i]) for i in range(n_nodes)])
        self.edge_targets=np.concatenate(nei)
        edge_vector = np.array(vector, dtype=np.float32)
        self.edge_index = np.concatenate([range(len(nei[i])) for i in range(n_nodes)])
        self.vectorij= edge_vector[self.edge_sources,self.edge_index]
        edge_distance = np.array(distance, dtype=np.float32)
        self.distance= edge_distance[self.edge_sources,self.edge_index]
        combine_sets=[]
        # gaussian radial
        N=n_Gaussian
        for n in range(1,N+1):
            phi=Phi(self.distance,cutoff)
            G=gaussian(self.distance,miuk(n,N,cutoff),betak(N,cutoff))
            combine_sets.append(phi*G)
        self.combine_sets=np.array(combine_sets, dtype=np.float32).transpose()

        # plane wave
        grid=n_grid_K
        kr=np.dot(self.vectorij,get_Kpoints_random(grid,lattice,volume).transpose()) 
        self.plane_wave=np.cos(kr)/np.sqrt(volume)

    def __len__(self):
        return len(self.nodes)
class AtomGraphDataset(Dataset):
    def __init__(self, path, filename,database, target_name,cutoff,N_shbf,N_srbf,n_grid_K,n_Gaussian):
        super(AtomGraphDataset, self).__init__()
        

        target_path = os.path.join(path, "targets_"+database+".csv")

        if target_name == 'band_gap' and database=='MP':
            target_path = os.path.join(path, "targets_"+database+'_Eg'+".csv")
        elif target_name == 'formation_energy_per_atom' and database=='MP':
            target_path = os.path.join(path, "targets_"+database+'_Ef'+".csv")

        df = pd.read_csv(target_path).dropna(axis=0,how='any') 

        if target_name == 'band_gap' and (database=='OQMD' or database=='MEGNet_2018'):
            df=df[df['band_gap']!=0]

                     
        graph_data_path = sorted(glob.glob(os.path.join(path, 'npz/'+filename+'*.npz')))
        print('The number of files = {}'.format(len(graph_data_path)))
        self.graph_data = load_graph_data(graph_data_path)
        graphs=self.graph_data.keys()

        self.graph_names=df.loc[df['id'].isin(graphs)].id.values.tolist()
        self.targets=np.array(df.loc[df['id'].isin(graphs)][target_name].values.tolist())
        print('the number of valid targets = {}'.format(len(self.targets)))
        print('start to constructe AtomGraph')
        graph_data=[]
        for i,name in enumerate(self.graph_names):
            graph_data.append(AtomGraph(self.graph_data[name],cutoff,N_shbf,N_srbf,n_grid_K,n_Gaussian))
            if i%2000==0 and i>0:
                print('{} graphs constructed'.format(i))
        print('finish constructe the graph')
        self.graph_data=graph_data
                           
        assert(len(self.graph_data)==len(self.targets))
        print('The number of valid graphs = {}'.format(len(self.targets)))
                
    def __getitem__(self, index):
        return self.graph_data[index], self.targets[index]

    def __len__(self):
        return len(self.graph_names)


# 构建torch的输入张量
class geo_CGNN_Input(object):
    def __init__(self,nodes,edge_distance,edge_sources, edge_targets, graph_indices, node_counts,combine_sets,plane_wave):
        self.nodes = torch.Tensor(nodes)
        self.edge_distance = torch.Tensor(edge_distance)
        self.edge_sources = torch.LongTensor(edge_sources)
        self.edge_targets = torch.LongTensor(edge_targets)
        self.graph_indices = torch.LongTensor(graph_indices)
        self.node_counts = torch.Tensor(node_counts)
        self.combine_sets=torch.Tensor(combine_sets)
        self.plane_wave=torch.Tensor(plane_wave)

    def __len__(self):
        return self.nodes.size(0)


def a_SBF(alpha,l,n,d,cutoff):
    root=float(jn_zeros(l,n)[n-1])
    return jn(l,root*d/cutoff)*sph_harm(0,l,np.array(alpha),0).real*np.sqrt(2/cutoff**3/jn(l+1,root)**2)

def a_RBF(n,d,cutoff):
    return np.sqrt(2/cutoff)*np.sin(n*np.pi*d/cutoff)/d

def get_Kpoints_random(q,lattice,volume):
    a0=lattice[0,:]
    a1=lattice[1,:]
    a2=lattice[2,:]
    unit=2*np.pi*np.vstack((np.cross(a1,a2),np.cross(a2,a0),np.cross(a0,a1)))/volume
    ur=[(2*r-q-1)/2/q for r in range(1,q+1)]
    points=[]
    for i in ur:
        for j in ur:
            for k in ur:
                points.append(unit[0,:]*i+unit[1,:]*j+unit[2,:]*k)
    points=np.array(points) 
    return points  


def Phi(r,cutoff):
    return 1-6*(r/cutoff)**5+15*(r/cutoff)**4-10*(r/cutoff)**3
def gaussian(r,miuk,betak):
    return np.exp(-betak*(np.exp(-r)-miuk)**2)
def miuk(n,K,cutoff):
    # n=[1,K]
    return np.exp(-cutoff)+(1-np.exp(-cutoff))/K*n
def betak(K,cutoff):
    return (2/K*(1-np.exp(-cutoff)))**(-2)


