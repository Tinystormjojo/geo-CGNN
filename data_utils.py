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

        #把数据集中的所有图统一编号
        nodes.append(graph.nodes) # 所有的节点
        edge_distance.append(graph.distance) #所有的边长度
        edge_sources.append(graph.edge_sources + total_count) # 边的头节点编号
        edge_targets.append(graph.edge_targets + total_count) # 边的尾节点编号
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
    input = DFTGNInput(nodes,edge_distance, edge_sources, edge_targets, graph_indices, node_counts,combine_sets,plane_wave)
    targets = torch.Tensor(targets)
    return input, targets

class AtomGraph(object):
    def __init__(self, graph,cutoff,N_shbf,N_srbf):
        lattice, self.nodes, neighbors,volume = graph 
        nei=neighbors[0] #存储每个节点的邻居编号
        distance=neighbors[1] #存储每个节点的邻居距离
        vector=neighbors[2] #存储每个节点的邻居距离向量
        lattice_diagonal=np.sum(lattice,axis=1)
        n_nodes = len(self.nodes) # 总原子个数
        self.nodes = np.array(self.nodes, dtype=np.float32)
        self.edge_sources = np.concatenate([[i] * len(nei[i]) for i in range(n_nodes)])
        self.edge_targets=np.concatenate(nei)
        lattice_diagonal=lattice_diagonal.reshape(1,len(lattice_diagonal)).repeat(len(self.edge_sources),axis=0)
        self.combine_sets=[]
        edge_distance = np.array(distance, dtype=np.float32)
        edge_vector = np.array(vector, dtype=np.float32)
        self.edge_index = np.concatenate([range(len(nei[i])) for i in range(n_nodes)])
        self.vectorij= edge_vector[self.edge_sources,self.edge_index]
        self.distance= edge_distance[self.edge_sources,self.edge_index]

        R=np.sqrt(np.sum(self.vectorij**2,axis=1))*np.sqrt(np.sum((lattice_diagonal)**2,axis=1))
        self.alphaij=np.arccos(np.sum(self.vectorij*lattice_diagonal,axis=1)/R)  
        self.alphaij[np.isnan(self.alphaij)]=0

        # 2D sphere Fourier Bessel basis
        for l in range(N_shbf):
            for n in range(1,N_srbf+1):
                thisone=a_SBF(self.alphaij,l,n,self.distance,cutoff)
                self.combine_sets.append(thisone)
        self.combine_sets=np.array(self.combine_sets, dtype=np.float32).transpose() # [N*(shbf*srbf)]

        # plane wave
        kr=np.dot(self.vectorij,get_Kpoints_4(lattice).transpose()) # N*26
        self.plane_wave=np.cos(kr)/np.sqrt(volume)
        

    def __len__(self):
        return len(self.nodes)
class AtomGraphDataset(Dataset):
    def __init__(self, path, filename,database, target_name,cutoff,N_shbf,N_srbf):
        super(AtomGraphDataset, self).__init__()
        

        target_path = os.path.join(path, "targets_"+database+".csv")

        df = pd.read_csv(target_path).dropna(axis=0,how='any') 

        if target_name == 'band_gap' and database=='OQMD':
            df=df[df['band_gap']!=0]
                     
        graph_data_path = sorted(glob.glob(os.path.join(path, 'npz/'+filename+'*.npz')))
        print('The number of files = {}'.format(len(graph_data_path)))
        self.graph_data = load_graph_data(graph_data_path)
        graphs=self.graph_data.keys()

        self.graph_names=df.loc[df['id'].isin(graphs)].id.values.tolist()
        self.targets=np.array(df.loc[df['id'].isin(graphs)][target_name].values.tolist())
        print('the number of valide targets = {}'.format(len(self.targets)))
        print('start to constructe AtomGraph')
        graph_data=[]
        for i,name in enumerate(self.graph_names):
            graph_data.append(AtomGraph(self.graph_data[name],cutoff,N_shbf,N_srbf))
            if i%2000==0 and i>0:
                print('{} graphs constructed'.format(i))
        print('finish constructe the graph')
        self.graph_data=graph_data
                           
        assert(len(self.graph_data)==len(self.targets))
        print('The number of valide graphs = {}'.format(len(self.targets)))
                
    def __getitem__(self, index):
        return self.graph_data[index], self.targets[index]

    def __len__(self):
        return len(self.graph_names)


# 构建torch的输入张量
class DFTGNInput(object):
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
    # 返回在 d,alpha处的SBF(l,n)
    # l阶第n个根 n>=1 l>=0
    # harm(_,l,_,theta)
    root=float(jn_zeros(l,n)[n-1])
    return jn(l,root*d/cutoff)*sph_harm(0,l,np.array(alpha),0).real*np.sqrt(2/cutoff**3/jn(l+1,root)**2)

def a_RBF(n,d,cutoff):
    #print(cutoff,n,d)
    return np.sqrt(2/cutoff)*np.sin(n*np.pi*d/cutoff)/d

def get_Kpoints_4(lattice):

    lattice=np.reshape(np.linalg.norm(lattice,axis=1),(3,1))
    #Special K points. unit: 2pi/abc 
    unit=2*np.pi/lattice
    unit=np.eye(3)*unit
    # A. Face-Centered-Cubic Bravais Lattice
    b1=np.dot(np.array([[7,3,1],
                [7,1,1],
                [5,5,1],
                [5,3,3],
                [5,3,1],
                [5,1,1],
                [3,3,3],
                [3,3,1],
                [3,1,1],
                [1,1,1]])/8,unit)
    weight1=np.array([[6,3,3,3,6,3,1,3,3,1]])/32
    b1=np.vstack((b1,np.dot(weight1,b1)))

    # B. Body-Centered-Cubic Bravais Lattice
    b2=np.dot(np.array([[1,1,1],
                [3,1,1],
                [3,3,1],
                [3,3,3],
                [5,1,1],
                [5,3,1],
                [5,3,3],
                [7,1,1]])/8,unit)
    weight2=np.array([[1,3,3,1,3,3,1,1]])/16
    b2=np.dot(weight2,b2)

    # C. Hexagonal Bravais Lattice
    unit3=2*np.pi/lattice
    unit3[1,0]*=np.sqrt(3)
    unit3=np.eye(3)*unit3
    b3=np.dot(np.array([[1,1,1],
                [1,1,3],
                [2,2,1],
                [2,2,3],
                [4,4,1],
                [4,4,3],
                [3,1,1],
                [3,1,3],
                [5,1,1],
                [5,1,3],
                [4,2,1],
                [4,2,3]])/np.array([[9,9,8]]),unit3)
    weight3=np.array([[1,1,1,1,1,1,2,2,2,2,2,2]])/18
    b3=np.vstack((b3,np.dot(weight3,b3)))

    # D. Simple-Cubic Bravais Lattice
    b4=np.dot(np.array([[1,1,1],
                [3,1,1],
                [3,3,1],
                [3,3,3]])/8,unit)
    weight4=np.array([[1,3,3,1]])/8
    b4=np.dot(weight4,b4)

    B=np.vstack((b1,b2,b3,b4)) # 26*3
    return B
