################################################
'''
This file is to extract all the information in need to construct the atom graph from .cif files
input:
--data_dir : the .cif file path, the cif files' name should be mp-XXXX or oqmd-XXXX
--output_path 
--name_database : MP or OQMD, 
--cutoff : radius of neighbourhood
--max_num_nbr : max numbers of neighbors
--compress_ratio : percentage of data you want to use

output:
Several .npz files to store the name, lattice vector, nodes features ,neighbors , cell volume
Will output to "output_path"
'''
################################################
from __future__ import print_function, division
from tqdm import tqdm
import csv
import functools
import json
import os
import random
import warnings
import glob
import numpy as np
from operator import itemgetter
from pymatgen.core.structure import Structure
from pydash import py_


def load_materials(filepath):
    try:
        data = np.load(filepath,allow_pickle=True)['materials']
    except UnicodeError:
        data = np.load(filepath, encoding='latin1')['materials']
    return data


def build_config(my_path,config_path):
    # 输入所有cubic.cif数据
    # 建立one-hot编码以及保存设置
    atoms=[]
    all_files = sorted(glob.glob(os.path.join(my_path,'oqmd-*.cif')))
    for path in tqdm(all_files):
        crystal = Structure.from_file(path)
        atoms += list(crystal.atomic_numbers)
    unique_z = np.unique(atoms)
    num_z = len(unique_z)
    print('unique_z:', num_z)
    print('min z:', np.min(unique_z))
    print('max z:', np.max(unique_z))
    z_dict = {z:i for i, z in enumerate(unique_z)}
    # Configuration file
    config = dict()
    config["atomic_numbers"] = unique_z.tolist()
    config["node_vectors"] = np.eye(num_z,num_z).tolist() # One-hot encoding
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return config

def process(config,data_path,radius,max_num_nbr):
    crystal = Structure.from_file(data_path)
    volume=crystal.lattice.volume
    coords=crystal.cart_coords
    lattice=crystal.lattice.matrix
    atoms=crystal.atomic_numbers
    material_id=data_path[:-4]
    atomnum=config['atomic_numbers']
    z_dict = {z:i for i, z in enumerate(atomnum)}
    one_hotvec=np.array(config["node_vectors"])
    atom_fea = np.vstack([one_hotvec[z_dict[atoms[i]]] for i in range(len(crystal))])

    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []

    for i,nbr in enumerate(all_nbrs):
        if len(nbr) < max_num_nbr:
            nbr_fea_idx.append(list(map(lambda x: x[2].tolist(), nbr)) +
                                [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[0].coords.tolist(), nbr)) +
                   [[coords[i][0]+radius,coords[i][1],coords[i][2]]] * (max_num_nbr -len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2].tolist(),
                                        nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[0].coords.tolist(),
                                    nbr[:max_num_nbr])))
    atom_fea=atom_fea.tolist()

    nbr_subtract=[]
    nbr_distance=[]

    for i in range(len(nbr_fea)):
        if nbr_fea[i] != []:
            x=nbr_fea[i]-coords[:,np.newaxis,:][i]
            nbr_subtract.append(x)
            nbr_distance.append(np.linalg.norm(x, axis=1).tolist())
        else:
            nbr_subtract.append(np.array([]))
            nbr_distance.append(np.array([]))

    nbr_fea_idx = np.array(nbr_fea_idx) 
    return material_id,lattice,atom_fea,nbr_fea_idx,nbr_distance,nbr_subtract,volume
##########

def main(data_dir, output_path,name_database ,cutoff,max_num_nbr,compress_ratio,chunk_size=10000):
    if not os.path.isdir(data_dir):
        print('Not found the data directory: {}'.format(data_dir))
        exit(1)
    #config_path=os.path.join(output_path, 'oqmd_config_onehot.json')
    if name_database=='MP':
        config_path='./database/mp_config_onehot.json'
    elif name_database=='OQMD':
        config_path='./database/oqmd_config_onehot.json'

    if os.path.isfile(config_path):
        print('config exists')
        with open(config_path) as f:
            config = json.load(f)
    else:
        print('buiding config')
        config=build_config(data_dir,config_path)

    if name_database=='MP':
        data_files = sorted(glob.glob(os.path.join(data_dir, 'mp-*.cif')))
    elif name_database=='OQMD':
        data_files = sorted(glob.glob(os.path.join(data_dir, 'oqmd-*.cif')))       
    
    for n, chunk in enumerate(tqdm(py_.chunk(data_files[:int(compress_ratio*len(data_files))], chunk_size))):
        graph_names = []
        graph_lattice=[]
        graph_nodes= []
        graph_edges = []
        graph_volume=[]
        graphs=dict()
        for file in chunk:
            material_id,lattice,atom_fea,nbr_fea_idx,nbr_distance,nbr_subtract,volume = process(config,file,cutoff,max_num_nbr)
            graph_lattice.append(lattice)
            graph_names.append(material_id[len(data_dir)+1:])
            graph_nodes.append(atom_fea)
            graph_edges.append((nbr_fea_idx,nbr_distance,nbr_subtract))
            graph_volume.append(volume)
        for name, lattice,nodes,neighbors,volume in tqdm(zip(graph_names,graph_lattice,graph_nodes, graph_edges,graph_volume)):
            graphs[name] = (lattice,nodes, neighbors,volume)
        np.savez_compressed(os.path.join(output_path,"my_graph_data_{}_{}_{}_{}_{:03d}.npz".format(name_database,int(cutoff),max_num_nbr,int(compress_ratio*100),n)), graph_dict=graphs)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Crystal Graph Coordinator.')
    parser.add_argument('--data_dir', metavar='PATH', type=str, default='database/cif',
                        help='The path to a data directory (default: database/cif)')      
    parser.add_argument('--output_path', metavar='PATH', type=str, default='database/npz',
                        help='The output path (default: database/npz)')               
    parser.add_argument('--name_database', metavar='N', type=str, default='OQMD',
                        help='name of database, MP or OQMD (default:OQMD)')
    parser.add_argument('--cutoff', metavar='N', type=float, default=8,
                        help='cutoff distance of neighbors (default : 8A)')
    parser.add_argument('--max_num_nbr', metavar='N', type=int, default=12,
                        help='max neighbors of each node (default : 12)')
    parser.add_argument('--compress_ratio', metavar='N', type=float, default=1,
                        help='compress_ratio (default : 1)')
    options = vars(parser.parse_args())

    main(**options)

