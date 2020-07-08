
import time
import json
import os
import copy
import numpy as np
import pandas as pd
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from models import Model,DFTGN_AtomGraph
from data_utils import AtomGraphDataset, Atomgraph_collate


def use_setpLR(param):
    ms = param["milestones"]
    return ms[0] < 0

def create_model(device, model_param, optimizer_param, scheduler_param):
    model=DFTGN_AtomGraph(**model_param)

    
    clip_value = optimizer_param.pop("clip_value")
    optim_name = optimizer_param.pop("optim")
    if optim_name == "sgd":
        optimizer = optim.SGD(model.parameters(), momentum=0.9,
                              nesterov=True, **optimizer_param)
    elif optim_name == "adam":
        optimizer = optim.Adam(model.parameters(), **optimizer_param)
    elif optim_name == "amsgrad":
        optimizer = optim.Adam(model.parameters(), amsgrad=True,
                               **optimizer_param)
    elif optim_name == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr_decay=0.1, **optimizer_param)
    else:
        raise NameError("optimizer {} is not supported".format(optim_name))
    use_cosine_annealing = scheduler_param.pop("cosine_annealing")
    if use_cosine_annealing:
        params = dict(T_max=scheduler_param["milestones"][0],
                      eta_min=scheduler_param["gamma"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    
    elif use_setpLR(scheduler_param):
        scheduler_param["step_size"] = abs(scheduler_param["milestones"][0])
        scheduler_param.pop("milestones")
        scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_param)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_param)
    N_block=model_param.pop('N_block')
    cutoff=model_param.pop('cutoff')
    max_nei=model_param.pop('max_nei')
    name=str(N_block)+'_'+str(cutoff)+'_'+str(max_nei)
    return Model(device, model, name, optimizer, scheduler, clip_value)

def main(device, model_param, optimizer_param, scheduler_param, dataset_param, dataloader_param,
         num_epochs, seed, load_model,pred,pre_trained_model_path):
    N_block=model_param['N_block']
    cutoff=model_param['cutoff']
    max_nei=model_param['max_nei']
    print("Seed:", seed)
    print()
    torch.manual_seed(seed)
    # Create dataset
    dataset = AtomGraphDataset(dataset_param["dataset_path"],dataset_param['datafile_name'],dataset_param["database"], dataset_param["target_name"],model_param['cutoff'],model_param['N_shbf'],model_param['N_srbf']) 
    
    dataloader_param["collate_fn"] = Atomgraph_collate

    model_param['n_node_feat'] = dataset.graph_data[0].nodes.shape[1]

    test_ratio=dataset_param['test_ratio']
    n_graph=len(dataset.graph_data)
    indices=list(range(n_graph))
    random.shuffle(indices)

    # special case, manually setting
    if test_ratio==-1:
        split = {"train": indices[0:40000], "val": indices[40000:45000], "test": indices[45000: 50000]}
    else:
    # normal case
        n_val=int(n_graph*test_ratio)
        n_train=n_graph-2*n_val
        split = {"train": indices[0:n_train], "val": indices[n_train:n_train+n_val], "test": indices[n_train+n_val: n_graph]}
        

    print(" ".join(["{}: {}".format(k, len(x)) for k, x in split.items()]))

    # Create a DFTGN model
    model = create_model(device, model_param, optimizer_param, scheduler_param)
    if load_model:
        print("Loading weights from mymodel.pth")
        model.load(model_path=pre_trained_model_path)
        print("Model loaded at: {}".format(pre_trained_model_path)) 
    

    if not pred:
    # Train
        train_sampler = SubsetRandomSampler(split["train"])
        val_sampler = SubsetRandomSampler(split["val"])
        train_dl = DataLoader(dataset, sampler=train_sampler,pin_memory=True, **dataloader_param)
        trainD=[n for n in train_dl]
        inp=[n for n in trainD[0]]
        inp=inp[0]
        print('******************',inp.nodes.size())

        val_dl = DataLoader(dataset, sampler=val_sampler,pin_memory=True, **dataloader_param)

        print('start training')
        model.train(train_dl, val_dl, num_epochs)


        if num_epochs > 0:
            model.save()

    # Test
    test_set = Subset(dataset, split["test"])
    test_dl = DataLoader(test_set,pin_memory=True, **dataloader_param)
    outputs, targets = model.evaluate(test_dl)
    names = [dataset.graph_names[i] for i in split["test"]]
    df_predictions = pd.DataFrame({"name": names, "prediction": outputs, "target": targets})

    name=str(N_block)+'_'+str(cutoff)+'_'+str(max_nei)
    df_predictions.to_csv("data/test_predictions_{}.csv".format(name), index=False)

    print("\nEND")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Crystal Graph Neural Networks")


    parser.add_argument("--n_hidden_feat", type=int, default=128)
    parser.add_argument("--conv_bias", type=bool, default=False) # 网络中是否有b
    parser.add_argument("--n_GCN_feat", type=int, default=128) #波函数频道
    parser.add_argument("--N_block", type=int, default=6)
    parser.add_argument("--N_shbf", type=int, default=6)
    parser.add_argument("--N_srbf", type=int, default=6)
    parser.add_argument("--cutoff", type=int, default=8) # 单位为埃
    parser.add_argument("--max_nei", type=int, default=12) # 用于命名
    parser.add_argument("--n_MLP_LR", type=int, default=3) # 最后线性回归层数
    parser.add_argument("--node_activation", type=str, default="Sigmoid")
    parser.add_argument("--MLP_activation", type=str, default="Elu")
    parser.add_argument("--use_node_batch_norm", type=bool, default=True)
    parser.add_argument("--use_edge_batch_norm", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=8e-3)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--clip_value", type=float, default=0)
    parser.add_argument("--milestones", nargs='+', type=int, default=[20])
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--cosine_annealing", action='store_true')
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--dataset_path", type=str, default='database')
    parser.add_argument("--datafile_name", type=str, default="my_graph_data_OQMD_8_12_100")
    parser.add_argument("--database", type=str, default="OQMD")
    parser.add_argument("--target_name", type=str, default='formation_energy_per_atom')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--load_model", action='store_true')
    parser.add_argument("--pred", action='store_true')
    parser.add_argument("--pre_trained_model_path", type=str, default='./pre_trained/model_Ef_OQMD.pth')
    options = vars(parser.parse_args())

    # set cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    # Model parameters
    model_param_names = ['n_hidden_feat','conv_bias','n_GCN_feat','N_block','N_shbf','N_srbf','cutoff','max_nei','n_MLP_LR','node_activation','MLP_activation','use_node_batch_norm','use_edge_batch_norm']
    model_param = {k : options[k] for k in model_param_names if options[k] is not None}
    if model_param["node_activation"].lower() == 'none':
        model_param["node_activation"] = None
    if model_param["MLP_activation"].lower() == 'none':
        model_param["MLP_activation"] = None
    print("Model_param:", model_param)
    print()

    # Optimizer parameters
    optimizer_param_names = ["optim", "lr", "weight_decay", "clip_value"]
    optimizer_param = {k : options[k] for k in optimizer_param_names if options[k] is not None}
    if optimizer_param["clip_value"] == 0.0:
        optimizer_param["clip_value"] = None
    print("Optimizer:", optimizer_param)
    print()

    # Scheduler parameters
    scheduler_param_names = ["milestones", "gamma", "cosine_annealing"]
    #scheduler_param_names = ["milestones", "gamma"]
    scheduler_param = {k : options[k] for k in scheduler_param_names if options[k] is not None}
    print("Scheduler:", scheduler_param)
    print()

    # Dataset parameters
    dataset_param_names = ["dataset_path",'datafile_name','database', "target_name","test_ratio"]
    dataset_param = {k : options[k] for k in dataset_param_names if options[k] is not None}
    print("Dataset:", dataset_param)
    print()

    # Dataloader parameters
    dataloader_param_names = ["num_workers", "batch_size"]
    dataloader_param = {k : options[k] for k in dataloader_param_names if options[k] is not None}
    print("Dataloader:", dataloader_param)
    print()

    main(device, model_param, optimizer_param, scheduler_param, dataset_param, dataloader_param,
         options["num_epochs"], options["seed"], options["load_model"],options["pred"],options["pre_trained_model_path"])
