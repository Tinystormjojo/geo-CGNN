# Quick Start

## 1. download the cif file to the folder "./database/cif"

The material used in our paper listed in files targets_XXX from the folder "./database"
There are already several cif files for reference

## 2. run AtomGraph.py to creat the input data of the model

e.g. run:

python AtomGraph.py --data_dir ./database/cif --name_database MP_test1 --cutoff 8 --max_num_nbr 12 --compress_ratio 1

The compressed {name_database}+{cutoff}+{max_num_nbr}+{compress_ratio}.npz will be generated to the folder "./database/npz". Pls read the code for the details of the parameters 

## 3. run process_geo_CGNN.py to train the model

  e.g. run:

  python process_geo_CGNN.py --n_hidden_feat 192 --n_GCN_feat 192 --cutoff 8 --max_nei 12 --n_MLP_LR 3 --num_epochs 300 --batch_size 300 --target_name formation_energy_per_atom --milestones 250 --gamma 0.1 --test_ratio 0.2 --datafile_name my_graph_data_MEGNetPrim_8_12_100_ --database CGCNN_Ef --n_grid_K 4 --n_Gaussian 64 --N_block 5 --lr 1e-3 &
