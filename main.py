import sys
import torch
import copy
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from model.trainer import *
from data_generator import get_data
from model.ebm import CategoricalScoreModel

# Loading data

graph_type = sys.argv[1]
config_id = sys.argv[2]
seed = sys.argv[3]
degree = sys.argv[4]



dataset = get_data('dataset', config_id, graph_type, seed, 6, degree)
trainer = Trainer(dataset, batch_size=512)


vocab_size = dataset.max_cardinality + 1


# Define configuration for the test model
config = argparse.Namespace(
    net_arch='mlp',
    vocab_size=vocab_size,
    cat_embed_size=None,
    num_layers=3,
    embed_dim=None,
    time_scale_factor=100.,
    discrete_dim=dataset.num_nodes 
)


# Create the model

num_epochs = 300
num_timesteps = 3000
base_dim = 2

ordering = []
corrects = 0


lr = 0.0001

    

for i in range(dataset.num_nodes):

    print(f'------------------- Processing variable {i}: -------------------')
    xt, true_leaf, keep_indices = trainer._prepare_data(ordering)
    xt = xt.to(trainer.device)

    config.cat_embed_size = config.embed_dim = config.discrete_dim * base_dim 
    

    model = CategoricalScoreModel(config)
    
    
    model.to(trainer.device)    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pred_leaf, correct = trainer.train(xt, model, optimizer, true_leaf, num_timesteps, num_epochs) 
    print(f'Final predicted leaf: {keep_indices[pred_leaf]} ({pred_leaf})')
    ordering.append(keep_indices[pred_leaf])
    config.discrete_dim = config.discrete_dim - 1

    if correct: corrects += 1


from utils.topo import D_top
from utils.io import write_pickle


top_order = ordering[::-1]
div = D_top(dataset.B_bin, top_order)
num_edges = dataset.B_bin.sum()
print(f'D-top: {div}/{num_edges}')
print(f'# correct nodes: {corrects}/{dataset.num_nodes}')
