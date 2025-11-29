import torch
import sys
import yaml
from src.data import DataPrep
from src.layers import Embedding, Flatten, Linear, Tanh, BatchNorm1d

def main():
    # Load the config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # Load the data
    data_path = config['data_path']
    with open('data_path','r') as f:
        words = f.read().splitlines()
    data_prep = DataPrep(words, block_size = config['block_size'])
    X,Y = data_prep.getData()

    # Load all the config 
    vocab_size = len(data_prep.string_to_int())
    block_size = config['block_size']
    n_hidden = config['n_hidden']
    emb_size = config['emb_size']

    # Create the Generator, using manual seed for reproducibility
    g = torch.Generator().manual_seed(config['seed'])

    # Build the model

    # Layers and parameters
    layers = [Embedding(vocab_size,emb_size),Flatten(),
            Linear((block_size * emb_size),n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden,n_hidden,bias = False),                 BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden,n_hidden,bias = False),                 BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden,n_hidden,bias = False),                 BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden,n_hidden,bias = False),                 BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden,vocab_size, bias = False), BatchNorm1d(vocab_size) ]
    
    # Adjusting some layers for efficiency
    with torch.no_grad():
    # last layer: make less confident
        layers[-1].bngain *= 0.1
        #layers[-1].weight *= 0.1
        # all other layers: apply gain
        for layer in layers[:-1]:
            if isinstance(layer, Linear):
                layer.weight *= config['gain'] #5/3

    # Collecting the parameters
    parameters = []
    for layer in layers:
        parameters += layer.parameters()


    

