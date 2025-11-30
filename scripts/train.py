import torch
import sys
import yaml
from src.data import DataPrep
from src.layers import Embedding, Flatten, Linear, Tanh, BatchNorm1d
import torch.nn.functional as F


# Load the config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
# Load the data
data_path = config['data_path']
with open(data_path,'r') as f:
    words = f.read().splitlines()

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

data_prep_train = DataPrep(words[:n1], block_size = config['block_size'])
data_prep_dev = DataPrep(words[n1:n2], block_size = config['block_size'])
data_prep_test = DataPrep(words[n2:], block_size = config['block_size'])

Xtr,  Ytr  = data_prep_train.getData()    # 80%
Xdev, Ydev = data_prep_dev.getData()   # 10%
Xte,  Yte  = data_prep_test.getData()    # 10%

# Load all the config 
vocab_size = len(data_prep_train.string_to_int())
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

# Now we train
def train_model(max_steps,batch_size):
    for i in range(max_steps):
        # Construting the mini-batch
        ix = torch.randint(0,X.shape[0],(batch_size,), generator = g)
        X_batch, Y_batch = X[ix], Y[ix]

        # Set the gradient first
        for p in parameters:
            p.requires_grad = True

        # Implement the forward pass
        activations = X_batch
        for layer in layers:
            activations = layer(activations)
        loss = F.cross_entropy(activations,Y_batch)

        # Implement the backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # Update
        lr = 0.01 if i < 20000 else 0.001
        for p in parameters:
            p.data += -lr * p.grad
        if i % 10000 == 0:
            print(f'Step {i}, Loss: {loss.item():.4f}')

    print(f'Final loss: {loss}')

train_model(max_steps=config['max_steps'], batch_size=config['batch_size'])



    

