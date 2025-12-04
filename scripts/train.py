import torch
import sys
import yaml
from src.data import DataPrep
from src.model import Makemore
import torch.nn.functional as F
import torch.optim as optim

# Load the config files
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

# Create the Generator, using manual seed for reproducibility
g = torch.Generator().manual_seed(config['seed'])

# Build the model
model = Makemore(vocab_size = data_prep_train.vocab_size,
                 block_size = config['block_size'],
                n_hidden = config['n_hidden'],
                emb_size = config['emb_size'],
                n_blocks = config['n_blocks'],
                generator = g
                )

# We shall create an evaluating function
@torch.no_grad()
def eval_split(X,Y,split = 'train'):
    # Set the data to eval mode
    model.eval_mode()
    # Compute the loss
    logits = model(X)
    loss = F.cross_entropy(logits,Y)
    print(f'{split} loss: {loss.item():.4f}')
    # Before returning, we set the data back to the training mode, for training later
    model.train_mode()
    return loss.item()

# Create an optimizer for our model, so that we won't need to adjust the lr by hand
optimizer = optim.AdamW(model.parameters,lr = config['lr'])
# Now we train
def train_model(max_steps,batch_size):
    for i in range(max_steps):
        # Construting the mini-batch
        ix = torch.randint(0,Xtr.shape[0],(batch_size,), generator = g)
        X_batch, Y_batch = Xtr[ix], Ytr[ix]
        # Set the model to training mode
        model.train_mode()
        # Implement the forward pass
        logits = model(X_batch)
        loss = F.cross_entropy(logits,Y_batch)

        # Implement the backward pass with the optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # Equal to updating the parameters
        # Set the gradient first
        for p in model.parameters:
            p.requires_grad = True
        
        # Now we don't need to set those things
        '''
        for p in parameters:
            p.grad = None
        loss.backward()

        # Update
        lr = 0.01 if i < 20000 else 0.001
        for p in parameters:
            p.data += -lr * p.grad
        '''

        # Now we evaluate it after some intervals
        if i % 10000 == 0:
            print(f'\nStep {i}:')
            eval_split(Xtr,Ytr,'train')
            eval_split(Xdev,Ydev,'val')

    print(f'\nFinal Evaluation:')
    eval_split(Xtr,Ytr,'train')
    eval_split(Xdev,Ydev,'val')
    eval_split(Xte,Yte,'test')
    print("\nSaving model checkpoint...")
    model.save_checkpoint(data_prep_train, 'models/checkpoint.pt')
    print("Training complete!")

train_model(max_steps=config['max_steps'], batch_size=config['batch_size'])



    

