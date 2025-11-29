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

