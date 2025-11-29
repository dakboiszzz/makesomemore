import torch
import sys
import yaml
from src.data import DataPrep
from src.layers import Embedding, Flatten, Linear, Tanh, BatchNorm1d

def main():
    # Load the config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    

