# The model architecture will be shown here, this is for generating when calling API
import torch
from layers import Embedding, Flatten, Linear, BatchNorm1d, Tanh

class Makemore():
    def __init__(self, vocab_size, block_size, emb_size = 10, n_hidden = 5, n_blocks = 5, generator = None):
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.block_size = block_size
        self.n_hidden = n_hidden

        # Build the layers
        self.layers = self._build_layers(vocab_size, block_size, emb_size, n_hidden, n_blocks, generator)

        # Take the parameters
        self.parameters = self._collect_parameters()
        self.init_weights()
    def build_layers(self, vocab_size, block_size, emb_size, n_hidden, n_blocks, generator):
        # Input layers
        layers = [
            Embedding(vocab_size, emb_size), Flatten(), #Embedding
            Linear((block_size * emb_size),n_hidden,bias = False),
            BatchNorm1d(n_hidden),
            Tanh()
        ]
        # Adding blocks
        for _ in range(n_blocks):
            layers.extend([
                Linear(n_hidden,n_hidden, bias = False),
                BatchNorm1d(n_hidden),
                Tanh()
            ])
        # Output layer
        layers.extend([
            Linear(n_hidden,vocab_size, bias = False), 
            BatchNorm1d(vocab_size) 
        ])
        return layers
    
