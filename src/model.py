# The model architecture will be shown here, this is for generating when calling API
import torch
from src.layers import Embedding, Flatten, Linear, BatchNorm1d, Tanh

class Makemore():
    def __init__(self, vocab_size, block_size, emb_size = 10, n_hidden = 5, n_blocks = 5, generator = None):
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.block_size = block_size
        self.n_hidden = n_hidden
        self.generator = generator

        # Build the layers
        self.layers = self._build_layers(vocab_size, block_size, emb_size, n_hidden, n_blocks, generator)

        # Take the parameters
        self.parameters = self._collect_parameters()
        self._init_weights()
    def _build_layers(self, vocab_size, block_size, emb_size, n_hidden, n_blocks, generator):
        # Input layers
        layers = [
            Embedding(vocab_size, emb_size, generator = self.generator), Flatten(), #Embedding
            Linear((block_size * emb_size),n_hidden,bias = False, generator = self.generator),
            BatchNorm1d(n_hidden),
            Tanh()
        ]
        # Adding blocks
        for _ in range(n_blocks):
            layers.extend([
                Linear(n_hidden,n_hidden, bias = False,generator = self.generator),
                BatchNorm1d(n_hidden),
                Tanh()
            ])
        # Output layer
        layers.extend([
            Linear(n_hidden,vocab_size, bias = False, generator = self.generator), 
            BatchNorm1d(vocab_size) 
        ])
        return layers
    def _collect_parameters(self):
        parameters = []
        for layer in self.layers:
            parameters += layer.parameters()
        return parameters
    def _init_weights(self):
        # Adjusting some layers for efficiency
        with torch.no_grad():
            # Last layer; Make less confident
            self.layers[-1].bngain != 0.1
            # Other linear layers apply gain = 1.0
            for layer in self.layers[:-1]:
                if isinstance(layer,Linear):
                    layer.weight *= 1.0
    # Implement the forward pass
    def __call__(self,x):
        return self.forward(x)
    def forward(self,x):
        activations = x
        for layer in self.layers:
            activations = layer(activations)
        return activations

    # Setting training and eval mode
    def train_mode(self):
        for layer in self.layers:
            layer.training = True
    def eval_mode(self):
        for layer in self.layers:
            layer.training = False
    
