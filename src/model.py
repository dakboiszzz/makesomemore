# The model architecture will be shown here, this is for generating when calling API
import torch
import torch.nn.functional as F
from src.layers import Embedding, Flatten, Linear, BatchNorm1d, Tanh

class Makemore():
    def __init__(self, vocab_size = 27, block_size = 3, emb_size = 10, n_hidden = 5, n_blocks = 5, generator = None,device = 'cpu'):
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.block_size = block_size
        self.n_hidden = n_hidden
        self.n_blocks = n_blocks
        self.generator = generator

        # Build the layers
        self.layers = self._build_layers(vocab_size, block_size, emb_size, n_hidden, n_blocks, generator)

        # Take the parameters
        self.parameters = self._collect_parameters()
        self._init_weights()

        # Move to device
        self.to(device)
    def _build_layers(self, vocab_size, block_size, emb_size, n_hidden, n_blocks, generator):
        # Input layers
        layers = [
            Embedding(vocab_size, emb_size, generator = self.generator), Flatten(), #Embedding
            Linear((block_size * emb_size),n_hidden,bias = False, generator = self.generator),
            BatchNorm1d(n_hidden),
            Tanh()
        ]
        # Adding blocks
        for _ in range(self.n_blocks):
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
            self.layers[-1].bngain *= 0.1
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
    def to(self,device):
        #Move all layers to device
        self.device = torch.device(device)
        for layer in self.layers:
            if hasattr(layer, 'emb_matrix'):
                layer.emb_matrix = layer.emb_matrix.to(device)
            if hasattr(layer, 'weight'):
                layer.weight = layer.weight.to(device)
                if layer.bias is not None:
                    layer.bias = layer.bias.to(device)
            if hasattr(layer, 'bngain'):
                layer.bngain = layer.bngain.to(device)
                layer.bnbias = layer.bnbias.to(device)
                layer.running_mean = layer.running_mean.to(device)
                layer.running_var = layer.running_var.to(device)
                layer.bnmean = layer.bnmean.to(device)
                layer.bnvar = layer.bnvar.to(device)
        return self
    def save_checkpoint(self,data_prep,filepath):
        checkpoint = {
            'model_state' : {},
            'vocab_size': self.vocab_size,
            'block_size': self.block_size,
            'emb_size': self.emb_size,
            'n_hidden': self.n_hidden,
            'n_blocks': self.n_blocks,
            'itos': data_prep.itos,
            'stoi': data_prep.stoi
        }
        for i,layer in enumerate(self.layers):
            for j, p in enumerate(layer.parameters()):
                checkpoint['model_state'][f'layer_{i}_param_{j}'] = p.data
        torch.save(checkpoint,filepath)
        print(f'Checkpoint saved to {filepath}')

    def load_checkpoint(self,filepath):
        checkpoint = torch.load(filepath)
        # Load parameters
        for i, layer in enumerate(self.layers):
            for j, p in enumerate(layer.parameters()):
                key = f'layer_{i}_param_{j}'
                if key in checkpoint['model_state']:
                    p.data = checkpoint['model_state'][key]
        return self, checkpoint['itos'], checkpoint['stoi'], checkpoint['block_size']
    def sample(self, itos, num_samples = 20,generator = None):
        self.eval_mode()
        names = []
        for _ in range(num_samples):
            out = []
            context = [0] * self.block_size
            while True:
                context_tensor = torch.tensor([context]).to(self.device)
                activations = context_tensor
                for layer in self.layers:
                    activations = layer(activations)
                
                logits = activations
                probs = F.softmax(logits, dim=1)

                ix = torch.multinomial(probs, num_samples=1,generator =generator).item()
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0:
                    break
                if len(out) > 50: # Safety limit
                    break
            name = ''.join(itos[i] for i in out[:-1])
            names.append(name)
        return names
    def generate(self, char,itos,stoi,generator = None):
        self.eval_mode()
        out = []
        context = [0,0,stoi[char]]
        while True:
            context_tensor = torch.tensor([context]).to(self.device)
            logits = self(context_tensor)
            probs = F.softmax(logits,dim = 1)
            ix = torch.multinomial(probs, num_samples = 1, generator = generator).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
            if len(out) > 50: # Safety limit
                break
        return ''.join(itos[i] for i in out[:-1])

