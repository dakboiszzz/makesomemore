import torch
import torch.nn.functional as F

# Load the dataset
words = open('names.txt').read().splitlines()

# Create DataPrep class, set block_size
class DataPrep():
    def __init__(self,words,block_size = 2):
        self.words = words
        self.block_size = block_size
    def getChars(self):
        self.chars = []
        ## This is my code but it's not really good
        """
        for word in self.words:
            for char in list(word):
                if char not in self.chars:
                    self.chars.append(char)
        self.chars = sorted(self.chars)
        """
        self.chars = sorted(list(set(''.join(self.words))))
        return self.chars
    def string_to_int(self):
        stoi = dict()
        chars = self.getChars()
        for index,char in enumerate(chars):
            stoi[char] = index + 1
        stoi['.'] = 0
        self.stoi = stoi
        return self.stoi
    def int_to_string(self):
        stoi = self.string_to_int()
        self.itos = {i:s for s,i in stoi.items()}
        return self.itos
    def getData(self):
        chars = self.getChars()

        stoi = self.string_to_int()
        itos = self.int_to_string()
        X = []
        Y = []
        for word in self.words:
            block = [0] * self.block_size
            word = word + '.'
            for char in word:
                X.append(block)
                index = stoi[char]
                Y.append(index)
                block = block[1:] + [index]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X,Y
class Embedding():
    def __init__(self,vocab_size,emb_size):
        self.emb_matrix = torch.randn((vocab_size,emb_size),generator = g)
    def __call__(self,x):
        return self.emb_matrix[x]
    def parameters(self):
        return [self.emb_matrix]
class Flatten():
    def __call__(self,x):
        return x.view(x.shape[0],-1)
    def parameters(self):
        return []
class Linear():
    def __init__(self,fan_in,fan_out,bias = True):
        # Use * rather than / for numerical stability
        self.weight = torch.randn((fan_in,fan_out),generator = g) * (fan_in ** -0.5)
        self.bias = torch.zeros(fan_out) if bias else None
    def __call__(self,x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
class Tanh():
    def __call__(self,x):
        return torch.tanh(x)
    def parameters(self):
        return []
    
class BatchNorm1d():
    def __init__(self,dim,eps = 1e-5,momentum = 0.001):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # parameters
        self.bngain = torch.ones(dim)
        self.bnbias = torch.zeros(dim)

        # running mean and variation

        self.running_mean = torch.zeros(dim)
        self.running_var  = torch.ones(dim)

    def __call__(self,x):
        if self.training:
            bnmean = torch.mean(x,0,keepdim = True)
            bnvar = torch.var(x,0,keepdim = True)
        else:
            bnmean = self.running_mean
            bnvar = self.running_var
        xhat = (x - bnmean) / torch.sqrt(bnvar + self.eps)
        self.out = self.bngain * xhat + self.bnbias
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * bnmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * bnvar
        return self.out
    def parameters(self):
        return [self.bngain,self.bnbias]


# Initialize 
Bag = DataPrep(words,3)
X,Y = Bag.getData()
g = torch.Generator().manual_seed(2147483647)
itos = Bag.int_to_string()

# Hyperparameters for the forward pass
vocab_size = len(Bag.string_to_int())
block_size = 3
n_hidden = 100
emb_size = 10

# Layers and parameters
layers = [Embedding(vocab_size,emb_size),Flatten(),
          Linear((block_size * emb_size),n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(),
          Linear(n_hidden,n_hidden,bias = False),                 BatchNorm1d(n_hidden), Tanh(),
          Linear(n_hidden,n_hidden,bias = False),                 BatchNorm1d(n_hidden), Tanh(),
          Linear(n_hidden,n_hidden,bias = False),                 BatchNorm1d(n_hidden), Tanh(),
          Linear(n_hidden,n_hidden,bias = False),                 BatchNorm1d(n_hidden), Tanh(),
          Linear(n_hidden,vocab_size, bias = False), BatchNorm1d(vocab_size) ]
with torch.no_grad():
  # last layer: make less confident
  layers[-1].bngain *= 0.1
  #layers[-1].weight *= 0.1
  # all other layers: apply gain
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 1.0 #5/3
parameters = []
for layer in layers:
    parameters += layer.parameters()


# Create loop / feeding minibatches
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


train_model(max_steps=100000, batch_size=32)