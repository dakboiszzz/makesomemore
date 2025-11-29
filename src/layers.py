import torch

# In this file we recreate some Pytorch layers, which is far less complicated than the originals
# We also create the method "parameters", which is simply for setting the gradient

# Embedding layer, changed the generator to the init, set to None but we will use our seed later
class Embedding():
    def __init__(self,vocab_size,emb_size, generator = None):
        self.emb_matrix = torch.randn((vocab_size,emb_size),generator = generator)
    def __call__(self,x):
        return self.emb_matrix[x]
    def parameters(self):
        return [self.emb_matrix]
    
# Flatten layer, using Pytorch trick
class Flatten():
    def __call__(self,x):
        return x.view(x.shape[0],-1)
    def parameters(self):
        return []
    
# Also take out the generator here, we have the flag variable for the bias
# which is useful in the case of BatchNorm, when we have to turn it down
class Linear():
    def __init__(self,fan_in,fan_out,bias = True, generator = None):
        # Use * rather than / for numerical stability
        self.weight = torch.randn((fan_in,fan_out),generator = generator) * (fan_in ** -0.5)
        self.bias = torch.zeros(fan_out) if bias else None
    def __call__(self,x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
# Our activation function
class Tanh():
    def __call__(self,x):
        return torch.tanh(x)
    def parameters(self):
        return []
    
# BatchNorm, separate training and eval mode
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
