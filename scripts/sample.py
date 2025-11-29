
import torch
import torch.nn.functional as F
from train import layers, Bag,block_size,itos

# Turn to the eval mode, not training mode
for layer in layers:
    if hasattr(layer, 'training'):
        layer.training = False
# Sample 
g_sample = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(20):
    out = []
    context = [0] * block_size
    
    while True:
        context_tensor = torch.tensor([context])
        activations = context_tensor
        for layer in layers:
            activations = layer(activations)
        
        logits = activations
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g_sample).item()
        
        context = context[1:] + [ix]
        out.append(ix)
        
        if ix == 0:
            break
    
    print(''.join(itos[i] for i in out))