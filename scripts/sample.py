
'''import torch
from src.model import Makemore 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Checkpoint loaded, device: {device}")

model = Makemore()

model.load_checkpoint('models/checkpoint.pt')

names = model.sample(itos, num_samples=50)
print(names)
'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import Makemore

# Load checkpoint with weights_only=False
checkpoint = torch.load('models/checkpoint.pt', weights_only=False)

# Create model
model = Makemore(
    vocab_size=checkpoint['vocab_size'],
    block_size=checkpoint['block_size'],
    emb_size=checkpoint['emb_size'],
    n_hidden=checkpoint['n_hidden'],
    n_blocks=checkpoint['n_blocks']
)

# Load weights
model.load_checkpoint('models/checkpoint.pt')

# Get mappings
itos = checkpoint['itos']
block_size = checkpoint['block_size']

# Sample
g_sample = torch.Generator().manual_seed(2147483647 + 10)
names = model.sample(itos, num_samples=20, generator=g_sample)

print("\nGenerated Names:\n")
print("=" * 40)
for i, name in enumerate(names, 1):
    print(f"{i:2d}. {name}")
print("=" * 40)