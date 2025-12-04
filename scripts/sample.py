import torch
from src.model import Makemore 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Checkpoint loaded, device: {device}")

model = Makemore()

model,itos,stoi,block_size =  model.load_checkpoint('models/checkpoint.pt')

names = model.sample(itos, num_samples=50)
print(names)