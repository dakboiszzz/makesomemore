import torch
import torch.nn.functional as F

def save_checkpoint(model,data_prep,filepath):
    checkpoint = {
        'model_state' : {},
        'vocab_size': model.vocab_size,
        'block_size': model.block_size,
        'emb_size': model.emb_size,
        'n_hidden': model.n_hidden,
        'n_blocks': model.n_blocks,
        'generator': model.generator,
        'itos': data_prep.itos,
        'stoi': data_prep.stoi
    }
    for i,layer in enumerate(model.layers):
        for j, p in enumerate(layer.parameters()):
            checkpoint['model_state'][f'layer_{i}_param{j}'] = p.data
    torch.save(checkpoint,filepath)
    print(f'Checkpoint saved to {filepath}')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    from src.model import Makemore
    # Reconstruct the model
    model = Makemore(
        vocab_size = checkpoint['vocab_size'],
        block_size = checkpoint['block_size'],
        emb_size = checkpoint['emb_size'],
        n_hidden = checkpoint['n_hidden'],
        n_blocks = checkpoint['n_blocks'],
        generator = checkpoint['generator']
    )
    # Load parameters
    for i, layer in enumerate(model.layers):
        for j, p in enumerate(layer.parameters()):
            key = f'layer_{i}_param_{j}'
            if key in checkpoint['model_state']:
                p.data = checkpoint['model_state'][key]
    return model, checkpoint['itos'], checkpoint['stoi'], checkpoint['block_size']