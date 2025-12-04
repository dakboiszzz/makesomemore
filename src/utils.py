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
        'itos': data_prep.itos,
        'stoi': data_prep.stoi
    }
    for i,layer in enumerate(model.layers):
        for j, p in enumerate(layer.parameters()):
            checkpoint['model_state'][f'layer_{i}_param{j}'] = p.data
    torch.save(checkpoint,filepath)
    print(f'Checkpoint saved to {filepath}')


