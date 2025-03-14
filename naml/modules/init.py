from naml.modules import torch, nn, F, Callable

def init_weights(m : nn.Module, init_fn : Callable) -> nn.Module:
    match m:
        case nn.Linear() | nn.Conv1d() | nn.Conv2d() | nn.Conv3d() | nn.Embedding():
            init_fn(m.weight)            
        case nn.GRU() | nn.LSTM():
            for name, param in m.named_parameters():
                if 'weight' in name:
                    init_fn(param)                
        case _:
            # raise NotImplementedError(f'init_weights for {m.__class__.__name__} is not implemented')
            pass
    return m

def init_weights_xavier_uniform(m : nn.Module):
    return init_weights(m, nn.init.xavier_uniform_)
