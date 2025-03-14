from naml.modules import nn

def grad_clip(m : nn.Module, theta : float) -> nn.Module:
    return nn.utils.clip_grad.clip_grad_norm_(m.parameters(), theta)