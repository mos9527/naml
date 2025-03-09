from naml.modules import torch, nn


def sequence_mask(
    X: torch.Tensor, lens: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    assert X.dim() == 2
    mask = torch.arange(X.size(1)).unsqueeze(0) < lens.unsqueeze(1)
    X[~mask] = value
    return X


def softmax_mask(X: torch.Tensor, lens: torch.Tensor):
    shape = X.shape
    X = X.reshape(-1, shape[-1])
    if lens.dim() == 1:
        lens = lens.repeat_interleave(shape[1])
    else:
        assert lens.shape == shape[:2]
        lens = lens.reshape(-1)
    X = sequence_mask(X, lens, -1e6)
    X = nn.functional.softmax(X.reshape(shape), dim=-1)
    return X
