from naml.modules import torch, nn, Generator, Tuple


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


def seq_partition_sample_random(
    X: torch.Tensor, batch_size: int, n_step: int
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    assert X.size(0) >= 2 * batch_size * n_step
    X = X[torch.randint(0, n_step - 1, (1,)) :]
    n_subseq = (X.size(0) - 1) // n_step
    o = torch.arange(0, n_subseq) * n_step
    o = o[torch.randperm(n_subseq)]
    for i in range(0, n_subseq, batch_size):
        Xs = [X[o[i + j] : o[i + j] + n_step] for j in range(batch_size)]
        Ys = [X[o[i + j] + 1 : o[i + j] + n_step + 1] for j in range(batch_size)]
        yield torch.stack(Xs), torch.stack(Ys)


def seq_partition_sample_sequential(
    X: torch.Tensor, batch_size: int, n_step: int
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    X = X[torch.randint(0, n_step, (1,)) :]
    n_tokens = X.size(0) - 1
    n_tokens -= n_tokens % batch_size
    Xs = X[:n_tokens].reshape(batch_size, -1)
    Ys = X[1 : n_tokens + 1].reshape(batch_size, -1)
    n_batch = Xs.size(1) // n_step
    for i in range(0, n_step * n_batch, n_step):
        yield torch.Tensor(Xs[:, i : i + n_step]), torch.Tensor(Ys[:, i : i + n_step])
