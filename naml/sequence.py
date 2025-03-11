from naml.modules import torch, nn, Generator, Tuple, Iterable

def zero_one_mask(
    size: Tuple[int,int], lens: torch.Tensor
):
    """Generate a 2D mask for a sequence of 0-1 values, where the elements beyond the respective lengths are zeroed."""    
    assert len(lens) == size[0]
    return torch.arange(size[1]).unsqueeze(0) < lens.unsqueeze(1)

def sequence_mask(
    X: torch.Tensor, lens: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Mask the elements of a sequence that are beyond the respective lengths by a value."""
    assert X.dim() == 2
    mask = zero_one_mask(X.shape, lens)
    X[~mask] = value
    return X

def softmax_mask(X: torch.Tensor, lens: torch.Tensor):
    """Apply softmax to a sequence and mask the elements that are beyond the respective lengths.
    Masked values are in effect set to zero.
    """
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

def seq_partition_sample_2D_random_iter(
    X: torch.Tensor, batch_size: int, n_step: int
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """Builds XY pairs where Y is the next element relative to X in the sequence.
    Sampled randomly (shuffled) from the sequence, with random start.

    Returns:
        Generator of [X[batch_size, n_step], Y[batch_size, n_step]] * len(X) // n_step / batch_size
    """
    X = X[torch.randint(0, n_step - 1, (1,)) :]
    n_subseq = (X.size(0) - 1) // n_step
    o = torch.arange(0, n_subseq) * n_step
    o = o[torch.randperm(n_subseq)]
    for i in range(0, n_subseq, batch_size):
        Xs = [X[o[i + j] : o[i + j] + n_step] for j in range(batch_size)]
        Ys = [X[o[i + j] + 1 : o[i + j] + n_step + 1] for j in range(batch_size)]
        yield torch.stack(Xs), torch.stack(Ys)


def seq_partition_sample_2D_sequential_iter(
    X: torch.Tensor, batch_size: int, n_step: int
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """Builds XY pairs where Y is the next element relative to X in the sequence.
    Sampled sequentially from the sequence, with random start.

    Returns:
        Generator of [X[batch_size, n_step], Y[batch_size, n_step]] * len(X) // n_step / batch_size
    """
    X = X[torch.randint(0, n_step, (1,)) :]
    n_tokens = X.size(0) - 1
    n_tokens -= n_tokens % batch_size
    Xs = X[:n_tokens].reshape(batch_size, -1)
    Ys = X[1 : n_tokens + 1].reshape(batch_size, -1)
    n_batch = Xs.size(1) // n_step
    for i in range(0, n_step * n_batch, n_step):
        yield torch.Tensor(Xs[:, i : i + n_step]), torch.Tensor(Ys[:, i : i + n_step])


def seq_partition_sample_1D_random_iter(
    X: torch.Tensor, batch_size: int, shuffle: bool = True
) -> Generator[torch.Tensor, None, None]:
    """Builds len(X)/batch_size partitions of a sequence in batch_size, optionally shuffled"""
    assert X.size(0) % batch_size == 0, "uneven batch size"
    n_subseq = X.size(0) // batch_size
    if shuffle:
        o = torch.randperm(n_subseq) * batch_size
    else:
        o = torch.arange(n_subseq) * batch_size
    for i in range(0, n_subseq):
        yield X[o[i] : o[i] + batch_size]


def seq_partition_sample_1D_sequential_iter(
    X: torch.Tensor, batch_size: int
) -> Generator[torch.Tensor, None, None]:
    """Builds len(X)/batch_size partitions of a sequence in batch_size, sequentially"""
    return seq_partition_sample_1D_random_iter(X, batch_size, shuffle=False)
