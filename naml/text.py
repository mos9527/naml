from collections import Counter
from naml.modules import List, Dict, Tuple, Set, Generator, torch, F


def replace_multiple(s: str, src: List[str], to: str) -> str:
    """Replace multiple patterns in a string with a single pattern."""
    for i in src:
        s = s.replace(i, to)
    return s


def tokenize_line(
    s: str, keep_sep: Set[str] | str, remove_sep: Set[str] | str
) -> Generator[str, None, None]:
    """O(kn) in space time helper function to split a string by multiple separators.

    Args:
        s: the string to split
        keep_sep: the set of separators to keep, will be kept as separate tokens
        remove_sep: the set of separators to remove

    Returns:
        a generator of tokens
    """
    keep_sep, remove_sep = set(keep_sep), set(remove_sep)
    seps = keep_sep | remove_sep
    assert all(len(sep) == 1 for sep in seps), "seps must be single characters"
    pfx = [i for sep in seps for i, c in enumerate(s) if c == sep]
    pfx.sort()
    prev = 0
    for i, cur in enumerate(pfx):
        sep = s[cur]
        if prev < cur:
            yield s[prev:cur]
        if sep in keep_sep:
            yield s[cur]
        prev = cur + 1
    if prev < len(s):
        yield s[prev:]


def tokenize(
    lines: List[str], keep_sep: Set[str] | str = "", remove_sep: Set[str] | str = " "
) -> List[List[str]]:
    """Tokenize a list of lines into a list of list of tokens.
    See `tokenize_line` for the separator options."""
    assert type(lines[0]) == str
    return [
        [token for token in tokenize_line(line, keep_sep, remove_sep)] for line in lines
    ]


def tokenize_char(lines: List[str]) -> List[List[str]]:
    """Tokenize a list of lines into a list of list of characters."""
    assert type(lines[0]) == str
    return [[token for token in line] for line in lines]


def flatten(tokens: List[List[str]]) -> List[str]:
    """Flatten a list of list of tokens into a list of tokens."""
    assert type(tokens[0]) == list
    return [token for line in tokens for token in line]


class Vocabulary(dict):
    reserved: List[str] = ["<unk>"]
    ivocab: List[str]  # index -> word, ordered by frequency

    def __init__(
        self,
        corpus: List[str | List[str]],
        min_freq: float = 0,
        reserved: List[str] = ["<unk>", "<pad>", "<eos>"],
    ):
        self.reserved = reserved
        counter = Counter(corpus)
        self.ivocab = []
        items = counter.most_common()
        self.clear()
        self.update({word: (i, 0) for i, word in enumerate(self.reserved)})
        self.update(
            {
                word: (i + len(self.reserved), count)
                for i, (word, count) in enumerate(items)
                if count >= min_freq
            }
        )
        self.ivocab += self.reserved
        self.ivocab += [word for word, count in items]

    @property
    def top_tokens(self) -> List[str]:
        return list(self.keys())[len(self.reserved) :]

    def freqs(self, tokens: List[str]) -> List[int]:
        return [self[token][1] for token in tokens]

    def to_indices(self, tokens: List[str]) -> torch.Tensor:
        return torch.Tensor([self[token][0] for token in tokens]).long()

    def to_tokens(self, indices: torch.Tensor) -> List[str]:
        return [self.ivocab[index] for index in indices]

    def truncate_pad(
        self, indices: torch.Tensor, n_steps: int, pad_token: str = "<pad>"
    ) -> torch.Tensor:
        pad_index = self[pad_token][0]
        return F.pad(indices, (0, n_steps), value=pad_index)[:n_steps].long()

    def to_indices_padded(
        self, lines: List[List[str]], n_steps: int, pad_token: str = "<pad>"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result = torch.stack(
            [
                self.truncate_pad(self.to_indices(line), n_steps, pad_token)
                for line in lines
            ]
        )
        lens = (result != self[pad_token][0]).sum(dim=1)
        return result.long(), lens.long()


from naml.sequence import seq_partition_sample_1D_sequential_iter


def pair_vocab_batch_sample_iter(
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    src_words: List[List[str]],
    target_words: List[List[str]],
    batch_size: int,
    num_steps: int,
) -> Generator[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None
]:
    """Builds XY pairs where Y is the paried element relative to X in the vocabularies.

    Returns:
        Generator of [X[batch_size, num_steps], X_len[batch_size], Y[batch_size, num_steps], Y_len[batch_size]]
    """
    X, X_len = src_vocab.to_indices_padded(src_words, num_steps)
    Y, Y_len = tgt_vocab.to_indices_padded(target_words, num_steps)
    for x, x_len, y, y_len in zip(
        seq_partition_sample_1D_sequential_iter(X, batch_size),
        seq_partition_sample_1D_sequential_iter(X_len, batch_size),
        seq_partition_sample_1D_sequential_iter(Y, batch_size),
        seq_partition_sample_1D_sequential_iter(Y_len, batch_size),
    ):
        yield x, x_len, y, y_len
