from collections import Counter
from naml.modules import List, Dict, Tuple


class Vocabulary(dict):
    reserved = ["<unk>"]
    ivocab: List[str]  # index -> word, ordered by frequency

    @staticmethod
    def tokenize(lines: List[str]) -> List[List[str]]:
        return [[token for token in line.split()] for line in lines]

    @staticmethod
    def tokenize_char(lines: List[str]) -> List[List[str]]:
        return [[token for token in line] for line in lines]

    @staticmethod
    def to_corpus(tokens: List[List[str]]) -> List[str]:
        return [token for line in tokens for token in line]

    def __init__(self, corpus: List[str | List[str]]):
        counter = Counter(corpus)
        self.ivocab = []
        items = counter.most_common()
        self.clear()
        self.update({word: (i, 0) for i, word in enumerate(self.reserved)})
        self.update(
            {
                word: (i + len(self.reserved), count)
                for i, (word, count) in enumerate(items)
            }
        )
        self.ivocab += self.reserved
        self.ivocab += [word for word, count in items]

    @property
    def unqiue_tokens(self):
        return list(self.keys())[len(self.reserved) :]

    def freqs(self, tokens: List[str]):
        return [self[token][1] for token in tokens]

    def to_indices(self, tokens: List[str]):
        return [self[token][0] for token in tokens]

    def to_tokens(self, indices: List[int]):
        return [self.ivocab[index] for index in indices]
