from naml.modules import torch, nn


class Seq2SeqEncoder(nn.Module):
    def __init__(self, n_vocab, n_embedding, n_hidden, n_layer, dropout_p):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, n_embedding)
        self.rnn = nn.GRU(n_embedding, n_hidden, n_layer, dropout=dropout_p)

    def forward(self, X : torch.Tensor):
        # X[batch_size, n_step, n_embedding]
        