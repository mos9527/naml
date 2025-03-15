from naml.modules import torch, nn
from naml.sequence import softmax_mask


class AdditiveAttention(nn.Module):
    M_w: torch.Tensor

    def __init__(self, n_key, n_query, n_hidden, dropout_p):
        super().__init__()
        self.W_k = nn.Linear(n_key, n_hidden, bias=False)
        self.W_q = nn.Linear(n_query, n_hidden, bias=False)
        self.w_v = nn.Linear(n_hidden, 1, bias=False)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lens: torch.Tensor = None
    ):
        # q[batch_size, n_query, n_hidden], k[batch_size, n_key, n_hidden]
        q, k = self.W_q(q), self.W_k(k)
        # q[batch_size, n_query, 1,     n_hidden]
        # k[batch_size, 1,       n_key, n_hidden]
        #               ^^^^^^^^ ^^^^^^ These would be broadcasted
        features = q.unsqueeze(2) + k.unsqueeze(1)
        # f[batch_size, n_query, n_key, n_hidden]
        scores = self.w_v(torch.tanh(features))
        # s[batch_size, n_query, n_key, 1]
        scores = scores.squeeze(-1)
        # s[batch_size, n_query, n_key]
        self.M_w = M_w = softmax_mask(scores, lens) if lens is not None else torch.softmax(scores, dim=-1)
        return self.dropout(M_w) @ v


class DotProductAttention(nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lens: torch.Tensor = None
    ):
        # q[batch_size, n_query, n_hidden], k[batch_size, n_key, n_hidden]
        assert q.shape[-1] == k.shape[-1]
        d = torch.Tensor([q.shape[-1]]).float()
        scores = (q @ k.transpose(1, 2)) / torch.sqrt(d)
        self.M_w = M_w = softmax_mask(scores, lens) if lens is not None else torch.softmax(scores, dim=-1)
        return self.dropout(M_w) @ v
