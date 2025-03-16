from naml.modules import torch, nn
from naml.sequence import softmax_mask


class AdditiveAttention(nn.Module):
    M_w: torch.Tensor

    def __init__(self, n_key, n_query, n_hidden, dropout_p):
        super().__init__()
        self.W_k = nn.Linear(n_key, n_hidden, bias=False)
        self.W_q = nn.Linear(n_query, n_hidden, bias=False)
        self.W_v = nn.Linear(n_hidden, 1, bias=False)
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
        scores = self.W_v(torch.tanh(features))
        # s[batch_size, n_query, n_key, 1]
        scores = scores.squeeze(-1)
        # s[batch_size, n_query, n_key]
        self.M_w = M_w = softmax_mask(scores, lens) if lens is not None else torch.softmax(scores, dim=-1)
        return self.dropout(M_w) @ v


class DotProductAttention(nn.Module):
    def __init__(self, n_key, n_query, n_hidden, dropout_p):
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

class MultiheadAttention(nn.Module):
    def __init__(self, n_key, n_query, n_hidden, n_heads, dropout_p, attn_class = DotProductAttention):
        super().__init__()
        self.W_q = nn.Linear(n_query, n_hidden, bias=False)
        self.W_k = nn.Linear(n_key, n_hidden, bias=False)
        self.W_v = nn.Linear(n_key, n_hidden, bias=False)
        self.W_o = nn.Linear(n_hidden, n_hidden, bias=False)        
        self.num_heads = n_heads
        self.head_size = n_hidden // n_heads
        self.attn = attn_class(self.head_size, self.head_size, self.head_size, dropout_p)

    def transpose_hidden(self, X: torch.Tensor):
        # X[batch_size, n_query;n_key(value) , n_hidden]
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, self.head_size)
        # X[batch_size, n_query;n_key(value), n_heads, head_size]
        X = X.permute(0, 2, 1, 3)
        # X[batch_size, n_heads, n_query;n_key(value), head_size]
        # X[batch_size * n_heads, n_query;n_key(value), head_size]
        return X.reshape(-1, X.shape[2], X.shape[3])     
    
    def transpose_output(self, X: torch.Tensor):
        # X[batch_size * n_heads, n_query;n_key(value), head_size]
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        # X[batch_size, n_heads, n_query;n_key(value), head_size]
        X = X.permute(0, 2, 1, 3)
        # X[batch_size, n_query;n_key(value), n_heads, head_size]
        # X[batch_size, n_query;n_key(value), n_hidden]
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lens: torch.Tensor = None):
        q, k, v = self.transpose_hidden(self.W_q(q)), self.transpose_hidden(self.W_k(k)), self.transpose_hidden(self.W_v(v))
        # q[batch_size * n_heads, n_query, head_size], k[batch_size * n_heads, n_key, head_size], v[batch_size * n_heads, n_key, head_size]
        if lens is not None:
            lens = lens.repeat_interleave(self.num_heads, dim=0)
        self.M_w = M_w = self.attn(q, k, v, lens)
        # M_w[batch_size * n_heads, n_query, head_size]
        M_w = self.transpose_output(M_w)
        # M_w[batch_size, n_query, n_hidden]
        return self.W_o(M_w)
