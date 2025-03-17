from naml.modules import torch, nn, List, Tuple
import math

class FFN(nn.Module):
    # Feed Forward Network
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.W1 = nn.Linear(n_in, n_hidden)
        self.relu = nn.ReLU()
        self.W2 = nn.Linear(n_hidden, n_out)

    def forward(self, X: torch.Tensor):
        return self.W2(self.relu(self.W1(X)))
    
class AddAndNorm(nn.Module):
    # Add and Normalize
    def __init__(self, norm_shape, dropout_p):
        super().__init__()
        self.norm = nn.LayerNorm(norm_shape)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        return self.norm(X + self.dropout(Y))
    
from naml.modules.attention import MultiheadAttention
class EncoderBlock(nn.Module):
    def __init__(
            self, n_keys, n_query, n_values, n_hidden,
            norm_shape, 
            ffn_n_in, ffn_n_hidden,
            n_heads, dropout_p):
        super().__init__()
        assert n_keys == n_query == n_values, "unsupported qkv shapes"
        self.attn = MultiheadAttention(n_keys, n_query, n_values, n_hidden, n_heads, dropout_p)
        self.add_norm1 = AddAndNorm(norm_shape, dropout_p)
        self.ffn = FFN(ffn_n_in, ffn_n_hidden, n_hidden)
        self.add_norm2 = AddAndNorm(norm_shape, dropout_p)
    
    def forward(self, X: torch.Tensor, lens : torch.Tensor) -> torch.Tensor:
        # X[batch_size, n_keys, n_hidden], lens[batch_size]
        Y = self.add_norm1(X, self.attn.forward(X, X, X, lens))        
        return self.add_norm2(Y, self.ffn(Y))
    
from naml.modules.encoding import SinusoidalPositionalEncoding
class Encoder(nn.Module):
    def __init__(self, vocab_size,
        n_keys, n_query, n_values, n_hidden,
        norm_shape, 
        ffn_n_in, ffn_n_hidden,
        n_heads, dropout_p,
        n_blocks,
        pos_encoder=SinusoidalPositionalEncoding,
        pos_maxlen=1000
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_hidden)
        self.encoding = pos_encoder(n_hidden, pos_maxlen, dropout_p)
        self.blocks = nn.Sequential(*[
            EncoderBlock(n_keys, n_query, n_values, n_hidden, norm_shape, ffn_n_in, ffn_n_hidden, n_heads, dropout_p)
            for _ in range(n_blocks)
        ])
        self.num_hiddens = n_hidden

    def forward(self, X: torch.Tensor, lens: torch.Tensor = None) -> torch.Tensor:
        # X[batch_size, seq_len], lens[batch_size]
        X = self.embedding.forward(X) * math.sqrt(self.num_hiddens)
        X = self.encoding.forward(X)
        for block in self.blocks:
            block : EncoderBlock
            X = block.forward(X, lens)
        return X, [None] * len(self.blocks)
    
class DecoderBlock(nn.Module):
    def __init__(self, n_keys, n_query, n_values, n_hidden,
                 norm_shape, ffn_n_in, ffn_n_hidden,
                 n_heads, dropout_p):
        super().__init__()
        self.attn1 = MultiheadAttention(n_keys, n_query, n_values, n_hidden, n_heads, dropout_p)
        self.add_norm1 = AddAndNorm(norm_shape, dropout_p)
        self.attn2 = MultiheadAttention(n_keys, n_query, n_values, n_hidden, n_heads, dropout_p)
        self.add_norm2 = AddAndNorm(norm_shape, dropout_p)
        self.ffn = FFN(ffn_n_in, ffn_n_hidden, n_hidden)
        self.add_norm3 = AddAndNorm(norm_shape, dropout_p)

    def forward(self, X: torch.Tensor, Y_enc : torch.Tensor, len_enc : torch.Tensor, K : torch.Tensor = None) -> Tuple[torch.Tensor,torch.Tensor]:
        # X[batch_size, n_keys, n_hidden], H_enc[batch_size, n_keys, n_hidden], len_enc[batch_size]
        if K is None:
            K = X
        else:
            K = torch.cat((K, X), dim=1)   
        len_dec = None
        if self.training:
            batch_size, num_steps = X.shape[:2]
            len_dec = torch.arange(1, num_steps + 1).repeat(batch_size, 1)
        X2 = self.attn1.forward(X, K, K, len_dec)
        Y = self.add_norm1.forward(X, X2)
        Y2 = self.attn2.forward(Y, Y_enc, Y_enc, len_enc)
        Y = self.add_norm2.forward(Y, Y2)
        return self.add_norm3.forward(Y, self.ffn.forward(Y)), K

class Decoder(nn.Module):
    def __init__(self, vocab_size,
                 n_keys, n_query, n_values, n_hidden,
                 norm_shape, 
                 ffn_n_in, ffn_n_hidden,
                 n_heads, dropout_p,
                 n_blocks,
                 pos_encoder=SinusoidalPositionalEncoding,
                 pos_maxlen=1000
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_hidden)
        self.encoding = pos_encoder(n_hidden, pos_maxlen, dropout_p)
        self.blocks = nn.Sequential(*[
            DecoderBlock(n_keys, n_query, n_values, n_hidden, norm_shape, ffn_n_in, ffn_n_hidden, n_heads, dropout_p)
            for _ in range(n_blocks)
        ])
        self.dense = nn.Linear(n_hidden, vocab_size)
        self.num_hiddens = n_hidden

    def forward(self, X: torch.Tensor, K_states : List[torch.Tensor], Y_enc : torch.Tensor, len_enc : torch.Tensor=None) -> torch.Tensor:
        # X[batch_size, seq_len], Y_enc[batch_size, seq_len, n_hidden], len_enc[batch_size]
        X = self.embedding.forward(X) * math.sqrt(self.num_hiddens)
        X = self.encoding.forward(X)        
        for i, block in enumerate(self.blocks):
            block : DecoderBlock            
            X, K_states[i] = block.forward(X, Y_enc, len_enc, K_states[i])        
        return self.dense.forward(X), K_states
    
class EncoderDecoder(nn.Module):
    """```
                                        X_Decoder
                                          |                                     
    X_Encoder -- Encoder --> E_Hidden -- Decoder --> Y_Output
    ```"""
    def __init__(self, encoder : Encoder, decoder : Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, X_enc : torch.Tensor, X_dec : torch.Tensor, lens_enc):
        '''X_enc[batch_size, num_steps], X_dec[batch_size, num_steps]'''
        Y_enc, K_states = self.encoder(X_enc, lens_enc)        
        return self.decoder(X_dec,K_states,Y_enc,lens_enc)