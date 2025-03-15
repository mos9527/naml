from naml.modules import torch, nn
from naml.sequence import zero_one_mask

class CELWithLengthMask(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        '''Cross Entropy Loss with Length Mask

Shapes:
```logits                             target
[ # batch n                        [ # batch n               
    [ # step m                         [ # step m            
        [ logits for 0 ... k ]             index for 0 ... n 
    ]                                  ]                     
]                                  ]     
CrossEntropyLoss expects           Returns with reduction=none
[ # batch n                        [ # batch n                    
    [ # class m                        [ # step m                 
        [ # dimension k                    loss for 0 ... m       
            logit at C[m,k]            ]                          
        ]                          ]                              
    ]                                                        
]'''    
        super().__init__(reduction='none')        
    def forward(self, logits: torch.Tensor, target: torch.Tensor, lens: torch.Tensor):        
        loss = super().forward(logits.permute(0, 2, 1), target)
        # Therefore a permute/transpose is needed
        mask = zero_one_mask(loss.shape, lens)
        loss *= mask
        return loss.mean(dim=1)
    
class Seq2SeqEncoder(nn.Module):
    # GRU for implementation
    # This is a slightly modified version of RNN from the one from Chapter 8
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout_p):
        super().__init__()
        self.vocab_size, self.embed_size, self.num_hiddens, self.num_layers, self.dropout_p = vocab_size, embed_size, num_hiddens, num_layers, dropout_p
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout_p)
        # self.dense = nn.Linear(num_hiddens, embed_size) 
        # Hidden states are used as is

    def forward(self, X : torch.Tensor):
        # X[batch_size, num_steps]
        X = self.embedding(X.T)        
        # X[num_steps, batch_size, embed_size]
        Y, H = self.rnn(X)
        # Y[num_steps, batch_size,num_hiddens], H[num_layers, batch_size, num_hiddens]
        return Y, H
    
    def begin_state(self, device : torch.device, batch_size : int):
        return torch.zeros((self.num_layers, batch_size, self.num_hiddens), device=device)
    
class Seq2SeqDecoder(nn.Module):    
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout_p):
        super().__init__()
        self.vocab_size, self.embed_size, self.num_hiddens, self.num_layers, self.dropout_p = vocab_size, embed_size, num_hiddens, num_layers, dropout_p
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,dropout=dropout_p)
        # [Embedding | Hidden]
        self.dense = nn.Linear(num_hiddens, vocab_size) 

    def forward(self, X : torch.Tensor, H_enc : torch.Tensor, *args):
        # X[batch_size, num_steps]
        X = self.embedding(X.T)        
        # X[num_steps, batch_size, embed_size]
        C = H_enc[-1].repeat(X.shape[0], 1, 1)
        # C[num_steps, batch_size, num_hiddens]
        XC = torch.cat((X, C), dim=2)        
        Y, H_enc = self.rnn(XC, H_enc)
        # Y[num_steps, batch_size,num_hiddens], H[num_layers, batch_size, num_hiddens]
        Y = self.dense(Y)
        # Y[num_steps, batch_size, vocab_size]
        Y : torch.Tensor = Y.permute(1, 0, 2)
        # Y[batch_size, num_steps, vocab_size]
        return Y, H_enc
    
    def begin_state(self, device : torch.device, batch_size : int):
        return torch.zeros((self.num_layers, batch_size, self.num_hiddens), device=device)

from naml.modules.attention import AdditiveAttention
class Seq2SeqAttentionDecoder(Seq2SeqDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout_p):
        super().__init__(vocab_size, embed_size, num_hiddens, num_layers, dropout_p)
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout_p)


    def forward(self, X : torch.Tensor, H_enc : torch.Tensor, Y_enc : torch.Tensor, lens_enc : torch.Tensor = None):
        # X[batch_size, num_steps]        
        X = self.embedding(X.T)                
        # X[num_steps, batch_size, embed_size]
        outputs = []
        for step in X:
            q = H_enc[-1].unsqueeze(1)
            # q[batch_size, 1, num_hiddens]
            k = v = Y_enc.permute(1,0,2)
            # k[batch_size, num_steps, num_hiddens]
            C = self.attention.forward(q, k, v, lens_enc)
            # C[num_steps, batch_size, num_hiddens]
            XC = torch.cat((C, step.unsqueeze(1)), dim=-1)
            XC = XC.permute(1, 0, 2)
            # XC[1, batch_size, num_hiddens + embed_size]
            Y, H_enc = self.rnn(XC, H_enc)
            # Y[1, batch_size,num_hiddens], H[num_layers, batch_size, num_hiddens]
            outputs.append(Y)
        Y = self.dense(torch.cat(outputs, dim=0))
        # Y[num_steps, batch_size, vocab_size]
        Y : torch.Tensor = Y.permute(1, 0, 2)
        # Y[batch_size, num_steps, vocab_size]
        return Y, H_enc
    
            
class EncoderDecoder(nn.Module):
    """```
                                        X_Decoder
                                          |                                     
    X_Encoder -- Encoder --> E_Hidden -- Decoder --> Y_Output
    ```"""
    def __init__(self, encoder : Seq2SeqEncoder, decoder : Seq2SeqDecoder | Seq2SeqAttentionDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, X_enc : torch.Tensor, X_dec : torch.Tensor, *decoder_args):
        Y_enc, H_enc = self.encoder(X_enc)        
        return self.decoder(X_dec, H_enc, Y_enc, *decoder_args)
    

from naml.modules import List
from naml.text import Vocabulary, pair_vocab_batch_sample_iter
from naml.util import run_epochs, ret_accumlated_loss
from naml.modules.init import init_weights_xavier_uniform
from naml.modules.utils import grad_clip
from naml.modules import torch

def train_seq2seq(
        net : EncoderDecoder, lr, epochs,
        src_vocab : Vocabulary, tgt_vocab : Vocabulary, 
        src_words : List[List[str]], tgt_words : List[List[str]],
        batch_size: int, num_steps: int
    ):
        net.train() # Train mode
        net.apply(init_weights_xavier_uniform)
        optim = torch.optim.Adam(net.parameters(), lr=lr) # XXX: Different optimizer?
        loss = CELWithLengthMask()
        data_iter = list(pair_vocab_batch_sample_iter(src_vocab, tgt_vocab, src_words, tgt_words,batch_size,num_steps))
        @run_epochs("Loss")
        def run_epoch():
            ret_loss = ret_accumlated_loss()
            optim.zero_grad()
            for x, x_len, y, y_len in data_iter:
                bos = torch.Tensor(tgt_vocab.to_indices(["<bos>"] * y.shape[0])).long().reshape(-1,1)
                y_in = torch.cat([bos, y[:, :-1]], 1) # XXX: Can we do this in data_iter instead?
                y_hat, _ = net.forward(x, y_in, x_len)
                l = loss.forward(y_hat, y, y_len)
                l.sum().backward()            
                grad_clip(net, 1)        
                optim.step()
                ret_loss.update(l.sum().detach(), y_len.sum())
            return ret_loss
        run_epoch(epochs)

def predict_seq2seq(net : EncoderDecoder, src_indices : torch.Tensor, tgt_vocab : Vocabulary, num_steps : int):
    net.eval()
    X_enc = src_indices.unsqueeze(0) # Only one batch -> [1, num_steps]
    Y_enc, H_enc = net.encoder(X_enc)
    H = H_enc

    X_dec = torch.Tensor(tgt_vocab.to_indices(["<bos>"])).long().unsqueeze(0)    
    for step in range(num_steps):
        Y_hat, H = net.decoder.forward(X_dec, H, Y_enc)
        X_dec = Y_hat.argmax(dim=2)
        pred = X_dec.squeeze(0).long().item()
        yield tgt_vocab.to_tokens([pred])[0]
