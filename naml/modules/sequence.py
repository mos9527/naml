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
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers):
        super().__init__()
        self.vocab_size, self.embed_size, self.num_hiddens, self.num_layers = vocab_size, embed_size, num_hiddens, num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers)
        # self.dense = nn.Linear(num_hiddens, embed_size) 
        # Hidden states are used as is

    def forward(self, X : torch.Tensor, H : torch.Tensor):
        # X[batch_size, num_steps]
        X = self.embedding(X.T)        
        # X[num_steps, batch_size, embed_size]
        Y, H = self.rnn(X, H)
        # Y[num_steps, batch_size,num_hiddens], H[num_layers, batch_size, num_hiddens]
        return Y, H
    
    def begin_state(self, device : torch.device, batch_size : int):
        return torch.zeros((self.num_layers, batch_size, self.num_hiddens), device=device)
    
class Seq2SeqDecoder(nn.Module):    
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers):
        super().__init__()
        self.vocab_size, self.embed_size, self.num_hiddens, self.num_layers = vocab_size, embed_size, num_hiddens, num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers)
        # [Embedding | Hidden]
        self.dense = nn.Linear(num_hiddens, vocab_size) 

    def forward(self, X : torch.Tensor, H : torch.Tensor):
        # X[batch_size, num_steps]
        X = self.embedding(X.T)        
        # X[num_steps, batch_size, embed_size]
        C = H[-1].repeat(X.shape[0], 1, 1)
        # C[num_steps, batch_size, num_hiddens]
        XC = torch.cat((X, C), dim=2)        
        Y, H = self.rnn(XC, H)
        # Y[num_steps, batch_size,num_hiddens], H[num_layers, batch_size, num_hiddens]
        Y = self.dense(Y)
        # Y[num_steps, batch_size, vocab_size]
        Y : torch.Tensor = Y.permute(1, 0, 2)
        # Y[batch_size, num_steps, vocab_size]
        return Y, H
    
    def begin_state(self, device : torch.device, batch_size : int):
        return torch.zeros((self.num_layers, batch_size, self.num_hiddens), device=device)

class EncoderDecoder(nn.Module):
    """```
                                        X_Decoder
                                          |                                     
    X_Encoder -- Encoder --> E_Hidden -- Decoder --> Y_Output
    ```"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, X_enc : torch.Tensor, X_dec : torch.Tensor):
        Y_enc, H_enc = self.encoder(X_enc)        
        return self.decoder(X_dec, H_enc)