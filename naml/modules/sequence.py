from naml.modules import torch, nn
from naml.sequence import zero_one_mask

class CELWithLengthMask(nn.CrossEntropyLoss):
    '''Cross Entropy Loss with Length Mask
```LOGITS                             TARGET
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
    def __init__(self, *args, **kwargs):
        super().__init__(reduction='none')        
    def forward(self, logits: torch.Tensor, target: torch.Tensor, lens: torch.Tensor):        
        loss = super().forward(logits.permute(0, 2, 1), target)
        # Therefore a permute/transpose is needed
        mask = zero_one_mask(loss.shape, lens)
        loss *= mask
        return loss.mean(dim=1)
    
class Seq2SeqEncoder(nn.Module):
    def __init__(self, n_vocab, n_embedding, n_hidden, n_layer, dropout_p):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, n_embedding)
        self.rnn = nn.GRU(n_embedding, n_hidden, n_layer, dropout=dropout_p)

    def forward(self, X: torch.Tensor):
        # X[batch_size, n_step, n_embedding]
        pass
