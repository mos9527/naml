from naml.modules import torch, nn

class SinusoidalPositionalEncoding(nn.Module):
    P : torch.Tensor # P[1, num_steps, embed_size]

    def __init__(self, embed_size, num_steps, dropout_p):
        super().__init__()
        self.embed_size, self.max_len = embed_size, num_steps        

        self.P = torch.zeros((1, num_steps, embed_size))
        # Extra dimension is for batch boardcasting
        self.dropout = nn.Dropout(dropout_p)

        X = torch.arange(0, num_steps).reshape(-1, 1) / torch.pow(10000, torch.arange(0, embed_size, 2) / embed_size)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
        self.P.requires_grad = False

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        '''X[batch_size, num_steps, embed_size]'''
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
