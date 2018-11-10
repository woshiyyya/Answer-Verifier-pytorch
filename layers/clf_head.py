import torch
import torch.nn as nn

class Clf_head(nn.Module):
    def __init__(self, config, dim_in):
        super(Clf_head, self).__init__()
        self.config = config
        self.ls = config['doc_maxlen']
        self.lq = config['query_maxlen']
        self.N = self.config['batch_size']
        self.dim_in = dim_in
        self.linear1 = nn.Linear(dim_in, 10)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(10 * (self.ls + self.lq), 2)

    def forward(self, S, Q):
        S = self.activation(self.linear1(S)).view(self.N, -1)
        Q = self.activation(self.linear1(Q)).view(self.N, -1)
        X = torch.cat([S, Q], dim=-1)
        return self.linear2(X)