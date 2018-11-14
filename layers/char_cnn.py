import torch
import torch.nn as nn
import torch.nn.functional as F
# My implementation of char_cnn
'''
    nn.Conv2D(in_channel, out_channel, kernel_size)
    input: (N, in_channel, H_in, W_in)
    output: (N, out_channel, H_out, W_out)
'''


class Char_CNN(nn.Module):
    def __init__(self, config):
        super(Char_CNN, self).__init__()
        self.c_outs = [50, 50, 50]
        self.outsize = sum(self.c_outs)
        self.K_sizes = [3, 4, 5]
        self.N = config['batch_size']
        self.lw = config['word_maxlen']
        self.emb_dim = config['char_emb_size']
        self.convs = nn.ModuleList([nn.Conv2d(1, c_out, (K_size, self.emb_dim)) for c_out, K_size in zip(self.c_outs, self.K_sizes)])
        self.activation = nn.ReLU()
        self.pdrop = config['dropout_cnn']
        self.dropout = F.dropout

    def forward(self, X):                                           # [N, sent_len, word_length, char_emd_dim]
        X = X.view([-1, self.lw, self.emb_dim])
        X = X.unsqueeze(1)                                          # [N*ls, 1, lw, emb]
        X = [self.activation(conv(X)).squeeze(3) for conv in self.convs]    # [N*ls, c_out, H_out] * k_num
        X = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in X]      # [N*ls, c_out] * k_num
        X = torch.cat(X, dim=-1)  # [N * ls, 300]
        X = X.view(self.N, -1, sum(self.c_outs))
        X = self.dropout(X, p=self.pdrop)
        return X


class CNN_Text(nn.Module):
    def __init__(self, config, h_dim):
        super(CNN_Text, self).__init__()
        self.config = config
        self.h_dim = h_dim
        self.c_in = 1
        self.c_out = 20
        self.K_sizes = [3, 4, 5]
        self.convs = nn.ModuleList([nn.Conv2d(self.c_in, self.c_out, (K_size, h_dim)) for K_size in self.K_sizes])
        self.dropout = nn.Dropout(config['dropout_cnn'])
        self.activation = nn.ReLU()
        self.linear = nn.Linear(len(self.K_sizes), 2)

    def forward(self, X):
        X = X.unsqueeze(1) # [N, 1, max_len, h_dim]
        X = [self.activation(conv(X)).squeeze(3) for conv in self.convs] # [N, c_out, H_out, 1] * k_num
        X = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in X] # [N, c_out] * k_num
        X = torch.cat(X)
        X = self.dropout(X)
        logit = self.linear(X)
        return logit


class HighwayMLP(nn.Module):
    def __init__(self,
                 input_size,
                 gate_bias=-2):

        super(HighwayMLP, self).__init__()

        self.activation_function = nn.ReLU()
        self.gate_activation = F.softmax

        self.normal_layer = nn.Linear(input_size, input_size)

        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):

        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x), dim=-1)

        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)

        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)