import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


class BiLSTMWrapper(nn.Module):
    """
    input size: (batch, seq_len, input_size)
    output size: (batch, seq_len, num_directions * hidden_size) last layer
    h_0: (num_layers * num_directions, batch, hidden_size)
    c_0: (num_layers * num_directions, batch, hidden_size)
    """
    def __init__(self, config, input_size, hidden_size):
        super(BiLSTMWrapper, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size  # TODO: verify the input size
        self.pdrop = config['dropout_lstm']
        self.n_layers = config['lstm_layers']
        self.bilstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, batch_first=True, bidirectional=True)
        self.h0 = None
        self.c0 = None

    def forward(self, X, X_mask, dropout=True, use_packing=False):
        if use_packing:
            N, max_len = X.shape[0], X.shape[-2]
            lens = torch.sum(X_mask, dim=-1)
            lens += torch.eq(lens, 0).long()  # Avoid length 0
            lens, indices = torch.sort(lens, descending=True)
            _, rev_indices = torch.sort(indices, descending=False)

            X = pack(X[indices], lens, batch_first=True)
            H, (h0, c0) = self.bilstm(X)
            H, _ = unpack(H, total_length=max_len, batch_first=True)

            # h0: [2, N, hidden_size]
            # H : [N, maxlen, hidden_size*2]
            H = H[rev_indices]
            h0 = torch.transpose(h0, 0, 1)
            h0 = h0[rev_indices].view(N, 2 * self.hidden_size)
            c0 = torch.transpose(c0, 0, 1)
            c0 = c0[rev_indices].view(N, 2 * self.hidden_size)
        else:
            H, (h0, c0) = self.bilstm(X)

        if X_mask is not None:
            H = H * X_mask.unsqueeze(-1).float()
        if dropout:
            H = F.dropout(H, self.pdrop, training=self.training)
        return H, (h0, c0)