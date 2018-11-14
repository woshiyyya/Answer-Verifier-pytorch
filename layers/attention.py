import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
NEG_INF = -1e29


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2)) # lq * lk
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, NEG_INF)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


def mask_softmax(W, mask1, mask2):  # W: [N, lq, ls], mask: [N, ls]
    j_mask = joint_mask(mask1, mask2)
    W = W + (1 - j_mask.float()) * NEG_INF
    return F.softmax(W, dim=-1)


def joint_mask(mask1, mask2):
    mask1 = mask1.unsqueeze(-1)
    mask2 = mask2.unsqueeze(-2)
    mask = mask1 * mask2
    return mask


class BiAttentionBlock(nn.Module):
    def __init__(self, config, self_attention = False):
        super(BiAttentionBlock, self).__init__()
        self.config = config
        self.fusion = FusionBlock(config)
        self.attn_dropout = nn.Dropout(config['dropout_attn'])
        self.self_attention = self_attention

    def _biattn(self, S, Q):
        len_s = S.shape[-2]
        Qt = torch.transpose(Q, -1, -2)
        a = torch.matmul(S, Qt)  # [N, ls, lq]
        # a = self.attn_dropout(a)
        if self.self_attention:
            diag_mask = torch.eye(len_s).byte().unsqueeze(0)
            if self.config['cuda']:
                diag_mask = diag_mask.cuda()
            a.data.masked_fill_(diag_mask, -float('inf'))
        at = torch.transpose(a, -1, -2)
        b = torch.matmul(F.softmax(a, dim=-1), Q)
        c = torch.matmul(F.softmax(at, dim=-1), S)
        return b, c

    def mask_biattn(self, S, Q, S_mask, Q_mask):
        len_s = S.shape[-2]
        Qt = torch.transpose(Q, -1, -2)
        a = torch.matmul(S, Qt)  # [N, ls, lq]
        # a = self.attn_dropout(a)
        if self.self_attention:
            diag_mask = torch.eye(len_s).byte().unsqueeze(0)
            if self.config['cuda']:
                diag_mask = diag_mask.cuda()
            a.data.masked_fill_(diag_mask, -float('inf'))
        at = torch.transpose(a, -1, -2)
        b = torch.matmul(mask_softmax(a, S_mask, Q_mask), Q)
        c = torch.matmul(mask_softmax(at, Q_mask, S_mask), S)
        return b, c

    def forward(self, S, Q, S_mask, Q_mask):
        B, C = self.mask_biattn(S, Q, S_mask, Q_mask)
        S_tilde = self.fusion(S, B)
        if self.self_attention:
            return S_tilde
        else:
            Q_tilde = self.fusion(Q, C)
            return S_tilde, Q_tilde


class FusionBlock(nn.Module):
    def __init__(self, config):
        super(FusionBlock, self).__init__()
        self.linear_r = nn.Linear(8 * config['hidden_size'], 2 * config['hidden_size'])
        self.linear_g = nn.Linear(8 * config['hidden_size'], 2 * config['hidden_size'])
        self.activation = torch.sigmoid

    def forward(self, x, y):  # x: 2h
        input_r = torch.cat([x, y, x*y, x-y], dim=-1)
        r = gelu(self.linear_r(input_r))
        input_g = torch.cat([x, y, x*y, x-y], dim=-1)
        g = self.activation(self.linear_g(input_g))
        o = g*r + (1-g)*x
        return o

