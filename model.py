import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.rnn import BiLSTMWrapper
from layers.attention import MultiHeadAttention, BiAttentionBlock
from layers.encoder import LexiconEncoder, EncoderLayer
NEG_INF = -1e29


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def mean_pooling(x, x_mask):
    mask = x_mask.unsqueeze(-1).float()
    length = torch.sum(x_mask.float(), dim=-1) # [N, ld]
    result = torch.sum(x * mask, dim=-2) / length.unsqueeze(-1)
    return result


def max_pooling(x, x_mask):
    mask = x_mask.unsqueeze(-1).float()
    x = x * mask
    result = F.max_pool2d(x, kernel_size=(x.shape[-2], 1)).squeeze(-2)
    return result


class VerifierModel(nn.Module):
    def __init__(self, config, embedding=None):
        super(VerifierModel, self).__init__()
        self.lexicon_encoder = \
            LexiconEncoder(config, embedding=embedding, use_char_emb=config['use_char_emb'], use_char_rnn=config['use_char_rnn'])
        encoded_size = self.lexicon_encoder.output_size
        self.lstm_emb = BiLSTMWrapper(config, encoded_size, config['hidden_size'])

        self.encoder_layer = EncoderLayer(config, embedding)
        attn_dim = config['hidden_size'] * 2
        self.attention_block_s0 = MultiHeadAttention(n_head=1, d_model=attn_dim, d_k=attn_dim, d_v=attn_dim)
        self.attention_block_q0 = MultiHeadAttention(n_head=1, d_model=attn_dim, d_k=attn_dim, d_v=attn_dim)
        self.lstm_attn0 = BiLSTMWrapper(config, config['hidden_size'] * 2, config['hidden_size'])

        self.biattention_block = BiAttentionBlock(config, self_attention=False)
        self.lstm_biattn_s = BiLSTMWrapper(config, config['hidden_size']*2, config['hidden_size'])
        self.lstm_biattn_q = BiLSTMWrapper(config, config['hidden_size'] * 2, config['hidden_size'])

        self.selfattention_block_s = BiAttentionBlock(config, self_attention=True)
        self.selfattention_block_q = BiAttentionBlock(config, self_attention=True)
        self.attention_block_s1 = MultiHeadAttention(n_head=1, d_model=attn_dim, d_k=attn_dim, d_v=attn_dim)
        self.attention_block_q1 = MultiHeadAttention(n_head=1, d_model=attn_dim, d_k=attn_dim, d_v=attn_dim)

        self.lstm_selfattn_s = BiLSTMWrapper(config, config['hidden_size'] * 4, config['hidden_size'])
        self.lstm_selfattn_q = BiLSTMWrapper(config, config['hidden_size'] * 4, config['hidden_size'])
        self.linear = nn.Linear(config['hidden_size'] * 4, 2) # without max pool
        # self.linear = nn.Linear(config['hidden_size']*8, 2)

    def forward(self, batch):
        S_mask = batch['doc_mask'].byte()
        Q_mask = batch['query_mask'].byte()
        s, q = self.encoder_layer(batch, S_mask, Q_mask)
        # print("lstm_emb", s.shape, q.shape)
        s_tilde, q_tilde = self.biattention_block(s, q, S_mask, Q_mask)
        s, _ = self.lstm_biattn_s(s_tilde, S_mask)
        q, _ = self.lstm_biattn_q(q_tilde, Q_mask)
        # print("lstm_biattn", s.shape, q.shape)
        s_hat = self.selfattention_block_s(s, s, S_mask, S_mask)
        q_hat = self.selfattention_block_q(q, q, Q_mask, Q_mask)
        # print("selfattn", s_hat.shape, q_hat.shape)
        s = torch.cat([s_tilde, s_hat], dim=-1)
        q = torch.cat([q_tilde, q_hat], dim=-1)
        # print("cat", s.shape, q.shape)
        s_bar, _ = self.lstm_selfattn_s(s, S_mask)
        q_bar, _ = self.lstm_selfattn_q(q, Q_mask)
        # print("lstm_selfattn", s_bar.shape, q_bar.shape)
        s_mean = mean_pooling(s_bar, S_mask)
        q_mean = mean_pooling(q_bar, Q_mask)
        # print("pool", s_mean.shape, q_mean.shape)
        s_max = max_pooling(s_bar, S_mask)
        q_max = max_pooling(q_bar, Q_mask)
        # logits = gelu(self.linear(torch.cat([s_mean, q_mean], dim=-1)))
        logits = self.linear(torch.cat([s_mean + s_max, q_mean + q_max], dim=-1))
        prob = F.softmax(logits, dim=-1)
        return prob
