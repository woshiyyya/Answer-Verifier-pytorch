import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers.char_cnn import Char_CNN
import math
from layers.clf_head import Clf_head
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

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


def mask_softmax(W, mask1, mask2):  # W: [N, lq, ls], mask: [N, ls]
    mask1 = mask1.unsqueeze(-1)
    mask2 = mask2.unsqueeze(-2)
    joint_mask = mask1 * mask2
    W = W + (1 - joint_mask.float()) * NEG_INF
    return F.softmax(W, dim=-1)


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
        a = self.attn_dropout(a)
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
        a = self.attn_dropout(a)
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
        B, C = self._biattn(S, Q)
        # B, C = self.mask_biattn(S, Q, S_mask, Q_mask)
        # S_mask = S_mask.unsqueeze(-1).float()
        # Q_mask = Q_mask.unsqueeze(-1).float()
        # B = B * S_mask
        # S = S * S_mask
        # C = C * Q_mask
        # Q = Q * Q_mask
        S_tilde = self.fusion(S, B)
        if self.self_attention:
            return S_tilde
        else:
            Q_tilde = self.fusion(Q, C)
            return S_tilde, Q_tilde


class FusionBlock(nn.Module):
    def __init__(self, config):
        super(FusionBlock, self).__init__()
        self.linear_r = nn.Linear(8*config['hidden_size'], 2*config['hidden_size'])
        self.linear_g = nn.Linear(8*config['hidden_size'], 2*config['hidden_size'])
        self.activation = torch.nn.ReLU()

    def forward(self, x, y):  # x: 2h
        input_r = torch.cat([x, y, x*y, x-y], dim=-1)
        r = gelu(self.linear_r(input_r))
        input_g = torch.cat([x, y, x*y, x-y], dim=-1)
        g = self.activation(self.linear_g(input_g))
        o = g*r + (1-g)*x
        return o


class LayerNorm(nn.Module):
    """Construct a layernorm module in the OpenAI style (epsilon inside the square root)."""
    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


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

        # if X_mask is not None:
        #      H = H * X_mask.unsqueeze(-1).float()
        # if dropout:
        #     H = F.dropout(H, self.pdrop)
        return H, (h0, c0)


class LexiconEncoder(nn.Module):
    def __init__(self, config, embedding=None, use_char_emb=False):
        super(LexiconEncoder, self).__init__()
        self.config = config
        self.use_char_emb = use_char_emb
        self.dropout_emb = nn.Dropout(config['dropout_emb'])
        emb_size = self.create_word_embedding(embedding, config)
        self.output_size = emb_size
        if use_char_emb:
            self.char_emb_size = self.create_char_embedding(config['char_vocab_size'], config['char_emb_size'])
            self.hidden_size = 25
            self.lstm = BiLSTMWrapper(config, self.char_emb_size, self.hidden_size)
            self.output_size += 2 * self.hidden_size
            # self.char_cnn = Char_CNN(config)
            # self.output_size += self.char_cnn.outsize

    @staticmethod
    def create_embed(vocab_size, embed_size, padding_idx=0):
        embed = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        # nn.init.normal_(embed.weight, std=0.02)
        return embed

    def create_word_embedding(self, embedding=None, config=None):
        vocab_size = config['vocab_size']
        embed_size = config['embed_size']
        self.embedding = self.create_embed(vocab_size, embed_size)
        if embedding is not None:
            self.embedding.weight.data = embedding
            if config['fix_embedding']:
                for p in self.embedding.parameters():
                    p.requires_grad = False
            else:
                assert config['tune_oov'] < embedding.size(0)
                fixed_embedding = embedding[config['tune_oov']:]
                self.register_buffer('fixed_embedding', fixed_embedding)
                self.fixed_embedding = fixed_embedding
        return embed_size

    def create_char_embedding(self, vocab_size, embed_size):
        self.char_embedding = self.create_embed(vocab_size, embed_size)
        return embed_size

    def patch(self, v):
        if self.config['cuda']:
            v = Variable(v.cuda(non_blocking=True))
        else:
            v = Variable(v)
        return v

    def forward(self, batch):
        # emb = self.embedding if self.training else self.eval_embed
        embedding = self.embedding
        doc_tok = self.patch(batch['doc_tok'])
        query_tok = self.patch(batch['query_tok'])
        doc_emb, query_emb = embedding(doc_tok), embedding(query_tok)
        if self.use_char_emb:
            doc_char = self.patch(batch['doc_char'])
            query_char = self.patch(batch['query_char'])
            doc_char_mask = self.patch(batch['doc_char_mask'])  # [N, ld, lw]
            query_char_mask = self.patch(batch['query_char_mask'])

            doc_char_emb = self.char_embedding(doc_char)
            query_char_emb = self.char_embedding(query_char)
            # doc_char_emb = self.char_cnn(doc_char_emb)
            # query_char_emb = self.char_cnn(query_char_emb)
            N = self.config['batch_size']
            ld = self.config['doc_maxlen']
            lq = self.config['query_maxlen']
            lw = self.config['word_maxlen']

            char_emb_size = self.char_emb_size
            doc_char_emb = doc_char_emb.view(-1, lw, char_emb_size) # [N*ld, lw, char_emb]
            query_char_emb = query_char_emb.view(-1, lw, char_emb_size)
            doc_char_mask = doc_char_mask.view(-1, lw)
            query_char_mask = query_char_mask.view(-1, lw)

            _H, (ht, _c) = self.lstm(doc_char_emb, doc_char_mask, dropout=False) # ht: [N*ld, 2 * hidden_size]
            doc_char = ht.view(N, ld, 2*self.hidden_size)
            _H, (ht, _c) = self.lstm(query_char_emb, query_char_mask, dropout=False)
            query_char = ht.view(N, lq, 2 * self.hidden_size)

            doc_emb = torch.cat([doc_char, doc_emb], dim=-1)
            query_emb = torch.cat([query_char, query_emb], dim=-1)
        return doc_emb, query_emb


class VerifierModel(nn.Module):
    def __init__(self, config, embedding=None):
        super(VerifierModel, self).__init__()
        self.lexicon_encoder = \
            LexiconEncoder(config, embedding=embedding, use_char_emb=config['use_char_emb'])
        encoded_size = self.lexicon_encoder.output_size
        self.lstm_emb = BiLSTMWrapper(config, encoded_size, config['hidden_size'])
        self.biattention_block = BiAttentionBlock(config, self_attention=False)
        self.lstm_biattn = BiLSTMWrapper(config, config['hidden_size']*2, config['hidden_size'])
        self.selfattention_block_s = BiAttentionBlock(config, self_attention=True)
        self.selfattention_block_q = BiAttentionBlock(config, self_attention=True)
        self.lstm_selfattn = BiLSTMWrapper(config, config['hidden_size']*4, config['hidden_size'])
        self.linear = nn.Linear(config['hidden_size'] * 4, 2) # without max pool
        # self.linear = nn.Linear(config['hidden_size']*8, 2)
        # self.clf_head = Clf_head(config, config['hidden_size'] * 2)

    def setup_eval_embed(self, eval_embed, padding_idx=0):
        self.network.lexicon_encoder.eval_embed = nn.Embedding(eval_embed.size(0), eval_embed.size(1), padding_idx=padding_idx)
        self.network.lexicon_encoder.eval_embed.weight.data = eval_embed
        for p in self.network.lexicon_encoder.eval_embed.parameters():
            p.requires_grad = False
        self.eval_embed_transfer = True

    def forward(self, batch):
        S_mask = batch['doc_mask'].byte()
        Q_mask = batch['query_mask'].byte()
        S_emb, Q_emb = self.lexicon_encoder(batch)
        # print("lexi", S_emb.shape, Q_emb.shape)
        s, _ = self.lstm_emb(S_emb, S_mask)
        q, _ = self.lstm_emb(Q_emb, Q_mask)
        # print("lstm_emb", s.shape, q.shape)
        s_tilde, q_tilde = self.biattention_block(s, q, S_mask, Q_mask)
        # print("biattn", s_tilde.shape, q_tilde.shape)
        s, _ = self.lstm_biattn(s_tilde, S_mask)
        q, _ = self.lstm_biattn(q_tilde, Q_mask)
        # print("lstm_biattn", s.shape, q.shape)
        s_hat = self.selfattention_block_s(s, s, S_mask, S_mask)
        q_hat = self.selfattention_block_q(q, q, Q_mask, Q_mask)
        # print("selfattn", s_hat.shape, q_hat.shape)
        s = torch.cat([s_tilde, s_hat], dim=-1)
        q = torch.cat([q_tilde, q_hat], dim=-1)
        # print("cat", s.shape, q.shape)
        s_bar, _ = self.lstm_selfattn(s, S_mask)
        q_bar, _ = self.lstm_selfattn(q, Q_mask)
        # print("lstm_selfattn", s_bar.shape, q_bar.shape)
        s_mean = mean_pooling(s_bar, S_mask)
        q_mean = mean_pooling(q_bar, Q_mask)
        # print("pool", s_mean.shape, q_mean.shape)
        s_max = max_pooling(s_bar, S_mask)
        q_max = max_pooling(q_bar, Q_mask)

        logits = gelu(self.linear(torch.cat([s_mean, q_mean], dim=-1)))
        # logits = gelu(self.linear(torch.cat([s_mean, q_mean, s_max, q_max], dim=-1)))
        # logits = gelu(self.clf_head(s_bar, q_bar))
        prob = F.softmax(logits, dim=-1)
        return prob
