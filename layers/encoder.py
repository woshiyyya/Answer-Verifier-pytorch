import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.autograd import Variable
from layers.BIDAF_layers import CharEmbeddingLayer, HighwayNetwork
from layers.rnn import BiLSTMWrapper


class ContextualEmbed(nn.Module):
    def __init__(self, path, vocab_size, emb_dim=300, embedding=None, padding_idx=0):
        super(ContextualEmbed, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        if embedding is not None:
            self.embedding.weight.data = embedding
        self.setup_eval_embed(embedding)

        state_dict = torch.load(path)
        self.rnn1 = nn.LSTM(300, 300, num_layers=1, bidirectional=True)
        self.rnn2 = nn.LSTM(600, 300, num_layers=1, bidirectional=True)
        state_dict1 = dict([(name, param.data) if isinstance(param, Parameter) else (name, param)
                        for name, param in state_dict.items() if '0' in name])
        state_dict2 = dict([(name.replace('1', '0'), param.data) if isinstance(param, Parameter) else (name.replace('1', '0'), param)
                        for name, param in state_dict.items() if '1' in name])
        self.rnn1.load_state_dict(state_dict1)
        self.rnn2.load_state_dict(state_dict2)
        for p in self.parameters():
            p.requires_grad = False
        self.output_size = 600

    def setup_eval_embed(self, eval_embed, padding_idx=0):
        self.eval_embed = nn.Embedding(eval_embed.size(0), eval_embed.size(1), padding_idx=padding_idx)
        self.eval_embed.weight.data = eval_embed
        for p in self.eval_embed.parameters():
            p.requires_grad = False

    def forward(self, x_idx, x_mask):
        # emb = self.embedding if self.training else self.eval_embed
        emb = self.embedding
        x_hiddens = emb(x_idx)
        lengths = x_mask.data.eq(1).long().sum(1)
        max_len = x_mask.size(1)
        lens, indices = torch.sort(lengths, 0, True)
        output1, _ = self.rnn1(pack(x_hiddens[indices], lens.tolist(), batch_first=True))
        output2, _ = self.rnn2(output1)
        output1 = unpack(output1, batch_first=True, total_length=max_len)[0]
        output2 = unpack(output2, batch_first=True, total_length=max_len)[0]
        _, _indices = torch.sort(indices, 0)
        output1 = output1[_indices]
        output2 = output2[_indices]
        return output1, output2


class LexiconEncoder(nn.Module):
    def __init__(self, config, embedding=None, use_char_emb=False, use_char_rnn=True, use_exm=True, use_cove=True):
        super(LexiconEncoder, self).__init__()
        self.config = config
        self.use_char_emb, self.use_char_rnn = use_char_emb, use_char_rnn
        self.use_exm, self.use_cove = use_exm, use_cove

        self.dropout_emb = nn.Dropout(config['dropout_emb'])
        self.dropout_cove = nn.Dropout(config['dropout_cov'])

        # GloVe Embedding
        emb_size = self.create_word_embedding(embedding, config)
        self.output_size = emb_size

        # CoVe Embedding
        cove_size = self.create_cove(config, embedding=embedding) if use_cove else 0
        self.cove_size = cove_size * 2
        self.output_size += cove_size * 2

        # Exact Match Feature
        if use_exm:
            self.output_size += 1

        # Character-level Embedding (Char-CNN/Char-RNN)
        if use_char_emb:
            if use_char_rnn:
                self.char_emb_size = self.create_char_embedding(config['char_vocab_size'], config['char_emb_dim'])
                self.hidden_size = 25
                self.lstm = BiLSTMWrapper(config, self.char_emb_size, self.hidden_size)
                self.output_size += 2 * self.hidden_size
            else:
                char_cnn_out_dim = 50
                self.char_cnn = CharEmbeddingLayer(char_single_embedding_dim=config['char_emb_dim'],
                                                   char_embedding_dim=char_cnn_out_dim,
                                                   filter_height=5,
                                                   dropout=0.3,
                                                   char_vocab_size=config['char_vocab_size'])
                self.output_size += char_cnn_out_dim
                self.highway_net = HighwayNetwork(char_cnn_out_dim)

    @staticmethod
    def create_embedding(vocab_size, embed_size, padding_idx=0):
        embed = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        return embed

    def create_word_embedding(self, embedding=None, config=None):
        vocab_size = config['vocab_size']
        embed_size = config['embed_size']
        self.embedding = self.create_embedding(vocab_size, embed_size)
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
        self.char_embedding = self.create_embedding(vocab_size, embed_size)
        return embed_size

    def create_cove(self, config, embedding=None, padding_idx=0):
        self.ContextualEmbed = ContextualEmbed(config['cove_path'], config['vocab_size'], embedding=embedding, padding_idx=padding_idx)
        return self.ContextualEmbed.output_size

    def patch(self, v):
        if self.config['cuda']:
            v = Variable(v.cuda(non_blocking=True))
        else:
            v = Variable(v)
        return v

    def forward(self, batch):
        embedding = self.embedding
        doc_tok = self.patch(batch['doc_tok'])
        doc_mask = self.patch(batch['doc_mask'])
        query_tok = self.patch(batch['query_tok'])
        query_mask = self.patch(batch['query_mask'])
        doc_input_list = []
        query_input_list = []

        doc_emb, query_emb = embedding(doc_tok), embedding(query_tok)
        doc_emb = self.dropout_emb(doc_emb)
        query_emb = self.dropout_emb(query_emb)
        doc_input_list.append(doc_emb)
        query_input_list.append(query_emb)

        if self.use_cove:
            doc_cove_low, doc_cove_high = self.ContextualEmbed(doc_tok, doc_mask)
            query_cove_low, query_cove_high = self.ContextualEmbed(query_tok, query_mask)
            doc_cove_low = self.dropout_cove(doc_cove_low)
            doc_cove_high = self.dropout_cove(doc_cove_high)
            query_cove_low = self.dropout_cove(query_cove_low)
            query_cove_high = self.dropout_cove(query_cove_high)
            doc_input_list.append(doc_cove_low)
            doc_input_list.append(doc_cove_high)
            query_input_list.append(query_cove_low)
            query_input_list.append(query_cove_high)

        if self.use_exm:
            doc_exm = self.patch(batch['doc_exm']).unsqueeze(-1)
            query_exm = self.patch(batch['query_exm']).unsqueeze(-1)
            doc_input_list.append(doc_exm)
            query_input_list.append(query_exm)

        if self.use_char_emb:
            doc_char = self.patch(batch['doc_char'])
            query_char = self.patch(batch['query_char'])
            doc_char_mask = self.patch(batch['doc_char_mask'])  # [N, ld, lw]
            query_char_mask = self.patch(batch['query_char_mask'])
            if self.use_char_rnn:
                doc_char_emb = self.char_embedding(doc_char)
                query_char_emb = self.char_embedding(query_char)
                N = self.config['batch_size']
                ld = self.config['doc_maxlen']
                lq = self.config['query_maxlen']
                lw = self.config['word_maxlen']

                char_emb_size = self.char_emb_size
                doc_char_emb = doc_char_emb.contiguous().view(-1, lw, char_emb_size) # [N*ld, lw, char_emb]
                query_char_emb = query_char_emb.contiguous().view(-1, lw, char_emb_size)
                doc_char_mask = doc_char_mask.contiguous().view(-1, lw)
                query_char_mask = query_char_mask.contiguous().view(-1, lw)

                _H, (ht, _c) = self.lstm(doc_char_emb, doc_char_mask, dropout=False) # ht: [N*ld, 2 * hidden_size]
                doc_char = ht.contiguous().view(N, ld, 2*self.hidden_size)
                _H, (ht, _c) = self.lstm(query_char_emb, query_char_mask, dropout=False)
                query_char = ht.contiguous().view(N, lq, 2 * self.hidden_size)

                doc_input_list.append(doc_char)
                query_input_list.append(query_char)
            else:
                doc_char_emb = self.char_cnn(doc_char, self.training)
                doc_char_emb = self.highway_net(doc_char_emb, self.training)

                query_char_emb = self.char_cnn(query_char, self.training)
                query_char_emb = self.highway_net(query_char_emb, self.training)

                doc_input_list.append(doc_char_emb)
                query_input_list.append(query_char_emb)

        doc_emb = torch.cat(doc_input_list, dim=2)
        query_emb = torch.cat(query_input_list, dim=2)
        return doc_emb, query_emb


class EncoderLayer(nn.Module):
    def __init__(self, config, embedding):
        super(EncoderLayer, self).__init__()
        self.lexicon_encoder = LexiconEncoder(config, embedding)
        input_size = self.lexicon_encoder.output_size
        self.doc_lstm = BiLSTMWrapper(config, input_size, config['hidden_size'])
        self.query_lstm = self.doc_lstm

    def forward(self, batch, s_mask, q_mask):
        doc_emb, query_emb = self.lexicon_encoder(batch)
        doc_ouput, _ = self.doc_lstm(doc_emb, s_mask)
        query_output, _ = self.query_lstm(query_emb, q_mask)
        return doc_ouput, query_output
