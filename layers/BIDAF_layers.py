import torch
import torch.nn as nn
import torch.nn.functional as F


class CharEmbeddingLayer(nn.Module):
    def __init__(self, char_single_embedding_dim, char_embedding_dim, filter_height, dropout, char_vocab_size):
        """
        :param char_single_embedding_dim: 8 as in the original implementation of BiDAF
        :param char_embedding_dim: 100 as in the original implementation of BiDAF
        :param filter_height: 5 as in the original implementation of BiDAF
        :param dropout:
        :param char_vocab_size: the size of character vocabulary
        """
        super(CharEmbeddingLayer, self).__init__()
        self.char_single_embedding_dim = char_single_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.filter_height = filter_height
        self.dropout = dropout
        self.char_vocab_size = char_vocab_size

        self.embedding_lookup = nn.Embedding(self.char_vocab_size, char_single_embedding_dim)
        self.cnn = nn.Conv1d(char_single_embedding_dim, char_embedding_dim, filter_height, padding=0)

    def forward(self, text_char, is_train):
        """
        :param text_char: character id of each word in the text, [batch_size, text_len, word_length]
        :param is_train: bool, whether it is on training
        """
        batch_size, max_length, max_word_length = text_char.size()
        # embedding look up
        text_char = text_char.contiguous().view(batch_size * max_length, max_word_length)
        text_char = self.embedding_lookup(text_char)
        assert text_char.size() == (batch_size * max_length, max_word_length, self.char_single_embedding_dim)

        # dropout + cnn
        if self.dropout < 1.0:
            assert is_train is not None
            if is_train:
                text_char = nn.Dropout(p=self.dropout)(text_char)
        text_char = torch.transpose(text_char, 1, 2)
        assert text_char.size() == (batch_size * max_length, self.char_single_embedding_dim, max_word_length), "text_char.size()=%s"%(text_char.size())
        text_char = self.cnn(text_char)
        assert text_char.size() == (batch_size * max_length, self.char_embedding_dim, text_char.size(2))
        text_char = text_char.contiguous().view(batch_size * max_length * self.char_embedding_dim, -1)
        text_char = nn.functional.relu(text_char)

        # maxpool
        text_char = torch.max(text_char, 1)[0]
        assert text_char.size() == (batch_size * max_length * self.char_embedding_dim, )
        text_char = text_char.contiguous().view(batch_size, max_length, self.char_embedding_dim)
        return text_char


class HighwayNetwork(nn.Module):
    def __init__(self, size, num_layers=1, dropout=0.5):
        super(HighwayNetwork, self).__init__()

        self.num_layers = num_layers
        self.size = size
        self.dropout = dropout
        self.trans = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

    def forward(self, x, is_train):
        # assert len(x.size()) == 2 and x.size(1) == self.size
        for layer in range(self.num_layers):
            gate = nn.functional.sigmoid(self.gate[layer](x))
            if is_train:
                gate = nn.functional.dropout(gate, p=self.dropout)

            trans = nn.functional.relu(self.trans[layer](x))
            if is_train:
                trans = nn.functional.dropout(trans, p=self.dropout)

            x = gate * trans + (1 - gate) * x
        return x