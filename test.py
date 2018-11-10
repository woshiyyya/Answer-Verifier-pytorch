from util.utils import BatchGen
import numpy as np
from random import randint
from layers.char_cnn import *
from layers.highway import *

def test_lstm():
    X = torch.FloatTensor(3, 4, 5).fill_(0)
    length = torch.LongTensor(3)
    for i in range(3):
        l = randint(1, 4)
        length[i] = l
        for j in range(l):
            X[i, j] = i+1
    print(X)
    bilstm = nn.LSTM(input_size=5, hidden_size=6, bidirectional=True,batch_first=True)
    H, (h0, c0) = bilstm(X)
    print("-------++++++--------\n", h0, h0.data.shape)
    print("-------++++++--------\n", H, H.data.shape)

    bilstm = nn.LSTM(input_size=5, hidden_size=6, bidirectional=True,batch_first=True)
    max_length = X.shape[-2]
    seq_lengths = length
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)
    _, desorted_indices = torch.sort(indices, descending=False)
    X = nn.utils.rnn.pack_padded_sequence(X[indices], sorted_seq_lengths, batch_first=True)
    print("---------------------\n", X, X.data.shape)
    H, (h0, c0) = bilstm(X)
    print("-------------hhh--------\n", c0, c0.data.shape)  # Amazing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Just the last hidden state
    print("---------------------\n", H, H.data.shape)
    H, _ = nn.utils.rnn.pad_packed_sequence(H, batch_first=True, total_length=max_length)
    print("---------------------\n", H, H.data.shape)
    H = H[desorted_indices]


def test_charcnn():
    config = {}
    config['batch_size'] = 8
    config['word_maxlen'] = 10
    config['char_emb_size'] = 5
    config['dropout_cnn'] = 0.1
    CNN = Char_CNN(config)
    highway = HighwayMLP(300)
    X = torch.rand((8, 7, 10, 5))
    out = CNN(X)
    out1 = highway(out)
    print(out1.shape)


def test_batch():
    config = {}
    config['doc_maxlen'] = 20
    config['query_maxlen'] = 10
    config['word_maxlen'] = 5
    config['cuda'] = 0
    data = {}
    data['doc'] = [[1 for _ in range(randint(1, 20))] for _ in range(7)]
    data['query'] = [[2 for _ in range(randint(1, 10))] for _ in range(7)]
    doc_len = [len(d) for d in data['doc']]
    query_len = [len(q) for q in data['query']]
    data['doc_char'] = [[[1 for _ in range(randint(1, 8))] for _ in range(doc_len[i])] for i in range(7)]
    data['query_char'] = [[[2 for _ in range(randint(1, 8))] for _ in range(query_len[i])]for i in range(7)]
    label = [randint(0, 1) for _ in range(7)]
    generator = BatchGen(config, data, label, 3, 20, 10, is_training=True)

    for i, batch in enumerate(generator):
        print("batch:", i)
        print(batch)


if __name__ == '__main__':
    test_lstm()