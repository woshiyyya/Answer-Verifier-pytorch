import torch
import pickle
import random
from tqdm import tqdm
from util.text_utils import normalize_text
import os
import json
from torch.autograd import Variable
import numpy


def set_environment(seed, set_cuda=False):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and set_cuda:
        torch.cuda.manual_seed_all(seed)


def load_data(path, use_char=True):
    data = dict()
    doc = []
    doc_char = []
    query = []
    query_char = []
    label = []
    with open(path, 'r') as f:
        for line in f:
            case = json.loads(line)
            doc.append(case['doc_glove'])
            query.append(case['query_glove'])
            if use_char:
                doc_char.append(case['doc_char'])
                query_char.append(case['query_char'])
            label.append(case['is_impossible'])
    data['doc'] = doc
    data['query'] = query
    if use_char:
        data['doc_char'] = doc_char
        data['query_char'] = query_char
    return data, label


def dump_data(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_meta_(path):
    meta = pickle.load(open(path, 'rb'))
    char_vocab = pickle.load(open("resource/char_vocab.pkl", 'rb'))
    return meta['vocab'], meta['tag_vocab'], meta['ner_vocab'], char_vocab, meta['embedding']


def load_meta(config):
    meta_path = config['meta_path']
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    embedding = torch.Tensor(meta['embedding'])
    config['vocab_size'] = len(meta['vocab'])
    # TODO: fuse char vocab into meta
    char_vocab = pickle.load(open("resource/char_vocab.pkl", 'rb'))
    config['char_vocab_size'] = len(char_vocab)
    return embedding, config


def load_glove_vocab(path, dim=300, glove_vocab_path = "resource/glove_vocab.pkl"):
    if os.path.exists(glove_vocab_path):
        vocab = pickle.load(open(glove_vocab_path, 'rb'))
    else:
        vocab = set()
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=2196017):
                elements = line.split()
                token = normalize_text(" ".join(elements[:-dim]))
                vocab.add(token)
        pickle.dump(vocab, open(glove_vocab_path, 'wb'))
    return vocab


class BatchGen(object):
    def __init__(self, config, data: dict, label, batch_size, doc_maxlen, query_maxlen, is_training=True):
        self.config = config
        self.batch_size = batch_size
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.word_maxlen = config['word_maxlen']
        self.training = is_training
        self.offset = 0
        self.total_num = len(label)
        [self.doc, self.query, self.doc_char, self.query_char, self.label] = \
            self.clip_tail([data['doc'], data['query'], data['doc_char'], data['query_char'], label])
        assert len(self.doc) == len(self.query) and len(self.doc) == len(self.label)

        if is_training:
            indices = list(range(self.total_num))
            random.shuffle(indices)
            self.doc = [self.doc[i] for i in indices]
            self.query = [self.query[i] for i in indices]
            self.doc_char = [self.doc_char[i] for i in indices]
            self.query_char = [self.query_char[i] for i in indices]
            self.label = [self.label[i] for i in indices]

        self.batches = [(self.doc[i: i+batch_size], self.query[i: i+batch_size],
                         self.doc_char[i: i+batch_size], self.query_char[i: i+batch_size], self.label[i: i+batch_size])
                        for i in range(0, self.total_num, batch_size)]

    def reset(self):
        if self.training:
            indices = list(range(len(self.batches)))
            random.shuffle(indices)
            self.batches = [self.batches[i] for i in indices]
        self.offset = 0

    def clip_tail(self, data):
        clip_num = self.total_num % self.batch_size
        self.total_num = self.total_num - clip_num
        cliped_data = [d[:-clip_num] for d in data]
        return cliped_data

    def patch(self, v):
        if self.config['cuda']:
            v = Variable(v.cuda(non_blocking=True))
        else:
            v = Variable(v)
        return v

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        batch_size = self.batch_size

        while self.offset < len(self):
            batch = self.batches[self.offset]
            batch_dict = {}
            doc_tok_tensor = torch.LongTensor(batch_size, self.doc_maxlen).fill_(0)
            query_tok_tensor = torch.LongTensor(batch_size, self.query_maxlen).fill_(0)
            doc_char_tensor = torch.LongTensor(batch_size, self.doc_maxlen, self.word_maxlen).fill_(0)
            query_char_tensor = torch.LongTensor(batch_size, self.query_maxlen, self.word_maxlen).fill_(0)

            for i, (doc, query, doc_char, query_char, _) in enumerate(zip(*batch)):
                d_len = min(len(doc), self.doc_maxlen)
                q_len = min(len(query), self.query_maxlen)
                doc_tok_tensor[i, :d_len] = torch.LongTensor(doc[:d_len])
                query_tok_tensor[i, :q_len] = torch.LongTensor(query[:q_len])
                doc_char = doc_char[:d_len]
                query_char = query_char[:q_len]
                for j, dch in enumerate(doc_char):
                    ch_len = min(len(dch), self.word_maxlen)
                    doc_char_tensor[i, j, :ch_len] = torch.LongTensor(dch[:ch_len])
                for j, qch in enumerate(query_char):
                    ch_len = min(len(qch), self.word_maxlen)
                    query_char_tensor[i, j, :ch_len] = torch.LongTensor(qch[:ch_len])

            batch_dict['doc_tok'] = self.patch(doc_tok_tensor)
            batch_dict['doc_mask'] = self.patch(1 - torch.eq(doc_tok_tensor, 0))
            batch_dict['doc_char'] = self.patch(doc_char_tensor)
            batch_dict['doc_char_mask'] = self.patch(1 - torch.eq(doc_char_tensor, 0))
            batch_dict['query_tok'] = self.patch(query_tok_tensor)
            batch_dict['query_mask'] = self.patch(1 - torch.eq(query_tok_tensor, 0))
            batch_dict['query_char'] = self.patch(query_char_tensor)
            batch_dict['query_char_mask'] = self.patch(1 - torch.eq(query_char_tensor, 0))
            batch_dict['label'] = self.patch(torch.LongTensor(batch[-1]))
            self.offset += 1
            yield batch_dict

