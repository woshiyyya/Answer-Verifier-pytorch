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


def load_data(path):
    data = []
    label = []

    with open(path, 'r') as f:
        for line in f:
            case = json.loads(line)
            data.append(case)
            label.append(case['is_impossible'])
    return data, label


def dump_data(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_meta_(path):
    meta = pickle.load(open(path, 'rb'))
    char_vocab = pickle.load(open("resource/char_vocab.pkl", 'rb'))
    return meta['vocab'], meta['tag_vocab'], meta['ner_vocab'], char_vocab, meta['embedding']


def add_figure(name, writer, global_step, train_loss, test_loss, train_acc, test_acc):
    writer.add_scalar(name + ' data/train_loss', train_loss, global_step)
    writer.add_scalars(name + ' data/loss_group', {'train_loss': train_loss, 'test_loss': test_loss}, global_step)
    writer.add_scalars(name + ' data/acc_group', {'train_acc': train_acc, 'test_acc': test_acc}, global_step)
    return


def load_meta(config):
    meta_path = config['meta_path']
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    embedding = torch.Tensor(meta['embedding'])
    config['vocab_size'] = len(meta['vocab'])
    config['tag_vocab_size'] = len(meta['tag_vocab'])
    config['ner_vocab_size'] = len(meta['ner_vocab'])
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
    def __init__(self, config, raw_data: dict, label, batch_size, is_training=True, augment=False):
        self.config = config
        self.total_num = len(label)

        self.data = raw_data
        self.label = label

        self.batch_size = batch_size
        self.ld = config['doc_maxlen']
        self.lq = config['query_maxlen']
        self.lw = config['word_maxlen']
        self.training = is_training
        self.offset = 0

        self.clip_tail()

        if augment:
            self.augmentation()

        if is_training:
            indices = list(range(self.total_num))
            random.shuffle(indices)
            self.data = [self.data[idx] for idx in indices]

        self.batches = [self.data[i: i + batch_size] for i in range(0, self.total_num, batch_size)]

    def augmentation(self):
        for i in range(self.total_num):
            if self.label[i]:
                self.data.append(self.data[i].copy())
                self.label.append(self.label[i])
        self.total_num = len(self.label)
        self.clip_tail()

    def reset(self):
        if self.training:
            indices = list(range(len(self.batches)))
            random.shuffle(indices)
            self.batches = [self.batches[i] for i in indices]
        self.offset = 0

    def clip_tail(self):
        clip_num = self.total_num % self.batch_size
        self.total_num = self.total_num - clip_num
        self.data = self.data[:self.total_num]

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
            batch_dict = dict()

            N = batch_size
            ld = self.ld
            lq = self.lq
            lw = self.lw

            doc_tok_tensor = torch.LongTensor(N, ld).fill_(0)
            doc_exm_tensor = torch.LongTensor(N, ld).fill_(0)
            doc_char_tensor = torch.LongTensor(N, ld, lw).fill_(0)
            doc_tag_tensor = torch.LongTensor(N, ld).fill_(0)
            doc_ner_tensor = torch.LongTensor(N, ld).fill_(0)

            query_tok_tensor = torch.LongTensor(N, lq).fill_(0)
            query_exm_tensor = torch.LongTensor(N, lq).fill_(0)
            query_char_tensor = torch.LongTensor(N, lq, lw).fill_(0)
            query_tag_tensor = torch.LongTensor(N, lq).fill_(0)
            query_ner_tensor = torch.LongTensor(N, lq).fill_(0)

            label_tensor = torch.LongTensor(N).fill_(0)

            for i, case in enumerate(batch):
                d_len = min(len(case['doc_glove']), ld)
                q_len = min(len(case['query_glove']), lq)

                # token index vector
                doc_tok_tensor[i, :d_len] = torch.LongTensor(case['doc_glove'][:d_len])
                query_tok_tensor[i, :q_len] = torch.LongTensor(case['query_glove'][:q_len])

                # char index vector
                doc_char = case['doc_char'][:d_len]
                query_char = case['query_char'][:q_len]
                for j, dch in enumerate(doc_char):
                    ch_len = min(len(dch), lw)
                    doc_char_tensor[i, j, :ch_len] = torch.LongTensor(dch[:ch_len])
                for j, qch in enumerate(query_char):
                    ch_len = min(len(qch), lw)
                    query_char_tensor[i, j, :ch_len] = torch.LongTensor(qch[:ch_len])

                # label vector
                label_tensor[i] = torch.LongTensor([case['is_impossible']])

                if self.config['use_exm']:
                    doc_exm_tensor[i, :d_len] = torch.LongTensor(case['doc_exm'][:d_len])
                    query_exm_tensor[i, :q_len] = torch.LongTensor(case['query_exm'][:q_len])

                if self.config['use_tag']:
                    doc_tag_tensor[i, :d_len] = torch.LongTensor(case['doc_tag'][:d_len])
                    query_tag_tensor[i, :q_len] = torch.LongTensor(case['query_tag'][:q_len])

                if self.config['use_ner']:
                    doc_ner_tensor[i, :d_len] = torch.LongTensor(case['doc_ner'][:d_len])
                    query_ner_tensor[i, :q_len] = torch.LongTensor(case['query_ner'][:q_len])

            batch_dict['doc_tok'] = self.patch(doc_tok_tensor)
            batch_dict['doc_mask'] = self.patch(1 - torch.eq(doc_tok_tensor, 0))
            batch_dict['doc_char'] = self.patch(doc_char_tensor)
            batch_dict['doc_char_mask'] = self.patch(1 - torch.eq(doc_char_tensor, 0))
            batch_dict['doc_exm'] = self.patch(doc_exm_tensor.float())
            batch_dict['doc_tag'] = self.patch(doc_tag_tensor)
            batch_dict['doc_ner'] = self.patch(doc_ner_tensor)

            batch_dict['query_tok'] = self.patch(query_tok_tensor)
            batch_dict['query_mask'] = self.patch(1 - torch.eq(query_tok_tensor, 0))
            batch_dict['query_char'] = self.patch(query_char_tensor)
            batch_dict['query_char_mask'] = self.patch(1 - torch.eq(query_char_tensor, 0))
            batch_dict['query_exm'] = self.patch(query_exm_tensor.float())
            batch_dict['query_tag'] = self.patch(query_tag_tensor)
            batch_dict['query_ner'] = self.patch(query_ner_tensor)

            batch_dict['label'] = self.patch(label_tensor)
            self.offset += 1
            yield batch_dict
