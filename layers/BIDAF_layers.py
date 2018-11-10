import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import reduce
from operator import mul

from .BIDAF_utils import exp_mask


class GetLogits(nn.Module):
    def __init__(self, input_size, input_keep_prob=1.0, output_size=1, function=None):
        super(GetLogits, self).__init__()

        self.input_keep_prob = input_keep_prob
        self.linear = nn.Linear(input_size, output_size)
        # self.linear = nn.Linear(input_size, output_size, bias=False)
        self.function = function

    def forward(self, args, mask=None, is_train=True):
        '''
        TODO:
        The weight decay can be added to the optimizer
        Also need to squeeze out
        '''
        def linear_logits(args):
            flat_args = [F.dropout(flatten(arg, 1), training=is_train, p=1-self.input_keep_prob) \
                            for arg in args]
            flat_outs = self.linear(torch.cat(flat_args, 1))
            # concatnate h and u in the last dim
            # flat_outs : [batch_size * (max(text_length))^2, 1]
            out = reconstruct(flat_outs, args[0], 1)
            # view out's dimension as the shape of args[0]
            # [batch_size, max(text_length)^2, 1]
            logits = out.squeeze(len(list(args[0].size())) - 1)
            # [batch_size, max(text_length)^2]
            if mask is not None:
                logits = exp_mask(logits, mask)

            return logits

        if self.function == 'tri_linear':
            new_arg = torch.mul(args[0], args[1])
            logit_args = [args[0], args[1], new_arg]
            logits = linear_logits(logit_args)

        elif self.function == 'linear':
            logits = linear_logits(args)
        else :
            print("Warning: Logits is not provided !")
            logits = None

        return logits


def reconstruct(tensor, ref, keep):
    # notice if tensor and ref have different dimensions
    ref_shape = list(ref.size())
    tensor_shape = list(tensor.size())
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] for i in range(tensor_start, len(tensor_shape))]
    target_shape = pre_shape + keep_shape
    out = tensor.view(target_shape)
    return out


def flatten(tensor, keep):
    fixed_shape = list(tensor.size())
    start = len(fixed_shape) - keep
    '''
    In this particular case, the dynamic shape is always the 
    same as the static shape
    '''
    left = reduce(mul, [fixed_shape[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] for i in range(start, len(fixed_shape))]
    flat = tensor.view(out_shape)
    return flat


def masked_softmax(logits, mask=None):
    if mask is not None:
        logits = exp_mask(logits, mask)

    flat_logits = flatten(logits, 1)
    flat_out = F.softmax(flat_logits)
    out = reconstruct(flat_out, logits, 1)
    return out


def softsel(target, logits, mask=None):
    out = masked_softmax(logits, mask)
    out = out.unsqueeze(len(out.size())).mul(target).sum(len(target.size()) - 2)
    return out


class FullConnectionLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout=1.0):
        super(FullConnectionLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = dropout

    def forward(self, x, is_train):
        if self.dropout < 1.0:
            assert is_train is not None
            if is_train:
                x = nn.Dropout(p=self.dropout)(x)
            x = self.linear(x)
        return x

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
    def __init__(self, size, num_layers, dropout):
        super(HighwayNetwork, self).__init__()

        self.num_layers = num_layers
        self.size = size

        self.trans = nn.ModuleList([FullConnectionLayer(size, size, dropout) for _ in range(num_layers)])
        self.gate = nn.ModuleList([FullConnectionLayer(size, size, dropout) for _ in range(num_layers)])

    def forward(self, x, is_train):
        assert len(x.size()) == 2 and x.size(1) == self.size
        for layer in range(self.num_layers):
            gate = nn.functional.sigmoid(self.gate[layer](x, is_train))
            trans = nn.functional.relu(self.trans[layer](x, is_train))
            x = gate * trans + (1 - gate) * x
        return x


class BiAttentionLayer(nn.Module):
    def __init__(self, input_feature_size, input_keep_prob=1.0):
        super(BiAttentionLayer, self).__init__()

        self.input_keep_prob = input_keep_prob
        # input_size: including h_aug, u_aug, hu_aug
        self.get_logits = GetLogits(input_feature_size * 3, function="tri_linear")

    def forward(self, h, u, h_mask=None, u_mask=None, is_train=True):
        dim_num = len(h.size())
        embedding_dim = h.size(dim_num - 1)
        h_length = h.size(dim_num - 2)
        u_length = u.size(dim_num - 2)
        batch_size = h.size(dim_num - 3)
        h_aug = h.unsqueeze(2).repeat(1, 1, u_length, 1)
        # [batch_size, h_length, u_length, embedding_dim(i.e. 2*basic_embedding_dim)]
        u_aug = u.unsqueeze(1).repeat(1, h_length, 1, 1)

        if h_mask is None:
            hu_mask = None
        else:
            h_mask_aug = h_mask.unsqueeze(2).repeat(1, 1, u_length)
            # [batch_size, h_length, u_length]
            u_mask_aug = u_mask.unsqueeze(1).repeat(1, h_length, 1)
            # [batch_size, h_length, u_length]
            hu_mask = h_mask_aug & u_mask_aug

        u_logits = self.get_logits((h_aug, u_aug), mask=hu_mask, is_train=is_train)
        # [N, M, JX, JQ]
        # [batch_size, h_length, u_length]
        u_a = softsel(u_aug, u_logits)
        # [N, M, JX, 2d]
        # [batch_size, h_length, embedding_dim]
        h_a = softsel(h, torch.max(u_logits, len(u_logits.size())-1)[0])
        # [N, M, 2d]
        # [batch_size, embedding_dim]
        h_a = h_a.unsqueeze(1).repeat(1, h_length, 1)
        # [N, M, JX, 2d]
        # [batch_size, h_length, embedding_dim]
        return u_a, h_a, u_logits


class AttentionLayer(nn.Module):
    def __init__(self, input_feature_size):
        super(AttentionLayer, self).__init__()
        self.bi_attention = BiAttentionLayer(input_feature_size)

    def forward(self, h, u, h_mask=None, u_mask=None, is_train=True, multi_flag=False):
        u_a, h_a, att = self.bi_attention(h, u, h_mask=h_mask, u_mask=u_mask, is_train=is_train, multi_flag=multi_flag)
        p0 = torch.cat([h, u_a, torch.mul(h, u_a), torch.mul(h, h_a)], len(h.size())-1)
        # [N, M, JX, 8d]
        # [batch_size, h_length, 4 * embedding_dim]
        return p0, att.clone() # .clone is like deepcopy
