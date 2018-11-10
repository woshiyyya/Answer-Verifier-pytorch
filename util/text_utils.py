import re
import spacy
import pandas
import unicodedata
from collections import Counter
from tqdm import tqdm

PAD = 'PADPAD'
UNK = 'UNKUNK'
STA = 'BOSBOS'
END = 'EOSEOS'

PAD_ID = 0
UNK_ID = 1
STA_ID = 2
END_ID = 3

DigitsMapper = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
                'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7','eight': '8', 'nine': '9', 'ten': '10'}


def normalize_text(text):
    if isinstance(text, str):
        return unicodedata.normalize('NFD', text)
    else:
        return ""


def standardize_text(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub(r'\s*\n\s*', ' \n ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()


class Vocabulary(object):
    def __init__(self):
        self.tok2ind = {PAD: PAD_ID, UNK: UNK_ID, STA: STA_ID, END: END_ID}
        self.ind2tok = {PAD_ID: PAD, UNK_ID: UNK, STA_ID: STA, END_ID: END}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return key in self.tok2ind

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.tok2ind.get(key, UNK_ID)
        if isinstance(key, int):
            return self.ind2tok.get(key, UNK)

    def __setitem__(self, key, value):
        if isinstance(key, str) and isinstance(value, int):
            self.tok2ind[key] = value
        elif isinstance(key, int) and isinstance(value, str):
            self.ind2tok[key] = value
        else:
            raise RuntimeError("Invalid key-val types")

    def add(self, token):
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def get_vocab(self):
        return [self[idx] for idx in range(0, len(self))]

    def transform(self, case):
        word = []
        for wd in case['doc_char']:
            for ch in wd:
                word.append(self[ch])
            word.append(" ")
        print("".join(word))

    @staticmethod
    def build(token_list):
        vocab = Vocabulary()
        for tok in token_list:
            vocab.add(tok)
        return vocab


def build_vocab(data, embedding_vocab, threads=24, tagner_on = False):
    if tagner_on:
        nlp = spacy.load('en', disable=['vectors', 'textcat', 'parser'])
    else:
        nlp = spacy.load('en', disable=['vectors', 'textcat', 'parser', 'ner', 'tagger'])

    docs = [standardize_text(str(doc)) for doc in data['answer_sentence']]
    querys = [standardize_text(str(query)) for query in data['question']]
    doc_tokened = [doc for doc in nlp.pipe(docs, batch_size=10000, n_threads=threads)]
    query_tokened = [query for query in nlp.pipe(querys, batch_size=10000, n_threads=threads)]
    merged = doc_tokened + query_tokened

    vocab, tag_vocab, ner_vocab = None, None, None
    counter = Counter()
    tag_counter = Counter()
    ner_counter = Counter()
    for sent in tqdm(merged, total=len(merged)):
        counter.update([normalize_text(token.text) for token in sent if len(normalize_text(token.text)) > 0])
        if tagner_on:
            tag_counter.update([tok.tag_ for tok in sent if len(tok.tag_) > 0])
            ner_counter.update(['{}_{}'.format(tok.ent_type_, tok.ent_iob_) for tok in sent])
    vocab = sorted([tok for tok in counter if tok in embedding_vocab], key=counter.get, reverse=True)
    vocab = Vocabulary.build(vocab)
    char_vocab = build_char_vocab(merged)
    if tagner_on:
        tag_vocab = sorted([tok for tok in tag_counter], key=tag_counter.get, reverse=True)
        ner_vocab = sorted([tok for tok in ner_counter], key=ner_counter.get, reverse=True)
        tag_vocab = Vocabulary.build(tag_vocab)
        ner_vocab = Vocabulary.build(ner_vocab)

    total = sum(counter.values())
    matched = sum(counter[tok] for tok in vocab)
    print("raw vocab : vocab in glove : glove vocab = {0}:{1}:{2}".format(len(counter), len(vocab), len(embedding_vocab)))
    print("OOV rate:", matched/total)
    return vocab, tag_vocab, ner_vocab, char_vocab


def build_char_vocab(tokened_docs, threshold = 100):
    char_counter = Counter()
    for doc in tokened_docs:
        for token in doc:
            char_counter.update([ch for ch in token.text])

    char_vocab = sorted([ch for ch in char_counter if char_counter[ch] > threshold], key=char_counter.get, reverse=True)
    char_vocab = Vocabulary.build(char_vocab)
    return char_vocab
