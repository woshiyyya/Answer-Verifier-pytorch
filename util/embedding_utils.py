from util.text_utils import normalize_text, END, STA
import numpy as np
import spacy
import json
from tqdm import tqdm
from util.logger import create_logger

logger = create_logger(__name__)
NLP = spacy.load('en', disable=['vectors', 'textcat', 'parser'])


def build_embedding(emb_path, vocab, dim=300):
    vocab_size = len(vocab)
    embedding = np.zeros((vocab_size, dim))
    with open(emb_path, 'r') as f:
        for line in tqdm(f, total=2196017):
            elements = line.split()
            tok = normalize_text(" ".join(elements[:-dim]))
            if tok in vocab:
                embedding[vocab[tok]] = [float(x) for x in elements[-dim:]]
    return embedding


def build_glove_idx(doc, vocab):
    return [vocab[tok.text] for tok in doc if len(tok.text) > 0]


def build_char_idx(doc, vocab):
    idx_list = []
    for tok in doc:
        if len(tok.text) > 0:
            idx_list.append([vocab[ch] for ch in tok.text])
    return idx_list


def build_ner_idx(doc, ner_vocab):
    return [ner_vocab["{}_{}".format(tok.ent_type_, tok.ent_iob_)] for tok in doc if len(tok.text) > 0]


def build_tag_idx(doc, tag_vocab):
    return [tag_vocab[tok.tag_] for tok in doc if len(tok.text) > 0]


def build_data(data, vocab, tag_vocab, ner_vocab, char_vocab, fout, append_answer=True):
    docs = [normalize_text('{} {}'.format(doc, END)) for doc in data["answer_sentence"]]
    if append_answer:
        querys = [normalize_text('{} {} {} {}'.format(ans, STA, query, END)) for ans, query in zip(data['answer'], data['question'])]
    else:
        querys = [normalize_text('{} {}'.format(query, END)) for query in data['question']]

    logger.info("parsing docs...")
    doc_tokened = [doc for doc in NLP.pipe(docs, batch_size=10000)]
    logger.info("parsing querys...")
    query_tokened = [query for query in NLP.pipe(querys, batch_size=10000)]

    logger.info("creating case...")
    writer = open(fout, 'w', encoding='utf-8')

    for i, (doc, query) in tqdm(enumerate(zip(doc_tokened, query_tokened))):
        case = dict()
        case['doc_glove'] = build_glove_idx(doc, vocab)
        case['doc_tag'] = build_tag_idx(doc, tag_vocab)
        case['doc_ner'] = build_ner_idx(doc, ner_vocab)
        case['doc_char'] = build_char_idx(doc, char_vocab)
        case['query_glove'] = build_glove_idx(query, vocab)
        case['query_tag'] = build_tag_idx(query, tag_vocab)
        case['query_ner'] = build_ner_idx(query, ner_vocab)
        case['query_char'] = build_char_idx(query, char_vocab)
        case['is_impossible'] = bool(data['is_impossible'][i])
        writer.write('{}\n'.format(json.dumps(case)))

    writer.close()
