import os
import argparse
import pickle
from util.text_utils import build_vocab
from util.utils import load_glove_vocab, dump_data, load_meta_
from util.embedding_utils import build_embedding, build_data
import pandas as pd
from pandas import DataFrame
from util.logger import create_logger


columns = ['id', 'question','answer_sentence', 'answer', 'is_impossible']
USER_DIR = os.path.expanduser('~')


def main():
    logger = create_logger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove_path', type=str, default=os.path.join(USER_DIR, "Workspace/data/glove.840B.300d.txt"))
    parser.add_argument('--train_path', type=str, default="data/squad_train_v.csv")
    parser.add_argument('--test_path',type=str, default="data/squad_dev_v.csv")
    parser.add_argument('--meta_path', type=str, default="resource/meta.pkl")
    parser.add_argument('--source_path', type=str, default="resource")
    config = parser.parse_args()

    logger.info("loading data...")
    train_data = DataFrame(pd.read_csv(config.train_path), columns=columns)
    test_data = DataFrame(pd.read_csv(config.test_path), columns=columns)
    '''
    glove_vocab = load_glove_vocab(config.glove_path, dim=300)
    merged_data = pd.concat([train_data, test_data])
    vocab, tag_vocab, ner_vocab, char_vocab = build_vocab(merged_data, glove_vocab, tagner_on=True)
    dump_data(char_vocab, "char_vocab.pkl")
    
    logger.info("building vocab...")
    vocab, tag_vocab, ner_vocab = build_vocab(merged_data, glove_vocab, tagner_on=True)

    logger.info("building embedding...")
    glove_embedding = build_embedding(config.glove_path, vocab)

    logger.info("dumping meta data...")
    meta = {"vocab": vocab, 'tag_vocab': tag_vocab, 'ner_vocab': ner_vocab, 'embedding': glove_embedding}
    dump_data(meta, config.meta_path)
    '''
    vocab, tag_vocab, ner_vocab, char_vocab, embedding = load_meta_(config.meta_path)
    logger.info("building train data...")
    train_input_path = os.path.join(config.source_path, "addexm_train_input.txt")
    build_data(train_data, vocab, tag_vocab, ner_vocab, char_vocab, fout=train_input_path)

    logger.info("building test data...")
    test_input_path = os.path.join(config.source_path, "addexm_test_input.txt")
    build_data(test_data, vocab, tag_vocab, ner_vocab, char_vocab, fout=test_input_path)


if __name__ == "__main__":
    main()