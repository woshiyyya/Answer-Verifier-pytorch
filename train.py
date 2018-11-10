import argparse
import torch
import torch.nn as nn
from model import VerifierModel
from util.utils import load_meta, load_data, dump_data, BatchGen, set_environment
from tqdm import tqdm
import numpy as np
from util.logger import create_logger

logger = create_logger(__name__)
tr_acc_log = []
te_acc_log = []

def print_doc_length(data_in):
    dmaxlen = 0
    qmaxlen = 0
    dtt = 0
    qtt = 0
    for i, j in zip(data_in['doc'], data_in['query']):
        dtt += len(i)
        qtt += len(j)
        dmaxlen = len(i) if len(i) > dmaxlen else dmaxlen
        qmaxlen = len(j) if len(j) > qmaxlen else qmaxlen
    print(dmaxlen, qmaxlen, dtt/len(data_in['doc']), qtt/len(data_in['doc']))


def evaluate(pred_prob, label):
    pred_prob = pred_prob.data.cpu().numpy()
    label = label.cpu().numpy()
    pred = np.argmax(pred_prob, axis=-1)
    accuracy = sum(pred == label)/pred.shape[0]
    return accuracy


def predict():
    tr_gen.reset()
    te_gen.reset()
    train_acc = 0
    test_acc = 0
    n_eval = config['n_eval'] // config['batch_size']
    name = config['name']
    model.eval()
    for i, te_batch in enumerate(te_gen):
        prob = model(te_batch)
        test_acc += evaluate(prob, te_batch['label'])
        test_loss = torch.sum(criterion(prob, te_batch['label']))

    for i, tr_batch in enumerate(tr_gen):
        if i == n_eval:
            break
        prob = model(tr_batch)
        train_acc += evaluate(prob, tr_batch['label'])

    train_acc /= n_eval
    test_acc /= len(te_gen)
    logger.info(name + "  train_acc: " + str(train_acc))
    logger.info(name + "  test_acc: " + str(test_acc))
    logger.info(name + "  test_loss:" + str(test_loss.cpu().detach().numpy()))
    tr_acc_log.append(train_acc)
    te_acc_log.append(test_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="no seed")
    parser.add_argument('--train_path', type=str, default="resource/addchar_train_input.txt")
    parser.add_argument('--test_path', type=str, default='resource/addchar_test_input.txt')
    parser.add_argument('--meta_path', type=str, default="resource/meta.pkl")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)

    parser.add_argument('--embed_size', type=int, default=300)  # glove 300d
    parser.add_argument('--char_emb_size', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=300)

    parser.add_argument('--lstm_layers', type=int, default=1)

    parser.add_argument('--dropout_emb', type=float, default=0.1)
    parser.add_argument('--dropout_attn', type=float, default=0.1)
    parser.add_argument('--dropout_resid', type=float, default=0.1)
    parser.add_argument('--dropout_lstm', type=float, default=0.3)
    parser.add_argument('--dropout_cnn', type=float, default=0.1)

    parser.add_argument('--doc_maxlen', type=int, default=100)
    parser.add_argument('--query_maxlen', type=int, default=40)
    parser.add_argument('--word_maxlen', type=int, default=10)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--fix_embedding', action='store_true', default=False)
    parser.add_argument('--tune_oov', action='store_true', default=True)
    parser.add_argument('--use_char_emb', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--n_eval', type=int, default=512)
    config = parser.parse_args().__dict__

    # set_environment(seed=420, set_cuda=config['cuda'])

    embedding, config = load_meta(config)
    train_data, train_label = load_data(config['train_path'])
    test_data, test_label = load_data(config['test_path'])

    print_doc_length(train_data)
    print_doc_length(test_data)

    generator = BatchGen(config, train_data, train_label,
                         config['batch_size'], config['doc_maxlen'], config['query_maxlen'])
    tr_gen = BatchGen(config, train_data, train_label,
                      config['batch_size'], config['doc_maxlen'], config['query_maxlen'])
    te_gen = BatchGen(config, test_data, test_label,
                      config['batch_size'], config['doc_maxlen'], config['query_maxlen'], is_training=False)

    logger.info("load data complete!")
    model = VerifierModel(config)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 betas=(config['b1'], config['b2']),
                                 eps=config['e'],
                                 weight_decay=config['decay'])
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss(reduce=False)

    if config['cuda']:
        model.cuda()
    for epc in range(config['epochs']):
        for i, batch in tqdm(enumerate(generator), total=len(generator)):
            model.train()
            prob = model(batch)
            train_loss = torch.sum(criterion(prob, batch['label']))
            train_loss.backward()
            if i % 500 == 0:
                # print(model.lstm_emb.bilstm.all_weights[0][0].grad)
                # print(model.lexicon_encoder.char_embedding.weight.grad.shape)
                # print(sum(model.lexicon_encoder.char_embedding.weight.grad).shape)
                print(model.lexicon_encoder.embedding.weight.grad)
                print(model.lexicon_encoder.embedding.weight.requires_grad)
            optimizer.step()
            optimizer.zero_grad()
            # print("step: ", i, "loss:", train_loss)
            if i % 500 == 0:
                predict()
        generator.reset()

    dump_data([tr_acc_log, te_acc_log], "log/" + config['name'] + ".pkl")