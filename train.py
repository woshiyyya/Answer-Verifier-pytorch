from config import set_args
import pandas as pd
import torch.nn as nn
from model import VerifierModel
from util.utils import *
from tqdm import tqdm
import numpy as np
from util.logger import create_logger
from tensorboardX import SummaryWriter

logger = create_logger(__name__)


def evaluate(pred_prob, label):
    pred_prob = pred_prob.data.cpu().numpy()
    label = label.cpu().numpy()
    pred = np.argmax(pred_prob, axis=-1)
    accuracy = sum(pred == label)/pred.shape[0]
    return accuracy


def submit(path):
    tr_ans = [[], []]
    te_ans = [[], []]
    tr_gen.reset()
    te_gen.reset()
    model.eval()
    for i, te_batch in enumerate(te_gen):
        prob = model(te_batch).detach().cpu().numpy()
        pred = np.argmax(prob, axis=-1)
        te_ans[0].append(prob[:, 1])
        te_ans[1].append(pred)

    for i, tr_batch in enumerate(tr_gen):
        prob = model(tr_batch).detach().cpu().numpy()
        pred = np.argmax(prob, axis=-1)
        tr_ans[0].append(prob[:, 1])
        tr_ans[1].append(pred)

    te_prob = np.expand_dims(np.concatenate(te_ans[0]), axis=1)
    te_pred = np.expand_dims(np.concatenate(te_ans[1]), axis=1)
    tr_prob = np.expand_dims(np.concatenate(tr_ans[0]), axis=1)
    tr_pred = np.expand_dims(np.concatenate(tr_ans[1]), axis=1)
    print(te_prob.shape)
    print(te_pred.shape)
    prediction_tr = np.hstack([tr_prob, tr_pred])
    prediction_te = np.hstack([te_prob, te_pred])
    pd.DataFrame(prediction_tr, columns=["tr_prob", "tr_pred"]).to_csv("tr.csv")
    pd.DataFrame(prediction_te, columns=['te_prob', "tr_pred"]).to_csv("te.csv")


def predict():
    va_gen.reset()
    te_gen.reset()
    train_acc = 0
    test_acc = 0
    tr_loss = torch.Tensor(1).fill_(0)
    te_loss = torch.Tensor(1).fill_(0)
    n_eval = config['n_eval'] // config['batch_size']
    name = config['name']

    model.eval()
    for i, te_batch in enumerate(te_gen):
        prob = model(te_batch)
        test_acc += evaluate(prob, te_batch['label'])
        te_loss += torch.sum(criterion(prob, te_batch['label'])).detach().cpu()
    te_loss /= (i + 1)

    for i, va_batch in enumerate(va_gen):
        if i == n_eval:
            break
        prob = model(va_batch)
        train_acc += evaluate(prob, va_batch['label'])
        tr_loss += torch.sum(criterion(prob, va_batch['label'])).detach().cpu()
    tr_loss /= (i + 1)

    train_acc /= n_eval
    test_acc /= len(te_gen)
    logger.info("Epoch: " + str(epc))
    logger.info(name + "  train_acc: " + str(train_acc))
    logger.info(name + "  test_acc: " + str(test_acc))
    logger.info(name + "  test_loss:" + str(te_loss.cpu().detach().numpy()))
    add_figure(name, writer, global_step, tr_loss, te_loss, train_acc, test_acc)


if __name__ == "__main__":
    config = set_args()
    global_step = 0
    writer = SummaryWriter(log_dir="figures")

    embedding, config = load_meta(config)
    train_data, train_label = load_data(config['train_path'])
    test_data, test_label = load_data(config['test_path'])

    generator = BatchGen(config, train_data, train_label, config['batch_size'], augment=False)
    tr_gen = BatchGen(config, train_data, train_label, config['batch_size'], is_training=False)
    va_gen = BatchGen(config, train_data, train_label, config['batch_size'])
    te_gen = BatchGen(config, test_data, test_label, config['batch_size'], is_training=False)

    logger.info("load data complete!")
    model = VerifierModel(config, embedding)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 betas=(config['b1'], config['b2']),
                                 eps=config['e'],
                                 weight_decay=config['decay'])
    criterion = nn.CrossEntropyLoss(reduce=False)

    if config['cuda']:
        model.cuda()
    for epc in range(config['epochs']):
        for i, batch in tqdm(enumerate(generator), total=len(generator)):
            global_step += 1
            model.train()
            prob = model(batch)
            train_loss = torch.sum(criterion(prob, batch['label']))
            train_loss.backward()
            if i % 500 == 0:
                print(model.encoder_layer.lexicon_encoder.embedding.weight.grad)
                if config['use_char_emb']:
                    print(model.encoder_layer.lexicon_encoder.char_cnn.embedding_lookup.weight.grad)
            optimizer.step()
            optimizer.zero_grad()
            if i % 500 == 0:
                predict()
        generator.reset()

    writer.export_scalars_to_json("figures/{}.json".format(config['name']))
    writer.close()
    submit(os.path.join(config["submit_dir"], config["name"] + ".csv"))