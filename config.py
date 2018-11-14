import argparse
from os.path import expanduser, join


def set_args():
    source_data_dir = join(expanduser("~"), "xyx", "data")
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="no_att_drop")
    parser.add_argument('--train_path', type=str, default="resource/addexm_train_input.txt")
    parser.add_argument('--test_path', type=str, default='resource/addexm_test_input.txt')
    parser.add_argument('--meta_path', type=str, default="resource/meta.pkl")
    parser.add_argument('--cove_path', default=source_data_dir + '/MT-LSTM.pt')
    parser.add_argument('--submit_dir', type=str, default="submission")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)

    parser.add_argument('--embed_size', type=int, default=300)  # glove 300d
    parser.add_argument('--char_emb_dim', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=300)

    parser.add_argument('--lstm_layers', type=int, default=1)

    parser.add_argument('--dropout_emb', type=float, default=0.3)
    parser.add_argument('--dropout_attn', type=float, default=0.3)
    parser.add_argument('--dropout_resid', type=float, default=0.1)
    parser.add_argument('--dropout_lstm', type=float, default=0)
    parser.add_argument('--dropout_cnn', type=float, default=0.1)
    parser.add_argument('--dropout_cov', type=float, default=0.4)

    parser.add_argument('--doc_maxlen', type=int, default=100)
    parser.add_argument('--query_maxlen', type=int, default=40)
    parser.add_argument('--word_maxlen', type=int, default=15)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--lr', type=float, default=4e-4) # 8e-4
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--grad_clipping', type=float, default=5)

    parser.add_argument('--tune_oov', action='store_true', default=True)
    parser.add_argument('--fix_embedding', action='store_true', default=False)
    parser.add_argument('--use_char_emb', action='store_true', default=False)
    parser.add_argument('--use_char_rnn', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--n_eval', type=int, default=512)
    config = parser.parse_args().__dict__
    return config
