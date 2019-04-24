import os
import time
import tensorflow as tf

from bert import tokenization, modeling

from ebsa_bert_model import EBSABertModel

from utils.misc import load_word2index
from utils.data import BertInputter
from utils.estimator import BinaryClassifierBuilder
from utils.runner import Runner

import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')

tf.app.flags.DEFINE_integer('max_aspect_len', 0, 'max length of aspects')
tf.app.flags.DEFINE_integer('max_context_len', 0, 'max length of contexts')
tf.app.flags.DEFINE_string('embedding_matrix', '', 'word ids to word vectors')

batch_size = 128
learning_rate = 0.01
n_epoch = 30

def main(_):
    start_time = time.time()
    

    tokenizer = tokenization.FullTokenizer("bert_models/uncased_L-12_H-768_A-12/vocab.txt")
    bert_configs = modeling.BertConfig.from_json_file("bert_models/uncased_L-12_H-768_A-12/bert_config.json")

    print('Loading data info ...')
    _, max_aspect_len, max_context_len = load_word2index('data/vocab.txt')
    inputter = BertInputter(tokenizer, max_aspect_len + max_context_len + 3)

    params = {
        "reduction_mode": 'conv',
        "num_hidden_layers": 6
    }
    
    bert_ckp_path = "bert_models/uncased_L-12_H-768_A-12/bert_model.ckpt"

    model_config = {
        'optimizer': 'adam', 
        'learning_rate': learning_rate, 
        'weight_decay': FLAGS.l2_reg,
        'exclude_vars_scope': 'bert/'
    }

    ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=bert_ckp_path,
                        vars_to_warm_start="bert*")

    eval_metrics = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1': [],
        'AUC': []
    } 

    for k in range(1, 6):
        
        print('Loading training data and testing data ...')
        train_data = inputter.get_input_fn('data/k5folds/train_k%d.tsv' % k, batch_size=batch_size, repeat=True)
        test_data = inputter.get_input_fn('data/k5folds/dev_k%d.tsv' % k, batch_size=batch_size, repeat=False, shuffle=False)
        
        best_model_dir = 'ckpts/bert_based_logically_ebsa_best_model_k%d' % k
        model_dir = 'ckpts/bert_based_logically_ebsa_k_%d' % k
        
        run_config = {
            'model_dir': model_dir,
            'best_model_dir': best_model_dir, 
            'train_steps': 2400, 
            'save_checkpoints_steps': 200, 
            'warm_start_from': ws
        }

        print("Training .... ")
        
        model = EBSABertModel(params, bert_configs)

        estimator = BinaryClassifierBuilder(model, model_config)

        runner = Runner(estimator, run_config)
    
        results = runner.train_and_evaluate(train_data, test_data)

        if not results is None:
            print(results)
            for k in results:
                if k in eval_metrics:
                    eval_metrics[k] += [results[k]]

    end_time = time.time()
    print('Time Costing: %s' % (end_time - start_time))

    for k in eval_metrics:
        if len(eval_metrics[k]) != 0:
            res = eval_metrics[k]
            print("{}: {:.3f}(±{:.3f})".format(k, np.mean(res), np.std(res)))


if __name__ == '__main__':
    tf.app.run()