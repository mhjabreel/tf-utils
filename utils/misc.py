import six
import os
import numpy as np
import tensorflow as tf

from errno import ENOENT

def clone_checkpoint(checkpoint_path_src, output_dir, model_name='model', id_=None):

    tf.logging.info("Listing variables...")

    var_list = tf.train.list_variables(checkpoint_path_src)
    reader = tf.train.load_checkpoint(checkpoint_path_src)
    variables = {}
    latest_step = None
    for name, _ in var_list:
        
        variables[name] = reader.get_tensor(name)
        if name.startswith("global_step"):
            latest_step = variables[name]
    
    latest_step = id_ or latest_step
    
    export_as_checkpoint(variables, latest_step, output_dir, model_name)


def export_as_checkpoint(variables, latest_step, output_dir, model_name='model'):

    if "global_step" in variables:
        del variables["global_step"]
    
    g = tf.Graph()
    with g.as_default():
        tf_vars = []
        for name, value in six.iteritems(variables):
            trainable = True
            dtype = tf.as_dtype(value.dtype)
            tf_vars.append(tf.get_variable(
                name,
                shape=value.shape,
                dtype=dtype))

        placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
        assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
        
        out_base_file = os.path.join(output_dir, model_name)

        global_step = tf.get_variable("global_step",
            initializer=tf.constant(latest_step, dtype=tf.int64),
            trainable=False)
        
        tf_vars.append(global_step)
        
        saver = tf.train.Saver(tf_vars, save_relative_paths=True)

    with tf.Session(graph=g) as sess:
        sess.run(tf.variables_initializer(tf_vars))
        for p, assign_op, value in zip(placeholders, assign_ops, six.itervalues(variables)):
            sess.run(assign_op, {p: value})
        tf.logging.info("\t\tSaving new checkpoint to %s" % output_dir)
        saver.save(sess, out_base_file)#, global_step=latest_step)

    return output_dir

def load_word2index(fname):

    word2id, max_aspect_len, max_context_len = {}, 0, 0
    word2id['<pad>'] = 0
    
    if not os.path.isfile(fname):
        raise IOError(ENOENT, 'Not a file', fname)
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.rstrip().split(' ')
            if len(content) == 3:
                max_aspect_len = int(content[1])
                max_context_len = int(content[2])
            else:
                word2id[content[0]] = int(content[1])
    print('There are %s words in the dataset, the max length of aspect is %s, and the max length of context is %s' % (
    len(word2id), max_aspect_len, max_context_len))
    return word2id, max_aspect_len, max_context_len


def load_word_embeddings(fname, embedding_dim, word2id):
    if not os.path.isfile(fname):
        raise IOError(ENOENT, 'Not a file', fname)

    word2vec = np.random.uniform(-0.01, 0.01, [len(word2id), embedding_dim])
    oov = len(word2id)
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.split(' ')
            if content[0] in word2id:
                word2vec[word2id[content[0]]] = np.array(list(map(float, content[1:])))
                oov = oov - 1
    word2vec[word2id['<pad>'], :] = 0
    print('There are %s words in vocabulary and %s words out of vocabulary' % (len(word2id) - oov, oov))
    return word2vec