import tensorflow as tf
import numpy as np


class BertInputter:
    
    def __init__(self, tokenizer, max_len):
        
        
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._features = None
        self._labels = None
    
    
    def _tokenize(self, sentence, aspect):
        
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0

        if tf.contrib.framework.is_tensor(sentence):
            sentence = tf.compat.as_text(sentence.numpy())
            
        if tf.contrib.framework.is_tensor(aspect):
            aspect = tf.compat.as_text(aspect.numpy())        
        
        
        tokens_a = self._tokenizer.tokenize(sentence)
        tokens_b = self._tokenizer.tokenize(aspect)

        n = len(tokens_a) + len(tokens_b) + 3
        if n > self._max_len:
            k = self._max_len - (3 + len(tokens_b))
            tokens_a = tokens_a[:k]
            
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

        ids = self._tokenizer.convert_tokens_to_ids(tokens)

        mask = [1] * len(ids)

        # Zero-pad up to the sequence length.

        while len(ids) < self._max_len:
            ids.append(0)
            mask.append(0)
            type_ids.append(0) 

        assert len(ids) == self._max_len
        assert len(mask) == self._max_len
        assert len(type_ids) == self._max_len 

        return np.array(ids, dtype=np.int32), \
            np.array(type_ids, dtype=np.int32), \
            np.array(mask, dtype=np.int32)   
    
    def get_input_fn(self, 
                     data_source, 
                     labelled=True, 
                     shuffle=True, 
                     reshuffle_each_iteration=True,
                     repeat=False, 
                     skip_rows=1,
                     buffer_size=None,
                     batch_size=100,
                     drop_remainder=False,
                     num_parallel_calls=None,
                     seed=None):
        
        def sentiment_to_polarity(sentiment):
            
            if tf.contrib.framework.is_tensor(sentiment):
                sentiment = tf.compat.as_text(sentiment.numpy())            
            if sentiment == 'positive':
                return 1.0
            
            return 0.0
        
        def parse_row(line):
            
            values = tf.strings.split([line], sep='\t').values
            
            tout = [tf.int32, tf.int32, tf.int32]
            ids, segments, masks = tf.py_function(self._tokenize, 
                                                  [values[0], values[1]],
                                                  Tout=tout)
            features = {
                'ids': tf.reshape(ids, [self._max_len]),
                'segments': tf.reshape(segments, [self._max_len]),
                'masks': tf.reshape(masks, [self._max_len])
            }
            
            if labelled:
                polarity = tf.py_function(sentiment_to_polarity, 
                                          [values[2]], 
                                          Tout=tf.float32)                
                return features, polarity
            
            return features
        
        buffer_size = buffer_size or batch_size * 20
        
        def input_fn_impl():
            dataset = tf.data.TextLineDataset(data_source).skip(skip_rows)
            
            if shuffle:
                dataset = dataset.shuffle(buffer_size, seed, 
                                        reshuffle_each_iteration)
            
            dataset = dataset.map(parse_row, num_parallel_calls=num_parallel_calls)
            
            if repeat:
                dataset = dataset.repeat()
            
            dataset = dataset.batch(batch_size, drop_remainder)
            iterator = dataset.make_initializable_iterator()
            
            tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
            next_batch = iterator.get_next()

            return next_batch

        return input_fn_impl
