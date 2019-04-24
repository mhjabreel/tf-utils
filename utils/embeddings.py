import h5py
import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self,
                vocab_size,
                embedding_size,
                trainable=False,
                initializer_range=0.02, 
                weights=None,
                dtype=tf.float32
                ):

        super().__init__()

        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._trainable = trainable
        self._initializer_range = initializer_range
        self._weights = weights

        if not weights is None:
            weights = weights.astype(dtype.as_numpy_dtype())
            initializer = tf.keras.initializers.constant(weights, dtype=dtype)
        else:
            initializer = tf.keras.initializers.uniform(-initializer_range, initializer_range)

        self.embedding_table = self.add_weight(
            "embeddings",
            shape=[vocab_size, embedding_size],
            dtype=dtype,
            trainable=trainable,
            initializer=initializer
        )        
    
    def build(self, input_shapes):
        self.built = True

    def call(self, x):
        print(x)
        return tf.nn.embedding_lookup(self.embedding_table, x)
    
    @classmethod
    def from_pretrained(cls, weights, trainable=False, dtype=tf.float32):
        print(cls)
        return cls(*weights.shape, trainable=trainable, dtype=dtype, weights=weights)
    
    @property
    def vocab_size(self):
        return self._vocab_size
    
    @property
    def embedding_size(self):
        return self._embedding_size


class TokenTypePostProcessor(EmbeddingLayer):

    def call(self, x, type_ids):

        seq_length = tf.shape(x)[1]

        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_type_ids = tf.reshape(type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_type_ids, depth=self.vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, self.embedding_table)
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                        [-1, seq_length, self.embedding_size])
        return x + token_type_embeddings


class PositionEmbeddingPostprocessor(EmbeddingLayer):
    
    def call(self, x):
        seq_length = tf.shape(x)[1]
        position_embeddings = tf.slice(self.embedding_table, [0, 0],
                                        [seq_length, -1])
        num_dims = len(x.shape.as_list())

        # Only the last two dimensions are relevant (`seq_length` and `width`), so
        # we broadcast among the first dimensions, which is typically just
        # the batch size.
        position_broadcast_shape = []
        for _ in range(num_dims - 2):
            position_broadcast_shape.append(1)
        position_broadcast_shape.extend([seq_length, self.embedding_size])

        position_embeddings = tf.reshape(position_embeddings,
                                        position_broadcast_shape)  
    
        return x + position_embeddings


class BertEmbedding(tf.keras.Model):

    def __init__(self,
        weights_file, 
        use_token_type_embeddings=True, 
        use_position_embeddings=True, 
        normalize=True,
        train_word_embeddings=False,
        train_token_types=False,
        train_position_embeddings=False,
        train_norm_layer=False,
        dropout_prob=0.1):

        super().__init__()

        with h5py.File(weights_file, 'r') as fin:
            embedding_weights = fin["/embeddings/word_embeddings"][...]
            self._word_embeddings = EmbeddingLayer.from_pretrained(embedding_weights, 
                dtype=tf.dtypes.as_dtype(embedding_weights.dtype),
                trainable=train_word_embeddings)
            
            if normalize:
                beta = fin["/embeddings/LayerNorm/beta"][...]
                gamma = fin["/embeddings/LayerNorm/gamma"][...]

                beta_initializer = tf.keras.initializers.constant(beta)
                gamma_initializer = tf.keras.initializers.constant(gamma)

                self._norm_layer = tf.keras.layers.BatchNormalization(
                    beta_initializer=beta_initializer, 
                    gamma_initializer=gamma_initializer,
                    trainable=train_norm_layer)

            self._dropout = tf.keras.layers.Dropout(dropout_prob)

            if use_token_type_embeddings:

                embedding_weights = fin["/embeddings/token_type_embeddings"][...]
                self._token_type_embeddings = EmbeddingLayer.from_pretrained(embedding_weights, 
                    dtype=tf.dtypes.as_dtype(embedding_weights.dtype),
                    trainable=train_token_types)

            if use_token_type_embeddings:

                embedding_weights = fin["/embeddings/token_type_embeddings"][...]
                self._token_type_embeddings = TokenTypePostProcessor.from_pretrained(embedding_weights, 
                    dtype=tf.dtypes.as_dtype(embedding_weights.dtype),
                    trainable=train_token_types)

            if use_position_embeddings:

                embedding_weights = fin["/embeddings/position_embeddings"][...]
                self._position_embeddings = PositionEmbeddingPostprocessor.from_pretrained(embedding_weights, 
                    dtype=tf.dtypes.as_dtype(embedding_weights.dtype),
                    trainable=train_position_embeddings)  

        self._use_token_type_embeddings = use_token_type_embeddings
        self._use_position_embeddings = use_position_embeddings
        self._normalize = normalize
    
    def call(self, ids, token_type_ids=None):

        output = self._word_embeddings(ids)

        if self._use_token_type_embeddings:
            assert not token_type_ids is None
            output = self._token_type_embeddings(output, token_type_ids)
        
        if self._use_position_embeddings:
            output = self._position_embeddings(output)
        
        if self._normalize:
            output = self._norm_layer(output)
        
        output = self._dropout(output)

        return output