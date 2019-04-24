import tensorflow as tf
import six
import abc

@six.add_metaclass(abc.ABCMeta)
class Model:

    def __init__(self, params, scope=None):

        self._params = params
        self._scope = scope       

    @abc.abstractmethod
    def _get_logits(self, inputs, mode=tf.estimator.ModeKeys.TRAIN):
        raise NotImplementedError()
    
    def __call__(self, features, mode=tf.estimator.ModeKeys.TRAIN):
        
        logits = self._get_logits(features, mode)
        return logits

    @abc.abstractmethod
    def get_predictions(self, logits):
        raise NotImplementedError()


@six.add_metaclass(abc.ABCMeta)
class _Classifier(Model):

    def __init__(self, params, reverse_target_vocab=None, scope=None):
        super(_Classifier, self).__init__(params, scope)
        self._reverse_target_vocab = reverse_target_vocab

    def get_predictions(self, logits):
        
        num_classes = self._params.get("num_classes", 2)
        if num_classes == 2:
            proba = tf.sigmoid(logits)
            predictions = tf.round(proba)
        else:
            proba = tf.nn.softmax(logits)
            predictions = tf.argmax(proba, axis=1)
        
        if not self._reverse_target_vocab is None:
            labels = self._reverse_target_vocab.lookup(tf.to_int64(predictions))
            return {'Predictions': predictions, 'Probabilities': proba, 'Labels': labels}
        
        return {'Predictions': predictions, 'Probabilities': proba}
