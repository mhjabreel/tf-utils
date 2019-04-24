import six
import abc
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class EstimatorBuilder:

    def __init__(self, model, params):

        # self._model_creator = model_creator
        self._params = params
        self._model = model #model_creator(params, None)
    
    def model_fn(self, scope=None, eval_hooks=None, external_eval_hooks=None):
        params = self._params

        def model_fn_impl(features, labels, mode):
            
            self._global_step = tf.train.get_or_create_global_step()

            scaffold = self._params.get("scaffold", None)
            exclude_vars_scope = self._params.get("exclude_vars_scope", None)

            train_op = None
            loss = None
            logits = self._model(features, mode)
            eval_hooks_ = None
            predictions = self._model.get_predictions(logits)
            eval_metric_ops = None
            
            if mode in (tf.estimator.ModeKeys.TRAIN, 
                        tf.estimator.ModeKeys.EVAL):
                
                assert not labels is None
                
                loss = self._get_loss(logits, labels)
                
                if mode == tf.estimator.ModeKeys.TRAIN:
                    optimizer = self._get_optimizer(params)
                    variables = list(tf.trainable_variables())

                    if not exclude_vars_scope is None:
                        exclude_vars = set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=exclude_vars_scope))
                        
                        variables = list(filter(lambda v: not v in exclude_vars, variables))
                    
                    gradients = tf.gradients(loss, variables)
                    #TODO: add gradient clip
                    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self._global_step)
                else:
                    train_op = None    
                    eval_metric_ops = self._evaluate(labels, predictions)           
            else:
                loss = None
            eval_hooks_ = []
            if not eval_hooks is None:
                eval_hooks_.extend(eval_hooks)
            if not external_eval_hooks is None:

                for name, hook in external_eval_hooks:
                    eval_hooks_.append(hook(name, labels, predictions['Predictions']))
            
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=eval_metric_ops,
                evaluation_hooks=eval_hooks_,
                scaffold=scaffold)
        
        return model_fn_impl
        

    def _get_optimizer(self, params):
    
        learning_rate = params.get("learning_rate")
        self._learning_rate = tf.constant(learning_rate)
          
        optimizer_name = params.get("optimizer")

        if optimizer_name == 'rms':
            optimizer = tf.train.RMSPropOptimizer(self._learning_rate,
                                                    params.get("rmsprop_decay"),
                                                    params.get("momentum"),
                                                    params.get("rmsprop_epsilon", 1e-6))
        elif optimizer_name == 'momentum':
            optimizer = tf.train.MomentumOptimizer(self._learning_rate,
                                                    params.get("momentum"))
        elif optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(self._learning_rate)
        else:
            raise ValueError('Invalid value of optimizer: %s' % optimizer_name)

        return optimizer      


    @abc.abstractmethod
    def _get_loss(self, logits, labels):
        raise NotImplementedError()

    @abc.abstractmethod
    def _evaluate(self, y_true, y_pred):
        raise NotImplementedError()        
    

class ClassifierBuilder(EstimatorBuilder):

    def __init__(self, model_creator, params):
        super(ClassifierBuilder, self).__init__(model_creator, params)
        self._num_classes = params.get("num_classes", 2)

    def _get_loss(self, logits, labels):
        
        if self._num_classes == 2:
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=tf.squeeze(logits, 1))
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int32), logits=logits)
        
        class_weights = self._params.get("class_weights", None)

        if class_weights is None:
            loss = tf.reduce_mean(losses)
        else:
            class_weights = tf.constant(class_weights)
            weights = tf.nn.embedding_lookup(class_weights, labels)
            loss = tf.losses.compute_weighted_loss(losses, weights=weights, reduction=tf.losses.Reduction.MEAN)
        
        weight_decay = self._params.get("weight_decay", None)

        if weight_decay and weight_decay > 0:

            loss_wd = (
                weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            )

            tf.summary.scalar('loss_wd', loss_wd)
            loss += loss_wd
        
        tf.summary.scalar("loss", loss)

        return loss

    def _evaluate(self, y_true, y_pred):
        
        if isinstance(y_pred, dict):
            y_pred = y_pred['Predictions']
        
        accuracy = tf.metrics.accuracy(y_true, y_pred)

        eval_metrics = {
            'Accuracy': accuracy
        } 

        return eval_metrics        


class BinaryClassifierBuilder(ClassifierBuilder):
    def __init__(self, model_creator, params):
        super(BinaryClassifierBuilder, self).__init__(model_creator, params)
        self._num_classes = 2

    def _evaluate(self, y_true, y_pred):
        
        if isinstance(y_pred, dict):
            scores = y_pred['Probabilities']
            y_pred = y_pred['Predictions']
        
        accuracy = tf.metrics.accuracy(y_true, y_pred)
        precision = tf.metrics.precision(y_true, y_pred)
        recall = tf.metrics.recall(y_true, y_pred)
        f1_0 = (2 * precision[0] * recall[0]) / (recall[0] + precision[0])
        f1_1 = (2 * precision[1] * recall[1]) / (recall[1] + precision[1])
        auc = tf.metrics.auc(y_true, scores, summation_method='careful_interpolation')

        eval_metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': (f1_0, f1_1),
            'AUC': auc
        } 

        return eval_metrics