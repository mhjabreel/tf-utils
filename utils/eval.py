import tensorflow as tf
from utils.misc import clone_checkpoint


class Evaluator(tf.train.CheckpointSaverListener):

    def __init__(self, eval_fn, estimator, callbacks=None, **kwargs):

        self._eval_fn = eval_fn
        self._callbacks = callbacks
        self._estimator = estimator

    
    def after_save(self, session, global_step_value):
        tf.logging.info("Step[%d]: Done writing checkpoint.", global_step_value)
        tf.logging.info("Step[%d]: Start evaluating the model.", global_step_value)
        res = self._eval_fn()
        if not self._callbacks is None:
            for clpk in self._callbacks:
                clpk(res, global_step_value, self._estimator.latest_checkpoint())


class BestCheckpointExporter:

    def __init__(self, output_dir, monitor='Accuracy', compare_fn=lambda x, y: y is None or x > y):

        self._monitor = monitor
        self._compare_fn = compare_fn
        self._output_dir = output_dir
        self._best_val = None
        self._best_results = None

        if not tf.gfile.IsDirectory(output_dir):
            tf.gfile.MakeDirs(output_dir)

    def __call__(self, results, global_step_value, check_pointpath):
        val = results[self._monitor]
        if self._compare_fn(val, self._best_val):
            old_val = "INF" if self._best_val is None else "%.4f" % (self._best_val)
            tf.logging.info("Step[%d]: Improvment in %s from %s to %.4f.", global_step_value, self._monitor, old_val, val)
            tf.logging.info("Step[%d]: Export the best checkpoint to %s.", global_step_value, self._output_dir)
            
            clone_checkpoint(check_pointpath, self._output_dir)
            
            self._best_val = val
            self._best_results = results
    
    @property
    def best_results(self):
        return self._best_results