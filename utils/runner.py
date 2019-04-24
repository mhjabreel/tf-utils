import tensorflow as tf
from utils.eval import Evaluator, BestCheckpointExporter

tf.logging.set_verbosity(tf.logging.INFO)

class Runner:

    def __init__(self,            
            estimator,
            config,
            eval_hooks=None,
            external_eval_hooks=None,
            session_config=None,
            seed=None):
            
        self._estimator = estimator
        self._config = config
        
        session_config_base = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
        )
        
        if session_config is not None:
            session_config_base.MergeFrom(session_config)

        save_checkpoints_steps = config.get('save_checkpoints_steps', 500)
        keep_checkpoint_max = config.get('keep_checkpoint_max', 5)

        run_config = tf.estimator.RunConfig(
            model_dir=config["model_dir"],
            session_config=session_config_base,
            save_checkpoints_steps=save_checkpoints_steps,
            keep_checkpoint_max=keep_checkpoint_max,
            tf_random_seed=seed)

        ws = config.get("warm_start_from", None)

        self._estimator = tf.estimator.Estimator(
            model_fn=estimator.model_fn(eval_hooks=eval_hooks, external_eval_hooks=external_eval_hooks),
            config=run_config,
            warm_start_from=ws
        )

        self._avg_ckpts = config.get('avg_ckpts', False)
        self._max_avg_ckpts = config.get('max_avg_ckpts', 5)
        self._avg_ckpt_dir = config.get('avg_ckpt_dir', None)
        self._model_dir = config["model_dir"]

        self._export_best_ckpt = config.get('export_best_ckpt', True)
        self._eval_monitor = config.get('_eval_monitor', "Accuracy")

        self._best_model_dir = config.get('best_model_dir', "best_model")

    def _build_train_spec(self, data_layer, checkpoint_path=None):
        train_hooks = None
        train_spec = tf.estimator.TrainSpec(
            input_fn=data_layer,
            max_steps=self._config.get("train_steps"),
            hooks=train_hooks)
        return train_spec

    def _build_eval_spec(self, data_layer, checkpoint_path=None):
        
        eval_spec = tf.estimator.EvalSpec(
            input_fn=data_layer,
            steps=None,
            throttle_secs=0,
            hooks=None)
        return eval_spec             
    
    def train(self, data_layer, checkpoint_path=None, saving_listeners=None):
        """Runs the training loop.
        Args:
            checkpoint_path: The checkpoint path to load the model weights from it.
        """
        if checkpoint_path is not None and tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        train_spec = self._build_train_spec(data_layer, checkpoint_path)

        self._estimator.train(train_spec.input_fn, hooks=train_spec.hooks, max_steps=train_spec.max_steps)
    
    def evaluate(self, data_layer, checkpoint_path=None):
        eval_spec = self._build_eval_spec(data_layer, checkpoint_path)
        return self._estimator.evaluate(eval_spec.input_fn, hooks=eval_spec.hooks, steps=eval_spec.steps, checkpoint_path=checkpoint_path)
    

    def train_and_evaluate(self, train_data_layer, eval_data_layer, checkpoint_path=None):
        if checkpoint_path is not None and tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        train_spec = self._build_train_spec(train_data_layer, checkpoint_path)

        saving_listeners = []

        eval_fn = lambda : self.evaluate(eval_data_layer)
        callbacks = []
        if self._export_best_ckpt:
            bce = BestCheckpointExporter(self._best_model_dir, self._eval_monitor)
            callbacks = [bce]

        saving_listeners.append(Evaluator(eval_fn, self._estimator, callbacks=callbacks))

        self._estimator.train(train_spec.input_fn,
                hooks=train_spec.hooks,
                max_steps=train_spec.max_steps,
                saving_listeners=saving_listeners)
        
        if self._export_best_ckpt:
            return bce.best_results
        return None

    def predict(self, input_fn, checkpoint_path=None):
        predictions = self._estimator.predict(input_fn, checkpoint_path=checkpoint_path)
        return predictions
