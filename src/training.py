import os
import logging
import math
import numpy as np
import tensorflow as tf
import keras
import keras_nlp
import plotly.express as px  # only used if plot=True for LR preview

from config import ModelConfiguration, configure_logging, InputData
from preprocessing import preprocess_data

# Configure logging
logging = configure_logging()

# Custom Loss

class CrossEntropy(keras.losses.Loss):
    """
    Cross-entropy with ignore label (-100) for non-first subwords.
    Assumes model outputs softmax probabilities (from_logits=False).
    """
    def __init__(self, ignore_label=-100, name="xent", **kwargs):
        super().__init__(name=name, **kwargs)
        self.ignore_label = int(ignore_label)

    def call(self, y_true, y_pred):
        # 1) Build mask of valid positions (not the ignore id)
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.not_equal(y_true, self.ignore_label)            # shape (B, T)

        # 2) Make labels safe for the TF op (replace ignored with a valid id, e.g. 0)
        safe_y_true = tf.where(mask, y_true, tf.zeros_like(y_true))

        # 3) Compute per-token loss, then zero out ignored positions and average over the mask
        per_tok = keras.losses.sparse_categorical_crossentropy(
            safe_y_true, y_pred, from_logits=False
        )  # shape (B, T)

        per_tok = tf.where(mask, per_tok, tf.zeros_like(per_tok))
        denom = tf.maximum(tf.reduce_sum(tf.cast(mask, per_tok.dtype)), 1.0)
        return tf.reduce_sum(per_tok) / denom

# Custom Metric

class FBetaScore(keras.metrics.Metric):
    """Micro F-beta over tokens, ignoring -100 and 'O' (index 9) by default."""
    def __init__(self, beta=1.0, name="fbeta_score", ignore_labels=(-100, 9), **kwargs):
        super().__init__(name=name, **kwargs)
        self.beta = float(beta)
        self.ignore_labels = set(ignore_labels)
        self.tp = self.add_weight(name="tp", initializer="zeros", dtype="float32")
        self.fp = self.add_weight(name="fp", initializer="zeros", dtype="float32")
        self.fn = self.add_weight(name="fn", initializer="zeros", dtype="float32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int64)

        mask = tf.ones_like(y_true, dtype=tf.bool)
        for lab in self.ignore_labels:
            mask = tf.logical_and(mask, tf.not_equal(y_true, tf.cast(lab, y_true.dtype)))

        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        equal = tf.equal(y_true, y_pred)
        tp = tf.reduce_sum(tf.cast(equal, tf.float32))
        errors = tf.reduce_sum(tf.cast(tf.logical_not(equal), tf.float32))
        fp = errors
        fn = errors

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        beta2 = self.beta ** 2
        precision = self.tp / (self.tp + self.fp + keras.backend.epsilon())
        recall    = self.tp / (self.tp + self.fn + keras.backend.epsilon())
        return (1.0 + beta2) * precision * recall / (beta2 * precision + recall + keras.backend.epsilon())

    def reset_states(self):
        self.tp.assign(0.0); self.fp.assign(0.0); self.fn.assign(0.0)

# LR Scheduler (cos/exp/step)

def get_lr_callback(batch_size=8, mode='cos', epochs=10, plot=False):
    lr_start, lr_max, lr_min = 6e-6, 2.5e-6 * batch_size, 1e-6
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        elif mode == 'exp':
            lr = (lr_max - lr_min) * (lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep)) + lr_min
        elif mode == 'step':
            lr = lr_max * (lr_decay ** ((epoch - lr_ramp_ep - lr_sus_ep) // 2))
        elif mode == 'cos':
            decay_total_epochs = epochs - lr_ramp_ep - lr_sus_ep + 3
            decay_epoch_index = epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        else:
            lr = lr_max
        return lr

    if plot:
        fig = px.line(
            x=np.arange(epochs),
            y=[lrfn(e) for e in np.arange(epochs)],
            title='LR Scheduler', markers=True,
            labels={'x': 'epoch', 'y': 'lr'},
        )
        fig.update_layout(yaxis=dict(showexponent='all', exponentformat='e'))
        fig.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)

# Model creation

def create_model():
    logging.info("TRAINING: Creating the model ...")
    backbone = keras_nlp.models.DebertaV3Backbone.from_preset(
        ModelConfiguration.preset
    )
    out = backbone.output
    out = keras.layers.Dense(ModelConfiguration.num_labels, name="logits")(out)
    out = keras.layers.Activation("softmax", dtype="float32", name="prediction")(out)
    model = keras.models.Model(backbone.input, out)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-5),
        loss=CrossEntropy(),
        metrics=[FBetaScore()],
    )
    return model

# Training loop

def train_model(processed_train_data, InputData=None):
    """
    Accepts either:
      (train_ds, valid_ds)  OR  (train_ds, valid_ds, class_weights)
    Runs a pre-flight check on one batch, then trains and saves artifacts.
    """
    # Unpack robustly (2 or 3 items)
    if not isinstance(processed_train_data, (list, tuple)):
        raise ValueError("preprocess_data() must return a tuple/list.")

    if len(processed_train_data) == 3:
        train_ds, valid_ds, class_weights = processed_train_data
        logging.info("TRAINING: Received class weights (note: not applied per-token automatically).")
    elif len(processed_train_data) == 2:
        train_ds, valid_ds = processed_train_data
        class_weights = None
    else:
        raise ValueError(f"Unexpected preprocess_data() return of length {len(processed_train_data)}.")

    # LR and model
    lr_cb = get_lr_callback(
        batch_size=ModelConfiguration.train_batch_size,
        mode=ModelConfiguration.lr_mode,
        epochs=ModelConfiguration.epochs,
        plot=False,
    )
    model = create_model()

    # Pre-flight dataset sanity check
    import tensorflow as tf
    card_tr = tf.data.experimental.cardinality(train_ds).numpy()
    card_va = tf.data.experimental.cardinality(valid_ds).numpy()
    logging.info("TRAINING: Dataset cardinality — train batches: %s  valid batches: %s", card_tr, card_va)
    if card_tr <= 0 or card_va <= 0:
        raise RuntimeError("Empty dataset detected (train or valid). Check seq_len/batch sizes and data split.")

    xb, yb = next(iter(train_ds))
    logging.info("TRAINING: One batch shapes — X: %s  y: %s",
                 {k: v.shape for k, v in xb.items()}, yb.shape)
    # quick dry-run
    model.train_on_batch(xb, yb)
    logging.info("TRAINING: Dry-run train_on_batch OK.")

    # Train / or load
    if ModelConfiguration.train:
        logging.info("TRAINING: Training the model ...")
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=ModelConfiguration.epochs,
            callbacks=[lr_cb],
            verbose=1,
        )

        # Evaluate once for logs
        _ = model.evaluate(valid_ds, return_dict=True, verbose=0)

        # SAVE CHECKPOINTS
        save_dir = getattr(InputData, "save_dir", None)
        if not save_dir:
            default_drive = "/content/drive/MyDrive/Colab Notebooks/PII-Data-Detection/models"
            save_dir = default_drive if os.path.isdir("/content/drive") else "models"
        os.makedirs(save_dir, exist_ok=True)
        weights_path = os.path.join(save_dir, "model.weights.h5")
        full_model_path = os.path.join(save_dir, "model_final.keras")

        try:
            model.save_weights(weights_path)
            logging.info("TRAINING: Saved weights to %s", weights_path)
        except Exception as e:
            logging.warning("TRAINING: Could not save weights: %s", e)

        try:
            model.save(full_model_path)
            logging.info("TRAINING: Saved full model to %s", full_model_path)
        except Exception as e:
            logging.warning("TRAINING: Could not save full model: %s", e)

        logging.info("TRAINING: Model is ready to use!")
        return model
    else:
        logging.info("TRAINING: Loading pre-trained model ...")
        if InputData is None or not hasattr(InputData, "trained_model"):
            raise ValueError("ModelConfiguration.train=False but InputData.trained_model is missing.")
        ckpt = InputData.trained_model
        if ckpt and os.path.exists(ckpt):
            model.load_weights(ckpt)
            logging.info("TRAINING: Loaded weights from %s", ckpt)
        else:
            logging.info("TRAINING: No checkpoint found; using fresh weights.")
        logging.info("TRAINING: Model is ready to use!")
        return model

# Script entry point

if __name__ == "__main__":
    try:
        processed = preprocess_data(InputData.train)
        _ = train_model(processed, InputData)
    except Exception as e:
        logging.error("Error: training script failed to run.\n%s", e)
        raise
