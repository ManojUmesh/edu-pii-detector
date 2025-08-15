import logging
import math
import plotly.express as px
import numpy as np
import keras
from keras import ops
import keras_nlp

from config import ModelConfiguration, configure_logging, InputData
from preprocessing import preprocess_data

# Configure logging
configure_logging()

# Custom Loss
class CrossEntropy(keras.losses.Loss):
    """Custom Cross Entropy Loss for sequence labeling tasks."""
    def call(self, y_true, y_pred):
        loss = keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        mask = ops.cast(ops.not_equal(y_true, 0), dtype=loss.dtype)
        return ops.sum(loss * mask) / ops.sum(mask)

# Custom Metric
class FBetaScore(keras.metrics.Metric):
    """Custom F-beta score metric for sequence labeling tasks."""
    def __init__(self, beta=1.0, name="fbeta_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.beta = beta
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = ops.argmax(y_pred, axis=-1)
        y_true = ops.cast(y_true, "int64")

        mask = ops.not_equal(y_true, 0)
        y_true = ops.boolean_mask(y_true, mask)
        y_pred = ops.boolean_mask(y_pred, mask)

        tp = ops.sum(ops.cast(ops.equal(y_true, y_pred), "float32"))
        fp = ops.sum(ops.cast(ops.not_equal(y_true, y_pred), "float32"))
        fn = ops.sum(ops.cast(ops.not_equal(y_pred, y_true), "float32"))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        beta_sq = self.beta ** 2
        precision = self.true_positives / (
            self.true_positives + self.false_positives + keras.backend.epsilon()
        )
        recall = self.true_positives / (
            self.true_positives + self.false_negatives + keras.backend.epsilon()
        )
        return (1 + beta_sq) * (precision * recall) / (
            beta_sq * precision + recall + keras.backend.epsilon()
        )

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

# Learning Rate Scheduler
def get_lr_callback():
    """Returns an exponential decay learning rate scheduler."""
    return keras.callbacks.LearningRateScheduler(
        lambda epoch: ModelConfiguration.lr * (0.9 ** epoch)
    )
