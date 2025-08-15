import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quiet TensorFlow logs

import keras
import keras_nlp
import tensorflow as tf

from preprocessing import preprocess_data
from config import ModelConfiguration, InputData, configure_logging

logging = configure_logging()

def build_model():
    """Minimal DeBERTa-based model for token classification."""
    # Create backbone (inputs/outputs are predefined tensors)
    backbone = keras_nlp.models.DebertaV3Backbone.from_preset(
        ModelConfiguration.preset
    )

    # Use the backbone's graph directly
    x = backbone.output                                  # [batch, seq_len, hidden]
    x = keras.layers.Dense(len(ModelConfiguration.labels), name="logits")(x)
    outputs = keras.layers.Activation("softmax", dtype="float32", name="prediction")(x)

    # Build a model that reuses the backbone's inputs (token_ids, padding_mask)
    model = keras.Model(backbone.input, outputs)
    return model

def train_model():
    logging.info("TRAINING: Preparing datasets...")
    ModelConfiguration.train = True
    train_ds, val_ds = preprocess_data(InputData.train)

    logging.info("TRAINING: Building model...")
    model = build_model()

    logging.info("TRAINING: Compiling model...")
    optimizer = keras.optimizers.Adam()
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metrics = ["accuracy"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    logging.info("TRAINING: Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=ModelConfiguration.epochs
    )

    # Save trained model
    save_path = os.path.join("models", "minimal_model.keras")
    os.makedirs("models", exist_ok=True)
    model.save(save_path)
    logging.info(f"TRAINING: Model saved to {save_path}")

    return history

if __name__ == "__main__":
    train_model()
