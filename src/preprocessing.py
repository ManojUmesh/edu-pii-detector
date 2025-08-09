import json
import numpy as np
import tensorflow as tf
import keras
import keras_nlp
from sklearn.model_selection import train_test_split
from config import ModelConfiguration, configure_logging

# Setup logging
logging = configure_logging()

# -----------------------------
# Step 1: Read raw data
# -----------------------------
def read_data(data_path):
    """
    Reads JSON file and converts tokens and labels into NumPy arrays.
    Labels are mapped to integer IDs based on ModelConfiguration.label2id.
    """
    logging.info(f"Reading data from {data_path} ...")
    with open(data_path, "r") as f:
        data = json.load(f)

    words = np.empty(len(data), dtype=object)
    labels = np.empty(len(data), dtype=object)

    for i, sample in enumerate(data):
        words[i] = np.array(sample["tokens"])
        labels[i] = np.array([ModelConfiguration.label2id[label] for label in sample["labels"]])

    return words, labels

# -----------------------------
# Step 2: Tokenizer configuration
# -----------------------------
def tokenizer_conf():
    """
    Loads the DeBERTa tokenizer from preset defined in ModelConfiguration.
    """
    logging.info("Creating tokenizer ...")
    return keras_nlp.models.DebertaV3Tokenizer.from_preset(ModelConfiguration.preset)

# -----------------------------
# Step 3: Minimal dataset builder
# -----------------------------
def build_dataset(words, labels=None, batch_size=4, seq_len=128, shuffle=False):
    """
    Converts words (and labels if provided) into a tf.data.Dataset.
    Minimal version: joins words into a string, tokenizes, pads/truncates.
    No subword alignment yet.
    """
    tokenizer = tokenizer_conf()

    def encode(example):
        # Join words into a single string
        text = tf.strings.reduce_join(example["words"], separator=" ")

        # Tokenize text → dense tensor
        ids = tokenizer(text)
        if isinstance(ids, tf.RaggedTensor):
            ids = ids.to_tensor()
        ids = tf.squeeze(ids)

        # Truncate and pad to seq_len
        ids = ids[:seq_len]
        pad_len = tf.maximum(0, seq_len - tf.shape(ids)[0])
        ids = tf.pad(ids, paddings=[[0, pad_len]])

        inputs = {"token_ids": ids}

        if labels is not None:
            lbl = example["labels"]
            lbl = lbl[:seq_len]
            lbl = tf.pad(lbl, paddings=[[0, tf.maximum(0, seq_len - tf.shape(lbl)[0])]],
                         constant_values=-100)
            return inputs, lbl

        return inputs

    slices = {"words": tf.ragged.constant(words)}
    if labels is not None:
        slices["labels"] = tf.ragged.constant(labels)

    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(encode, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1024, seed=ModelConfiguration.seed)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# -----------------------------
# Step 4: Wrapper function
# -----------------------------
def preprocess_data(data_path):
    """
    Main preprocessing entry point.
    Returns train/valid datasets if in training mode.
    """
    keras.utils.set_random_seed(ModelConfiguration.seed)

    words, labels = read_data(data_path)

    if ModelConfiguration.train:
        train_words, val_words, train_labels, val_labels = train_test_split(
            words, labels, test_size=0.3, random_state=ModelConfiguration.seed
        )
        train_ds = build_dataset(train_words, train_labels, 
                                 batch_size=ModelConfiguration.train_batch_size,
                                 seq_len=ModelConfiguration.train_seq_len,
                                 shuffle=True)
        val_ds = build_dataset(val_words, val_labels, 
                               batch_size=ModelConfiguration.train_batch_size,
                               seq_len=ModelConfiguration.train_seq_len)
        return train_ds, val_ds
    else:
        test_ds = build_dataset(words, labels=None, 
                                batch_size=ModelConfiguration.infer_batch_size,
                                seq_len=ModelConfiguration.infer_seq_len)
        return words, None, None, test_ds
