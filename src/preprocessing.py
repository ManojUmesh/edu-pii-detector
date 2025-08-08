import json
import numpy as np
import tensorflow as tf
import keras
import keras_nlp
import keras_nlp
from sklearn.model_selection import train_test_split
from config import ModelConfiguration, configure_logging

# Setup logging
logging = configure_logging()

# Step 1: Read raw data
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

# Step 2: Tokenizer configuration
def tokenizer_conf():
    """
    Loads the DeBERTa tokenizer from preset defined in ModelConfiguration.
    """
    logging.info("Creating tokenizer ...")
    return keras_nlp.models.DebertaV3Tokenizer.from_preset(ModelConfiguration.preset)

# Step 3: Simple tokenization and dataset creation
def build_dataset(words, labels=None, batch_size=4, seq_len=128, shuffle=False):
    """
    Converts words (and labels if provided) into a tf.data.Dataset.
    This version is minimal — no special tokens or complex alignment yet.
    """
    logging.info("Building dataset ...")
    tokenizer = tokenizer_conf()

    def encode(example):
        tokens = tokenizer(example["words"]).to_tensor(shape=(seq_len,))
        if labels is not None:
            return {"token_ids": tokens}, example["labels"]
        return {"token_ids": tokens}

    slices = {"words": tf.ragged.constant(words)}
    if labels is not None:
        slices["labels"] = tf.ragged.constant(labels)

    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(encode, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1024, seed=ModelConfiguration.seed)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Step 4: Wrapper function
def preprocess_data(data_path):
    """
    Main preprocessing entry point.
    Returns train/valid datasets if in training mode,
    otherwise returns dataset for inference.
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
