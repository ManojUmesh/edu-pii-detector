import json
from typing import List, Tuple, Optional
from collections import Counter

import numpy as np
import tensorflow as tf
import keras
import keras_nlp
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from config import ModelConfiguration, InputData, configure_logging

# Logging & (optional) mixed precision

logging = configure_logging()
try:
    keras.mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

# I/O: read JSON data

def read_data(path: str) -> Tuple[List[List[str]], Optional[List[List[str]]]]:
    """
    Read a JSON file that contains a list of records.
      Train JSON: [{"tokens": [...], "labels": [...]}, ...]
      Test  JSON: [{"tokens": [...]}, ...]  (no "labels" key)
    Returns:
      words:  List[List[str]]
      labels: List[List[str]] | None
    """
    logging.info("PREPROCESSING: Reading data ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words: List[List[str]] = []
    labels: Optional[List[List[str]]] = []

    has_labels = isinstance(data, list) and len(data) > 0 and ("labels" in data[0])

    for x in tqdm(data, total=len(data)):
        words.append(x["tokens"])
        if has_labels:
            labels.append(x["labels"])

    if not has_labels:
        labels = None

    return words, labels

# Tokenizer

def tokenizer_conf():
    """Create tokenizer from preset."""
    logging.info("PREPROCESSING: Creating tokenizer ...")
    return keras_nlp.models.DebertaV3Tokenizer.from_preset(ModelConfiguration.preset)

# Label mapping & alignment helpers

def map_labels_to_ids(label_seq: List[str]) -> List[int]:
    """Map string labels to ids; unknowns fall back to 'O'."""
    l2i = ModelConfiguration.label2id
    o_id = l2i.get("O", 0)
    return [l2i.get(lbl, o_id) for lbl in label_seq]


def words_to_subtokens_and_labels(
    words: List[str],
    word_label_ids: List[int],
    tokenizer,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a sentence (words + word-level label ids) into:
      token_ids:           (seq_len,)  -> vocab piece ids
      token_labels:        (seq_len,)  -> -100 on non-first subwords/pad
      word_ids_per_token:  (seq_len,)  -> word index per subword, -1 for pad
    """
    def to_python_list(batch_tokens):
        # RaggedTensor path
        if hasattr(batch_tokens, "to_list"):
            data = batch_tokens.to_list()
            return data[0] if len(data) else []
        # Tensor path
        if tf.is_tensor(batch_tokens):
            arr = batch_tokens.numpy()
            if arr.ndim == 2:
                return arr[0].tolist()
            return arr.tolist()
        # Fallback
        arr = np.array(batch_tokens, dtype=object)
        if arr.ndim == 2:
            return arr[0].tolist()
        return arr.tolist()

    subtoken_ids: List[int] = []
    subtoken_labels: List[int] = []
    word_ids_per_token: List[int] = []

    for w_idx, (w, lab_id) in enumerate(zip(words, word_label_ids)):
        pieces_rt = tokenizer([w])     # tokenize single word
        pieces = to_python_list(pieces_rt)
        if not pieces:
            continue

        subtoken_ids.extend(pieces)
        # first subword of this word gets the label; others -100
        subtoken_labels.append(lab_id)
        if len(pieces) > 1:
            subtoken_labels.extend([-100] * (len(pieces) - 1))
        # record the source word index for each subword
        word_ids_per_token.extend([w_idx] * len(pieces))

    # truncate
    subtoken_ids = subtoken_ids[:seq_len]
    subtoken_labels = subtoken_labels[:seq_len]
    word_ids_per_token = word_ids_per_token[:seq_len]

    # pad
    pad = seq_len - len(subtoken_ids)
    if pad > 0:
        subtoken_ids.extend([0] * pad)              # 0 is PAD for DeBERTa preset
        subtoken_labels.extend([-100] * pad)        # ignored by loss/metric
        word_ids_per_token.extend([-1] * pad)       # -1 marks pad/special

    return (
        np.array(subtoken_ids, dtype="int32"),
        np.array(subtoken_labels, dtype="int32"),
        np.array(word_ids_per_token, dtype="int32"),
    )

# Corpus -> fixed arrays

def build_arrays(
    all_words: List[List[str]],
    all_labels: Optional[List[List[str]]],
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert whole corpus into arrays:
      token_ids:        (N, seq_len)
      token_labels:     (N, seq_len)
      padding_mask:     (N, seq_len)  (1 for real tokens, 0 for pads)
      word_ids_per_tok: (N, seq_len)  (word index per subword, -1 for pads)
    """
    tokenizer = tokenizer_conf()

    N = len(all_words)
    token_ids = np.zeros((N, seq_len), dtype="int32")
    token_labels = np.full((N, seq_len), fill_value=-100, dtype="int32")
    padding_mask = np.zeros((N, seq_len), dtype="int32")
    word_ids_map = np.full((N, seq_len), fill_value=-1, dtype="int32")

    logging.info("PREPROCESSING: Getting token ids ...")
    for i in tqdm(range(N)):
        words = all_words[i]
        if all_labels is None:
            # inference path: use "O" over all words for shape compatibility
            label_ids = [ModelConfiguration.label2id.get("O", 0)] * len(words)
        else:
            label_ids = map_labels_to_ids(all_labels[i])

        ids_row, lab_row, wids_row = words_to_subtokens_and_labels(
            words, label_ids, tokenizer, seq_len
        )
        token_ids[i] = ids_row
        token_labels[i] = lab_row
        padding_mask[i] = (ids_row != 0).astype("int32")
        word_ids_map[i] = wids_row

    return token_ids, token_labels, padding_mask, word_ids_map

# tf.data Dataset builders

def as_tf_dataset(
    token_ids: np.ndarray,
    token_labels: np.ndarray,
    padding_mask: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """
    Yields: ({'token_ids': ..., 'padding_mask': ...}, labels)
    """
    x = {"token_ids": token_ids, "padding_mask": padding_mask}
    y = token_labels

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(token_ids), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def as_tf_dataset_infer(
    token_ids: np.ndarray,
    padding_mask: np.ndarray,
    batch_size: int,
) -> tf.data.Dataset:
    x = {"token_ids": token_ids, "padding_mask": padding_mask}
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Class-weight computation (token-level, ignoring -100)

def compute_class_weights(label_ds: tf.data.Dataset, num_labels: int, ignore_label: int = -100):
    """
    Estimate class weights from a dataset of (x, y) where y is (B, T).
    Returns a np.ndarray of shape (num_labels,) normalized to mean ~ 1.
    """
    logging.info("PREPROCESSING: Computing class weights (approx) ...")
    counts = Counter()
    for _, y in label_ds.take(200):
        y = y.numpy().reshape(-1)
        y = y[y != ignore_label]
        counts.update(y.tolist())

    freq = np.ones(num_labels, dtype=np.float32)
    for k, v in counts.items():
        if 0 <= int(k) < num_labels:
            freq[int(k)] = float(v)

    inv = 1.0 / np.sqrt(freq)
    inv = inv * (num_labels / inv.sum())
    return inv


# Orchestration

def preprocess_data(data_path: str):
    """
    If ModelConfiguration.train:
        returns (train_ds, val_ds, class_weights)
    else:
        returns (words, None, (token_ids, word_ids_map), test_ds)
    """
    # lengths / batch sizes
    seq_len_train = getattr(ModelConfiguration, "train_seq_len", 1024)
    bs_train = getattr(ModelConfiguration, "train_batch_size", 8)
    seq_len_val = getattr(ModelConfiguration, "val_seq_len", seq_len_train)
    bs_val = getattr(ModelConfiguration, "val_batch_size", bs_train)

    # inference overrides if provided
    if not ModelConfiguration.train:
        seq_len_val = getattr(ModelConfiguration, "infer_seq_len", seq_len_val)
        bs_val = getattr(ModelConfiguration, "infer_batch_size", bs_val)

    words, labels = read_data(data_path)

    if ModelConfiguration.train and labels is not None:
        logging.info("PREPROCESSING: Building dataset ...")
        w_tr, w_va, l_tr, l_va = train_test_split(
            words, labels, test_size=0.2, random_state=ModelConfiguration.seed, shuffle=True
        )

        tr_ids, tr_labs, tr_mask, _ = build_arrays(w_tr, l_tr, seq_len_train)
        va_ids, va_labs, va_mask, _ = build_arrays(w_va, l_va, seq_len_val)

        train_ds = as_tf_dataset(tr_ids, tr_labs, tr_mask, bs_train, shuffle=True)
        val_ds   = as_tf_dataset(va_ids, va_labs, va_mask, bs_val, shuffle=False)

        class_weights = compute_class_weights(train_ds, ModelConfiguration.num_labels, ignore_label=-100)

        return train_ds, val_ds, class_weights

    # Inference path (labels can be None)
    logging.info("PREPROCESSING: Building dataset ...")
    te_ids, _te_labs, te_mask, te_word_ids = build_arrays(words, labels, seq_len_val)
    test_ds = as_tf_dataset_infer(te_ids, te_mask, bs_val)

    # Return a tuple compatible with predict.py, plus the word-index map
    return words, None, (te_ids, te_word_ids), test_ds


# Quick check

if __name__ == "__main__":
    ModelConfiguration.train = True
    train_ds, val_ds, cw = preprocess_data(InputData.train)
    xb, yb = next(iter(train_ds))
    logging.info("Train batch token_ids shape: %s", xb["token_ids"].shape)
    logging.info("Train batch labels shape: %s", yb.shape)
    logging.info("Class weights: %s", cw)
