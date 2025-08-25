import os
import re
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import keras_nlp
import spacy

from config import ModelConfiguration, InputData, configure_logging
from preprocessing import preprocess_data  # -> (words, None, (token_ids, word_ids_map), test_ds)

logging = configure_logging()

# Model

def build_inference_model() -> keras.Model:
    backbone = keras_nlp.models.DebertaV3Backbone.from_preset(
        ModelConfiguration.preset
    )
    x = backbone.output
    x = keras.layers.Dense(ModelConfiguration.num_labels, name="logits")(x)
    x = keras.layers.Activation("softmax", dtype="float32", name="prediction")(x)
    return keras.Model(backbone.input, x, name="deberta_tokcls")

def load_model_weights_or_model(model: keras.Model) -> keras.Model:
    path = getattr(InputData, "trained_model", None)
    if not path:
        raise ValueError("InputData.trained_model is not set.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights/model not found: {path}")
    lower = path.lower()
    if lower.endswith(".keras") or os.path.isdir(path):
        logging.info("PREDICTING: Loading full model from %s", path)
        return keras.models.load_model(path, compile=False)
    logging.info("PREDICTING: Loading weights from %s", path)
    model.load_weights(path)
    return model

# Labels & placeholders

BIO_TO_PLACEHOLDER = {
    "B-USERNAME": "<USERNAME>", "I-USERNAME": "<USERNAME>",
    "B-ID_NUM": "<ID>", "I-ID_NUM": "<ID>",
    "B-PHONE_NUM": "<PHONE>", "I-PHONE_NUM": "<PHONE>",
    "B-EMAIL": "<EMAIL>", "I-EMAIL": "<EMAIL>",
    "B-STREET_ADDRESS": "<ADDRESS>", "I-STREET_ADDRESS": "<ADDRESS>",
    "B-URL_PERSONAL": "<URL>", "I-URL_PERSONAL": "<URL>",
    "B-NAME_STUDENT": "<NAME>", "I-NAME_STUDENT": "<NAME>",
    "O": ""
}
def label_id_to_placeholder(label_id: int) -> str:
    s = ModelConfiguration.id2label.get(int(label_id), "O")
    return BIO_TO_PLACEHOLDER.get(s, "<PII>" if s != "O" else "")

def label_str_to_id(label_str: str) -> int:
    return ModelConfiguration.label2id.get(label_str, ModelConfiguration.label2id.get("O", 0))

# spaCy + regex

# Only map PERSON → NAME, GPE/LOC/FAC → ADDRESS.
SPACY_TO_BIOLABEL = {
    "PERSON": "B-NAME_STUDENT",
    "GPE": "B-STREET_ADDRESS",
    "LOC": "B-STREET_ADDRESS",
    "FAC": "B-STREET_ADDRESS",
}

# Stricter regex:
EMAIL_TOKEN_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
URL_TOKEN_RE   = re.compile(r"^(https?://|www\.)[^\s]+$", re.IGNORECASE)
# Phone: at least 8 digits overall; allow common separators; avoid pure digit noise by min length 9
PHONE_TOKEN_RE = re.compile(r"^(?=(?:.*\d){8,})\+?[0-9][0-9()\-.\s]{7,}$")

# Light heuristics to reduce FPs
STOPWORDS = {
    "the","a","an","of","and","or","to","in","on","for","at","by","with","from","as",
    "is","are","was","were","be","been","has","have","had","it","this","that"
}
def looks_like_common_word(tok: str) -> bool:
    t = tok.strip()
    if len(t) <= 2: return True
    if t.lower() in STOPWORDS: return True
    if t.isalpha() and t.islower() and len(t) <= 4: return True
    return False

def build_word_spans(words: List[str]) -> List[Tuple[int, int]]:
    spans, pos = [], 0
    for w in words:
        start, end = pos, pos + len(w)
        spans.append((start, end))
        pos = end + 1
    return spans

def spacy_regex_word_labels(words: List[str], nlp) -> List[int]:
    """
    Conservative per-word labels from spaCy + regex.
    Priority: spaCy if PERSON/GPE/LOC/FAC; else regex for EMAIL/URL/PHONE.
    Common words/short words are ignored.
    """
    out = ["O"] * len(words)
    text = " ".join(words)
    doc = nlp(text)
    spans = build_word_spans(words)

    # spaCy entities
    for ent in doc.ents:
        mapped = SPACY_TO_BIOLABEL.get(ent.label_)
        if not mapped:
            continue
        ent_s, ent_e = ent.start_char, ent.end_char
        for i, (w_s, w_e) in enumerate(spans):
            if not (w_e <= ent_s or w_s >= ent_e):
                if not looks_like_common_word(words[i]):
                    out[i] = mapped

    # Regex fallbacks for tokens still 'O'
    for i, w in enumerate(words):
        if out[i] != "O": 
            continue
        lw = w.lower()
        if EMAIL_TOKEN_RE.match(w):
            out[i] = "B-EMAIL"
        elif URL_TOKEN_RE.match(lw):
            out[i] = "B-URL_PERSONAL"
        elif PHONE_TOKEN_RE.match(w) and any(ch.isdigit() for ch in w):
            out[i] = "B-PHONE_NUM"

    return [label_str_to_id(lbl) for lbl in out]

# Fusion policy

# Only let spaCy/regex override if the model is confident about 'O' (non-PII)

O_ID = None
P_O_THRESHOLD = 0.80

def first_subtoken_positions_and_probs(y_probs_row: np.ndarray,
                                       wid_row: np.ndarray,
                                       tok_row: np.ndarray,
                                       num_labels: int,
                                       seq_len: int,
                                       ) -> Tuple[List[int], List[np.ndarray]]:
    """
    For each FIRST subtoken of a word (wid_row gives word indices), collect:
      - its position in the flat sequence
      - the probability vector y_probs_row[pos, :]
    Returns aligned lists of equal length.
    """
    valid = (tok_row != 0)
    probs = y_probs_row[valid]
    wids  = wid_row[valid]
    seen = set()
    positions = []
    prob_vectors = []
    for pos, widx in enumerate(wids):
        if widx < 0:
            continue
        if int(widx) in seen:
            continue
        seen.add(int(widx))
        positions.append(pos)
        prob_vectors.append(probs[pos])
    return positions, prob_vectors

# I/O helpers

def save_submission(doc_ids, token_pos, label_ids, token_strs):
    logging.info("PREDICTING: Writing submission.csv ...")
    df = pd.DataFrame({
        "document": doc_ids,
        "token": token_pos,
        "label_id": label_ids,
        "token_string": token_strs,
    })
    df = df.rename_axis("row_id").reset_index()
    df["label"] = df["label_id"].map(ModelConfiguration.id2label)
    df.to_csv("submission.csv", index=False)

def save_processed_texts(redacted_texts):
    logging.info("PREDICTING: Writing processed_data.csv ...")
    pd.Series(redacted_texts).to_csv("processed_data.csv", index=False, header=False)

# Main

def make_predictions(model: keras.Model, processed_test_data):
    """
    processed_test_data when train=False:
        words, None, (token_ids, word_ids_map), test_ds
    """
    global O_ID
    words_list, _, packed_ids, test_ds = processed_test_data
    token_ids, word_ids_map = packed_ids  # (N, L), (N, L)
    O_ID = ModelConfiguration.label2id.get("O", 0)

    # 1) Predict probabilities
    logging.info("PREDICTING: Running inference ...")
    probs = model.predict(test_ds, verbose=1)  # (N, L, num_labels)
    preds = np.argmax(probs, axis=-1)

    # spaCy once
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Install with: python -m spacy download en_core_web_sm"
        )

    # 2) Per document: model priority + gated overrides
    logging.info("PREDICTING: Post-processing ...")
    doc_ids, token_pos, label_ids, token_strs = [], [], [], []
    redacted_texts = []

    for doc_idx, (words, y_prob_row, y_pred_row, tok_row, wid_row) in enumerate(
        zip(words_list, probs, preds, token_ids, word_ids_map)
    ):
        # Model→word (first-subtoken) labels + P(O) per word
        pos_list, prob_vecs = first_subtoken_positions_and_probs(
            y_prob_row, wid_row, tok_row, ModelConfiguration.num_labels, y_prob_row.shape[0]
        )
        word_labels_model = [O_ID] * len(words)
        word_P_O = [1.0] * len(words)  # default
        # fill from first-subtoken info
        for pos, p in zip(pos_list, prob_vecs):
            widx = wid_row[tok_row != 0][pos]
            if 0 <= widx < len(words):
                word_labels_model[int(widx)] = int(np.argmax(p))
                word_P_O[int(widx)] = float(p[O_ID])

        # spaCy+regex labels
        word_labels_sr = spacy_regex_word_labels(words, nlp)

        # Conservative fusion:
        # - If model != O → keep model.
        # - Else (model == O) → take spaCy/regex only if P(O) >= threshold and SR label != O.
        final_labels = []
        for i in range(len(words)):
            m = word_labels_model[i]
            if m != O_ID:
                final_labels.append(m)
            else:
                sr = word_labels_sr[i]
                if sr != O_ID and word_P_O[i] >= P_O_THRESHOLD and not looks_like_common_word(words[i]):
                    final_labels.append(sr)
                else:
                    final_labels.append(O_ID)

        # Build redacted text
        red_tokens = []
        for i, w in enumerate(words):
            ph = label_id_to_placeholder(final_labels[i])
            red_tokens.append(ph if ph else w)
        redacted_texts.append(" ".join(red_tokens))

        # Submission → all final non-O words
        for i, li in enumerate(final_labels):
            if li != O_ID:
                doc_ids.append(doc_idx)
                token_pos.append(i)        # word index
                label_ids.append(li)
                token_strs.append(words[i])

    save_submission(doc_ids, token_pos, label_ids, token_strs)
    save_processed_texts(redacted_texts)
    logging.info("PREDICTING: Done!")

if __name__ == "__main__":
    try:
        ModelConfiguration.train = False
        model = build_inference_model()
        model = load_model_weights_or_model(model)
        processed = preprocess_data(InputData.test)
        make_predictions(model, processed)
    except Exception as e:
        logging.error("Prediction failed: %s", e)
        raise
