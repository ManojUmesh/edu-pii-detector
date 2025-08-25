import os
import re
import logging
from typing import List, Iterable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import keras_nlp
import spacy

from config import ModelConfiguration, InputData, configure_logging
from preprocessing import preprocess_data  # returns (words, None, (token_ids, word_ids_map), test_ds)

logging = configure_logging()  # console + file

# Build the SAME model as training

def build_inference_model() -> keras.Model:
    """Rebuild the token-classification model (backbone + dense + softmax)."""
    backbone = keras_nlp.models.DebertaV3Backbone.from_preset(
        ModelConfiguration.preset
    )
    x = backbone.output
    x = keras.layers.Dense(ModelConfiguration.num_labels, name="logits")(x)
    # keep softmax with float32 for numerical stability
    x = keras.layers.Activation("softmax", dtype="float32", name="prediction")(x)
    return keras.Model(backbone.input, x, name="deberta_tokcls")


def load_model_weights_or_model(model: keras.Model) -> keras.Model:
    """Load weights if available; otherwise try loading a full saved model."""
    path = getattr(InputData, "trained_model", None)
    if not path:
        raise ValueError("InputData.trained_model is not set.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights/model not found: {path}")

    lower = path.lower()
    if lower.endswith(".keras") or os.path.isdir(path):
        logging.info("PREDICTING: Loading full model from %s", path)
        model = keras.models.load_model(path, compile=False)
        return model

    logging.info("PREDICTING: Loading weights from %s", path)
    model.load_weights(path)
    return model

# Label → placeholder mapping

# Map BIO tag (string) to a placeholder. Unknowns fall back to <PII>.
BIO_TO_PLACEHOLDER = {
    "B-USERNAME": "<USERNAME>",
    "I-USERNAME": "<USERNAME>",
    "B-ID_NUM": "<ID>",
    "I-ID_NUM": "<ID>",
    "B-PHONE_NUM": "<PHONE>",
    "I-PHONE_NUM": "<PHONE>",
    "B-EMAIL": "<EMAIL>",
    "I-EMAIL": "<EMAIL>",
    "B-STREET_ADDRESS": "<ADDRESS>",
    "I-STREET_ADDRESS": "<ADDRESS>",
    "B-URL_PERSONAL": "<URL>",
    "I-URL_PERSONAL": "<URL>",
    "B-NAME_STUDENT": "<NAME>",
    "I-NAME_STUDENT": "<NAME>",
    "O": "",  # non-PII
}

def label_id_to_placeholder(label_id: int) -> str:
    label_str = ModelConfiguration.id2label.get(int(label_id), "O")
    return BIO_TO_PLACEHOLDER.get(label_str, "<PII>" if label_str != "O" else "")


# ----------------------------
# spaCy helpers & regex fallbacks
# ----------------------------
SPACY_TO_PLACEHOLDER = {
    "PERSON": "<NAME>",
    "ORG": "<ORG>",
    "GPE": "<ADDRESS>",
    "LOC": "<ADDRESS>",
    "FAC": "<ADDRESS>",
    "NORP": "<NAME>",
    "LANGUAGE": "<NAME>",
    # EMAIL/PHONE/URL are not standard in en_core_web_sm;
    # we add regex fallbacks below
}

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
PHONE_RE = re.compile(r"^\+?[0-9(). -]{7,}$")
URL_RE   = re.compile(r"^(https?://|www\.)", re.IGNORECASE)

def build_word_spans(words: List[str]) -> List[Tuple[int, int]]:
    """Return list of (start_char, end_char) for each word in ' '.join(words)."""
    spans = []
    pos = 0
    for w in words:
        start = pos
        end = start + len(w)
        spans.append((start, end))
        pos = end + 1  # +1 for the joining space
    return spans

def spacy_placeholders_for_words(words: List[str], nlp) -> List[str]:
    """
    Run spaCy on reconstructed text and return a list of placeholders per word
    ('' if none). Uses overlap between spaCy entity spans and word character spans.
    Also applies regex fallbacks for EMAIL/PHONE/URL.
    """
    text = " ".join(words)
    doc = nlp(text)
    spans = build_word_spans(words)
    out = [""] * len(words)

    # Mark entities
    for ent in doc.ents:
        ph = SPACY_TO_PLACEHOLDER.get(ent.label_, "")
        if not ph:
            continue
        ent_start, ent_end = ent.start_char, ent.end_char
        for i, (w_start, w_end) in enumerate(spans):
            if not (w_end <= ent_start or w_start >= ent_end):  # overlap?
                out[i] = ph if not out[i] else out[i]

    # Regex fallbacks (apply if not already assigned)
    for i, w in enumerate(words):
        if out[i]:
            continue
        lw = w.lower()
        if EMAIL_RE.match(w):
            out[i] = "<EMAIL>"
        elif URL_RE.match(lw):
            out[i] = "<URL>"
        elif PHONE_RE.match(w) and any(ch.isdigit() for ch in w):
            out[i] = "<PHONE>"

    return out

# Saving helpers

def save_submission(document_ids: List[int],
                    token_positions: List[int],
                    label_ids: List[int],
                    token_strings: List[str]) -> None:
    """Write submission.csv with consistent lengths."""
    logging.info("PREDICTING: Writing submission.csv ...")
    df = pd.DataFrame({
        "document": document_ids,
        "token": token_positions,
        "label_id": label_ids,
        "token_string": token_strings,
    })
    df = df.rename_axis("row_id").reset_index()
    df["label"] = df["label_id"].map(ModelConfiguration.id2label)
    df.to_csv("submission.csv", index=False)

def save_processed_texts(redacted_texts: List[str]) -> None:
    logging.info("PREDICTING: Writing processed_data.csv ...")
    pd.Series(redacted_texts).to_csv("processed_data.csv", index=False, header=False)

# Main prediction routine

def make_predictions(model: keras.Model, processed_test_data):
    """
    processed_test_data (from preprocess_data when train=False) returns:
        words, None, (token_ids, word_ids_map), test_ds
    """
    # Unpack
    words_list, _, packed_ids, test_ds = processed_test_data
    token_ids, word_ids_map = packed_ids  # shapes: (N, seq_len), (N, seq_len)

    # 1) Model inference
    logging.info("PREDICTING: Running inference ...")
    probs = model.predict(test_ds, verbose=1)
    preds = np.argmax(probs, axis=-1)  # (N, seq_len)

    # 2) Post-process per document (align subword → word using word_ids_map)
    logging.info("PREDICTING: Post-processing ...")

    doc_ids: List[int] = []
    token_pos: List[int] = []
    label_ids: List[int] = []
    token_strs: List[str] = []
    redacted_texts: List[str] = []

    # Load spaCy once
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Install with: python -m spacy download en_core_web_sm"
        )

    O_ID = ModelConfiguration.label2id.get("O", 0)

    for doc_idx, (words, y_pred_row, tok_row, wid_row) in enumerate(
        zip(words_list, preds, token_ids, word_ids_map)
    ):
        # valid subword positions
        valid = (tok_row != 0)
        y_pred_row = y_pred_row[valid]
        wid_row = wid_row[valid]

        # FIRST-subword → word label
        word_level_labels = [O_ID] * len(words)
        seen = set()
        for pos, widx in enumerate(wid_row):
            if widx < 0 or widx >= len(words) or widx in seen:
                continue
            seen.add(int(widx))
            word_level_labels[int(widx)] = int(y_pred_row[pos])

        # spaCy placeholders per word
        sp_ph = spacy_placeholders_for_words(words, nlp)

        # Build final tokens (UNION rule: if either flags PII, redact;
        # prefer model placeholder if it exists, else spaCy’s)
        redacted_tokens = []
        for i, w in enumerate(words):
            model_ph = label_id_to_placeholder(word_level_labels[i])  # '' or '<TAG>'
            final_ph = model_ph or sp_ph[i]
            redacted_tokens.append(final_ph if final_ph else w)

        # For submission.csv we still report model-detected PII positions
        pii_word_idx = [i for i, lab in enumerate(word_level_labels) if lab != O_ID]
        for i in pii_word_idx:
            doc_ids.append(doc_idx)
            token_pos.append(i)                # word index as "token" position
            label_ids.append(word_level_labels[i])
            token_strs.append(words[i])

        redacted_texts.append(" ".join(redacted_tokens))

    # 3) Save outputs
    save_submission(doc_ids, token_pos, label_ids, token_strs)
    save_processed_texts(redacted_texts)

    logging.info("PREDICTING: Done!")

# Script entry

if __name__ == "__main__":
    try:
        # Inference mode only
        ModelConfiguration.train = False

        # Build & load model
        model = build_inference_model()
        model = load_model_weights_or_model(model)

        # Prepare data
        processed = preprocess_data(InputData.test)

        # Run predictions
        make_predictions(model, processed)

    except Exception as e:
        logging.error("Prediction failed: %s", e)
        raise
