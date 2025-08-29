import os
import re
import json
import logging
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import keras_nlp
import spacy
import argparse

from config import ModelConfiguration, InputData, configure_logging
from io_ingest import load_texts
from preprocessing import preprocess_texts_for_inference, preprocess_data

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
    out = ["O"] * len(words)
    text = " ".join(words)
    doc = nlp(text)
    spans = build_word_spans(words)

    for ent in doc.ents:
        mapped = SPACY_TO_BIOLABEL.get(ent.label_)
        if not mapped:
            continue
        ent_s, ent_e = ent.start_char, ent.end_char
        for i, (w_s, w_e) in enumerate(spans):
            if not (w_e <= ent_s or w_s >= ent_e):
                if not looks_like_common_word(words[i]):
                    out[i] = mapped

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

O_ID = None
P_O_THRESHOLD = 0.80

def first_subtoken_positions_and_probs(y_probs_row: np.ndarray,
                                       wid_row: np.ndarray,
                                       tok_row: np.ndarray,
                                       num_labels: int,
                                       seq_len: int,
                                       ) -> Tuple[List[int], List[np.ndarray]]:
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

# Helpers to read gold labels for val split

def _load_tokens_labels_from_json(json_path: str):
    """Reads a JSON/JSONL val file and returns (tokens_list, labels_list)."""
    with open(json_path, "r", encoding="utf-8") as f:
        head = f.read(1024)
        f.seek(0)
        # JSONL?
        if "\n" in head.strip():
            toks, labs = [], []
            for line in f:
                line = line.strip()
                if not line: continue
                row = json.loads(line)
                toks.append(row["tokens"])
                labs.append(row["labels"])
            return toks, labs
        # JSON (list or {"data":[...]})
        data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        toks = [row["tokens"] for row in data]
        labs = [row["labels"] for row in data]
        return toks, labs

# Main

def make_predictions(model: keras.Model,
                     processed_data_tuple,
                     eval_jsonl_out: Optional[str] = None,
                     gold_labels_list: Optional[List[List[str]]] = None):
    """
    processed_data_tuple should be a 4-tuple:
        (words_list, labels_list_or_None, (token_ids, word_ids_map), dataset)

    - If labels_list_or_None is provided (validation), they are written to JSONL as labels_true.
    - If gold_labels_list is provided, it overrides labels_list_or_None.
    - If neither, labels_true is [].
    """
    global O_ID
    words_list, labels_list, packed_ids, dataset = processed_data_tuple
    token_ids, word_ids_map = packed_ids
    O_ID = ModelConfiguration.label2id.get("O", 0)

    logging.info("PREDICTING: Running inference ...")
    probs = model.predict(dataset, verbose=1)  # (N, L, C)
    preds = np.argmax(probs, axis=-1)

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Install with: python -m spacy download en_core_web_sm"
        )

    if gold_labels_list is not None:
        logging.info("PREDICTING: Using gold labels from val.json for labels_true.")
        labels_list = gold_labels_list

    # Sanity checks
    if labels_list is not None:
        assert len(labels_list) == len(words_list), \
            f"Gold labels count ({len(labels_list)}) != sentences ({len(words_list)})"
        for i, (w, g) in enumerate(zip(words_list, labels_list)):
            if len(w) != len(g):
                raise ValueError(f"Sentence {i} mismatch: {len(w)} tokens vs {len(g)} labels")

    logging.info("PREDICTING: Post-processing ...")
    doc_ids, token_pos, label_ids, token_strs = [], [], [], []
    redacted_texts = []
    eval_rows = [] if eval_jsonl_out else None

    for doc_idx, (words, y_prob_row, y_pred_row, tok_row, wid_row) in enumerate(
        zip(words_list, probs, preds, token_ids, word_ids_map)
    ):
        pos_list, prob_vecs = first_subtoken_positions_and_probs(
            y_prob_row, wid_row, tok_row, ModelConfiguration.num_labels, y_prob_row.shape[0]
        )

        word_labels_model = [O_ID] * len(words)
        word_P_O = [1.0] * len(words)
        valid_mask = (tok_row != 0)
        wids_compact = wid_row[valid_mask]
        for pos, p in zip(pos_list, prob_vecs):
            widx = int(wids_compact[pos])
            if 0 <= widx < len(words):
                word_labels_model[widx] = int(np.argmax(p))
                word_P_O[widx] = float(p[O_ID])

        word_labels_sr = spacy_regex_word_labels(words, nlp)

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

        # Redacted text
        red_tokens = []
        for i, w in enumerate(words):
            ph = label_id_to_placeholder(final_labels[i])
            red_tokens.append(ph if ph else w)
        redacted_texts.append(" ".join(red_tokens))

        # Submission rows (final non-O only)
        for i, li in enumerate(final_labels):
            if li != O_ID:
                doc_ids.append(doc_idx)
                token_pos.append(i)
                label_ids.append(li)
                token_strs.append(words[i])

        # JSONL eval row
        if eval_rows is not None:
            pred_labels_str = [ModelConfiguration.id2label[int(li)] for li in final_labels]
            # per-word prob vectors (first-subtoken)
            word_prob_vectors = []
            o_idx = ModelConfiguration.label2id["O"]
            for j in range(len(words)):
                if j < len(prob_vecs):
                    word_prob_vectors.append(prob_vecs[j].tolist())
                else:
                    dummy = [0.0] * ModelConfiguration.num_labels
                    dummy[o_idx] = 1.0
                    word_prob_vectors.append(dummy)

            # labels_true: prefer labels_list if available; else []
            if labels_list is not None:
                labels_true = labels_list[doc_idx]
                # normalize ints -> strings if needed
                if labels_true and isinstance(labels_true[0], int):
                    labels_true = [ModelConfiguration.id2label[int(g)] for g in labels_true]
            else:
                labels_true = []

            eval_rows.append({
                "tokens": words,
                "labels_true": labels_true,
                "labels_pred": pred_labels_str,
                "probs": word_prob_vectors
            })

    # Standard outputs
    save_submission(doc_ids, token_pos, label_ids, token_strs)
    save_processed_texts(redacted_texts)

    # JSONL with probs (and labels_true if available)
    if eval_rows is not None:
        with open(eval_jsonl_out, "w", encoding="utf-8") as f:
            for r in eval_rows:
                f.write(json.dumps(r) + "\n")
        logging.info("PREDICTING: Wrote %s", eval_jsonl_out)

    logging.info("PREDICTING: Done!")

def predict(input_path: Optional[str] = None,
            csv_text_column: str = "text",
            dataset_split: str = "val",
            eval_jsonl_out: Optional[str] = None):
    """
    dataset_split:
      - 'val'       : uses InputData.val (expects gold BIO labels) -> labels_true included
      - 'test'      : uses InputData.test                           -> labels_true empty
      - 'inference' : uses input_path                               -> labels_true empty
    """
    ModelConfiguration.train = False

    model = build_inference_model()
    model = load_model_weights_or_model(model)

    if dataset_split == "inference":
        if not input_path:
            raise ValueError("For 'inference' split, --input_path is required.")
        raw_texts = load_texts(input_path, csv_text_column=csv_text_column)
        processed = preprocess_texts_for_inference(raw_texts)  # -> (words, None, (tok_ids,wids), ds)
        gold_labels = None

    elif dataset_split == "test":
        processed = preprocess_data(InputData.test)            # -> (words, None, (tok_ids,wids), ds)
        gold_labels = None

    else:  # 'val'
        # First, try to get a pipeline output with labels in slot-1:
        processed = preprocess_data(InputData.val)             # -> (words, labels_or_None, (tok_ids,wids), ds)

        # If slot-1 is None or empty, robustly read gold labels directly from val.json:
        need_fallback = (processed[1] is None) or (len(processed[1]) == 0)
        if need_fallback:
            logging.info("PREDICTING: preprocess_data returned no labels; loading gold labels from InputData.val")
            raw_tokens, raw_labels = _load_tokens_labels_from_json(InputData.val)

            # Sanity: sentence counts must match
            words_list = processed[0]
            assert len(words_list) == len(raw_labels), \
                f"val sentences ({len(words_list)}) != gold labels ({len(raw_labels)})"
            # Optional: verify token counts per sentence (can comment out if your pipeline re-tokenizes)
            for i, (w, t) in enumerate(zip(words_list, raw_tokens)):
                if len(w) != len(t):
                    raise ValueError(f"Sentence {i} tokens mismatch: preprocess={len(w)} vs val.json={len(t)}")

            # Replace labels in slot-1
            processed = (processed[0], raw_labels, processed[2], processed[3])
            gold_labels = raw_labels
        else:
            gold_labels = processed[1]

    make_predictions(model, processed, eval_jsonl_out=eval_jsonl_out, gold_labels_list=gold_labels)

    out = {
        "submission_path": os.path.abspath("submission.csv"),
        "processed_path": os.path.abspath("processed_data.csv"),
    }
    if eval_jsonl_out:
        out["eval_jsonl_out"] = os.path.abspath(eval_jsonl_out)
    return out

if __name__ == "__main__":
    try:
        ModelConfiguration.train = False

        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_split", type=str, default="val",
                            choices=["val","test","inference"],
                            help="Choose data source: 'val' includes labels_true; 'test' and 'inference' do not.")
        parser.add_argument("--input_path", type=str, default=None,
                            help="For 'inference' split: path to a .txt/.docx/.pdf/.csv or directory.")
        parser.add_argument("--csv_text_column", type=str, default="text",
                            help="Column name for CSV inference input (default: 'text').")
        parser.add_argument("--eval_jsonl_out", type=str, default="val_predictions_bert.jsonl",
                            help="Where to write JSONL with per-token probabilities (and labels_true when available).")
        args = parser.parse_args()

        model = build_inference_model()
        model = load_model_weights_or_model(model)

        if args.dataset_split == "inference":
            if not args.input_path:
                raise ValueError("For 'inference', provide --input_path.")
            raw_texts = load_texts(args.input_path, csv_text_column=args.csv_text_column)
            processed = preprocess_texts_for_inference(raw_texts)
            gold_labels = None
        elif args.dataset_split == "test":
            processed = preprocess_data(InputData.test)
            gold_labels = None
        else:  # 'val'
            processed = preprocess_data(InputData.val)
            if (processed[1] is None) or (len(processed[1]) == 0):
                logging.info("PREDICTING: preprocess_data returned no labels; loading gold labels from InputData.val")
                raw_tokens, raw_labels = _load_tokens_labels_from_json(InputData.val)
                words_list = processed[0]
                assert len(words_list) == len(raw_labels), \
                    f"val sentences ({len(words_list)}) != gold labels ({len(raw_labels)})"
                for i, (w, t) in enumerate(zip(words_list, raw_tokens)):
                    if len(w) != len(t):
                        raise ValueError(f"Sentence {i} tokens mismatch: preprocess={len(w)} vs val.json={len(t)}")
                processed = (processed[0], raw_labels, processed[2], processed[3])
                gold_labels = raw_labels
            else:
                gold_labels = processed[1]

        make_predictions(model, processed, eval_jsonl_out=args.eval_jsonl_out, gold_labels_list=gold_labels)

    except Exception as e:
        logging.error("Prediction failed: %s", e)
        raise
