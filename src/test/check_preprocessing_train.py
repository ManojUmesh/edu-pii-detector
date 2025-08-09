import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quiet TF logs

from preprocessing import preprocess_data
from config import ModelConfiguration, InputData

# --- CONFIG ---
ModelConfiguration.train = True

# --- RUN ---
train_ds, val_ds = preprocess_data(InputData.train)

# --- CHECK ONE BATCH ---
x, y = next(iter(train_ds))
print("token_ids shape:", x["token_ids"].shape)      # (batch, seq_len)
print("padding_mask shape:", x["padding_mask"].shape)
print("labels shape:", y.shape)

# --- BASIC ASSERTIONS ---
B, L = x["token_ids"].shape
assert y.shape == (B, L), "labels must align with token_ids"
assert x["padding_mask"].dtype == bool or x["padding_mask"].dtype == "bool", "padding_mask must be boolean"
assert (y.numpy() == -100).any(), "labels should contain -100 for non-start subwords/pad"
print("✔ training-mode preprocessing basic checks passed")
