from preprocessing import preprocess_data
from config import ModelConfiguration

# Toggle training mode ON for this test
ModelConfiguration.train = True

# Use the relative path that matches your repo structure
data_path = "input/train.json"

train_ds, val_ds = preprocess_data(data_path)

for x, y in train_ds.take(1):
    print("Train batch token_ids shape:", x["token_ids"].shape)
    print("Train batch labels shape:", y.shape)
