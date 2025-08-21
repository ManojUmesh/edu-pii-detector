import logging

# Configure the logging module (idempotent)
def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if this is called multiple times (e.g., in Colab)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        fh = logging.FileHandler('logfile.log')
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logging

# Class to define input data paths
class InputData:
    train = "/content/drive/MyDrive/Colab Notebooks/PII-Data-Detection/input/train.json"
    test = "/content/drive/MyDrive/Colab Notebooks/PII-Data-Detection/input/test.json"
    sample = "/content/drive/MyDrive/Colab Notebooks/PII-Data-Detection/input/sample_submission.csv"
    save_dir = "/content/drive/MyDrive/Colab Notebooks/PII-Data-Detection/models"
    trained_model = "/content/drive/MyDrive/Colab Notebooks/PII-Data-Detection/models/model.weights.h5"

# Configuration parameters for the model
class ModelConfiguration:
    seed = 2024
    preset = "deberta_v3_small_en"   # KerasNLP preset
    # Training
    train_seq_len = 768
    train_batch_size = 8
    val_seq_len = 768
    val_batch_size = 8
    epochs = 10
    lr_mode = "cos"                  # 'cos' | 'exp' | 'step'
    # Inference
    infer_seq_len = 768
    infer_batch_size = 2
    pred_threshold = 0.60
    
    # Labels / mapping
    labels = [
        "B-USERNAME", "B-ID_NUM", "I-PHONE_NUM", "I-ID_NUM",
        "I-NAME_STUDENT", "B-EMAIL", "I-STREET_ADDRESS",
        "B-STREET_ADDRESS", "B-URL_PERSONAL", "O",
        "I-URL_PERSONAL", "B-PHONE_NUM", "B-NAME_STUDENT"
    ]
    id2label = dict(enumerate(labels))# integer values for BIO mapping
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(labels) # number of PII tags

    train = True   # whether to train or use already trained
