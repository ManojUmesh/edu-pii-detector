import argparse
from config import ModelConfiguration, InputData, configure_logging

logging = configure_logging()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train",
        action="store_true",
        help="Train the model using InputData.train.",
    )
    p.add_argument(
        "--predict-after",
        action="store_true",
        help="After training, immediately run prediction.",
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a .txt/.docx/.pdf/.csv file or a directory of files for prediction. "
             "If omitted, prediction uses InputData.test (JSON).",
    )
    p.add_argument(
        "--csv-text-column",
        type=str,
        default="text",
        help="Column name to read from CSV files during prediction (default: 'text').",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # TRAIN

    if args.train:
        from preprocessing import preprocess_data
        from training import train_model

        logging.info("MAIN: Training mode (ModelConfiguration.train=True).")
        ModelConfiguration.train = True

        processed = preprocess_data(InputData.train)  # (train_ds, val_ds, class_weights)
        _ = train_model(processed, InputData)         # saves weights/model internally
        logging.info("MAIN: Training finished; checkpoints saved.")

        if not args.predict_after:
            return

    # PREDICT

    from predict import predict
    logging.info("MAIN: Prediction mode (ModelConfiguration.train=False).")
    ModelConfiguration.train = False

    paths = predict(input_path=args.input, csv_text_column=args.csv_text_column)
    logging.info(
        "MAIN: Prediction complete.\n"
        f" - submission.csv    -> {paths['submission_path']}\n"
        f" - processed_data.csv -> {paths['processed_path']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("MAIN failed: %s", e)
        raise
