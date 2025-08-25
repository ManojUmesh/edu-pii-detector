from config import ModelConfiguration, InputData, configure_logging

logging = configure_logging()

def main():

    RUN_PRED_AFTER_TRAIN = False
    if ModelConfiguration.train:
        #TRAIN MODE
        from preprocessing import preprocess_data
        from training import train_model

        logging.info("MAIN: Training mode (ModelConfiguration.train=True).")
        processed = preprocess_data(InputData.train)
        _ = train_model(processed, InputData)
        logging.info("MAIN: Training completed and checkpoints saved.")

        if RUN_PRED_AFTER_TRAIN:
            logging.info("MAIN: Switching to inference to generate predictions after training.")
            # flip the flag and fall through to predict
            ModelConfiguration.train = False
        else:
            return

    #PREDICT MODE
    if not ModelConfiguration.train:
        from predict import predict
        logging.info("MAIN: Inference mode (ModelConfiguration.train=False).")
        paths = predict()
        logging.info(
            "MAIN: Prediction completed.\n"
            f" - submission.csv -> {paths['submission_path']}\n"
            f" - processed_data.csv -> {paths['processed_path']}"
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("MAIN failed: %s", e)
        raise
