# Script to train machine learning model.
import argparse
import logging
import pandas as pd
import dvc.api
from dvclive import Live
import io
import pickle
from ml.data import process_data
from ml.model import inference, compute_model_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
logger = logging.getLogger()


def go(args):

    # Reading testing data
    logger.info("Start: testing_data reading from s3 remote storage")
    testing_data_bytes = dvc.api.read(
        # insert the path of the file that exists in the storage
        path=args.input_artifact,
        # select the remote storage that exists in .dvc/config
        remote=args.remote_storage,
        # reading data as bytes
        mode="rb"
    )
    testing_data_byte_stream = io.BytesIO(testing_data_bytes)
    testing_dataframe = pd.read_csv(testing_data_byte_stream)
    logger.info("End: testing_data reading from s3 remote storage")

    # Reading models
    logger.info("Start: models reading from s3 remote storage")
    with dvc.api.open(
        # insert the path of the file that exists in the storage
        path=args.models_path,
        # select the remote storage that exists in .dvc/config
        remote=args.remote_storage,
        # reading data as bytes
        mode="rb"
    ) as model_file:
        models = pickle.load(model_file)

    logger.info("End: models reading from s3 remote storage")

    # Selecting category columns only
    cat_features = testing_dataframe.select_dtypes(include=['object']).columns
    cat_features = cat_features[:-1]

    # processing testing data
    logger.info("Start: processing testing data")
    X_test, y_test, _, _ = process_data(testing_dataframe,
                                        categorical_features=cat_features,
                                        label="salary",
                                        training=False,
                                        encoder=models["encoder"],
                                        lb=models["lb"]
                                        )
    logger.info("End: processing testing data")

    # Evaluating model
    logger.info("Start: predictions")
    preds = inference(model=models["model"], X=X_test)
    logger.info("End: predictions")

    logger.info("Start: Evaluation")
    precision, recall, fbeta = compute_model_metrics(y_test, preds=preds)

    with Live(resume=True, dir="../../dvclive") as live:
        live.next_step()
        live.log_metric("precision", precision)
        live.log_metric("recall", recall)
        live.log_metric("fbeta", fbeta)

    logger.info(f"precision= {precision}"
                f"\n recall= {recall}"
                f"\n fbeta= {fbeta}"
                "\n End: Evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model training")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="The path of testing data file",
        required=True
    )

    parser.add_argument(
        "--models_path",
        type=str,
        help="The path of models file",
        required=True
    )

    parser.add_argument(
        "--remote_storage",
        type=str,
        help="the remote name that exists in .dvc/config",
        required=True
    )

    args = parser.parse_args()
    go(args)
