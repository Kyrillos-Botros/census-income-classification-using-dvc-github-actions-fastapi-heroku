# Script to train machine learning model.
import argparse
import logging
import pandas as pd
import dvc.api
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

    logger.info("Start: testing_data reading from s3 remote storage")
    testing_data_bytes = dvc.api.read(
        path= "testing_data", # insert the path of the file that exists in the storage
        remote= args.remote_storage, # select the remote storage that exists in .dvc/config if there is more than one
        mode= "rb" # reading data as bytes
    )
    testing_data_byte_stream = io.BytesIO(testing_data_bytes)
    testing_dataframe = pd.read_csv(testing_data_byte_stream)
    logger.info("End: testing_data reading from s3 remote storage")

    logger.info("Start: models reading from s3 remote storage")
    models = dvc.api.read(
        path= args.models, # insert the path of the file that exists in the storage
        remote= args.remote_storage, # select the remote storage that exists in .dvc/config if there is more than one
        mode= "rb" # reading data as bytes
    )
    models = pickle.load(models)
    logger.info("End: models reading from s3 remote storage")

    cat_features = testing_dataframe.select_dtypes(include=['object']).columns


    logger.info("Start: processing testing data")
    X_test, y_test, _, _ = process_data(
        testing_dataframe, categorical_features=cat_features, label="salary", training=False, encoder=models["encoder"], lb= models["lb"]
    )
    logger.info("End: processing testing data")

    logger.info("Start: predictions")
    preds = inference(model= models["model"], X= X_test)
    logger.info("End: predictions")

    logger.info("Start: Evaluation")
    precision, recall, fbeta= compute_model_metrics(y_test, preds=preds)
    logger.info(f"precision= {precision}"
                f"\n recall= {recall}"
                f"\n fbeta= {fbeta}"
                "\n End: Evaluation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model training")

    parser.add_argument(
        "--input_artifact",
        type= str,
        help= "The name of preprocessed data file",
        required= True
    )

    parser.add_argument(
        "--remote_storage",
        type= str,
        help= "the remote name that exists in .dvc/config",
        required=True
    )

    parser.add_argument(
        "--test_size",
        type= float,
        help= "The size of test data",
        required= False,
        default= 0.2
    )

    parser.add_argument(
        "--random_state",
        type= int,
        help= "Random State",
        required= False,
        default= 42
    )

    args = parser.parse_args()
    go(args)