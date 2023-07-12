# Script to train machine learning model.
import argparse
import logging
import pandas as pd
import dvc.api
import io
import os
import pickle
import tempfile
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model

logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s - %(message)s"
                    )
logger = logging.getLogger()

def go(args):

    logger.info("Start: training data reading from s3 remote storage")
    training_data_bytes = dvc.api.read(
        path= "training_data", # insert the path of the file that exists in the storage
        remote= args.remote_storage, # select the remote storage that exists in .dvc/config if there is more than one
        mode= "rb" # reading data as bytes
    )
    training_data_byte_stream = io.BytesIO(training_data_bytes)
    training_dataframe = pd.read_csv(training_data_byte_stream)
    logger.info("End: training_data reading from s3 remote storage")

    cat_features = training_dataframe.select_dtypes(include=['object']).columns

    logger.info("Start: processing training data")
    X_train, y_train, encoder, lb = process_data(
        training_dataframe, categorical_features=cat_features, label="salary", training=True
    )
    logger.info("End: processing training data")

    logger.info("Start: Training model")
    model = train_model(X_train, y_train)
    logger.info("End: Training model")


    logger.info("Start: Uploading models to the remote storage")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # creating a dictionary with all models to be saved in one pkl
        models = {
            "encoder": encoder,
            "lb": lb,
            "model": model
        }

        temp_path = os.path.join(tmp_dir, args.output_artifact)

        # Saving models as pkl file
        with open(temp_path, 'wb') as f:
            pickle.dump(models, f)

        os.system(f"cd {tmp_dir} && dvc add . "
                  f"&& git add ."
                  f"&& git commit -m tracking models "
                  f"&& git push"
                  f"&& dvc push --remote {args.remote_storage} "
                  f"&& cd ..")
    logger.info("End: Uploading models to the remote storage")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model training")


    parser.add_argument(
        "--remote_storage",
        type= str,
        help= "the remote name that exists in .dvc/config",
        required=True
    )

    parser.add_argument(
        "--random_state",
        type= int,
        help= "Random State",
        required= False,
        default= 42
    )

    parser.add_argument(
        "--output_artifact",
        type= str,
        help= "model_name.pkl",
        required= True
    )

    args = parser.parse_args()
    go(args)