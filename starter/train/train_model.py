# Script to train machine learning model.
import argparse
import logging
import pandas as pd
import dvc.api
from dvclive import Live
import io
import os
import pickle
import json
from starter.ml.data import process_data
from starter.ml.model import train_model

logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s - %(message)s"
                    )
logger = logging.getLogger()

def go(args):

    # Reading training dataset
    logger.info("Start: training data reading from s3 remote storage")
    training_data_bytes = dvc.api.read(
        path= args.input_artifact, # insert the path of the file that exists in the storage
        remote= args.remote_storage, # select the remote storage that exists in .dvc/config if there is more than one
        mode= "rb" # reading data as bytes
    )
    training_data_byte_stream = io.BytesIO(training_data_bytes)
    training_dataframe = pd.read_csv(training_data_byte_stream)
    logger.info("End: training_data reading from s3 remote storage")

    # Select category data only
    cat_features = training_dataframe.select_dtypes(include=['object']).columns
    cat_features = cat_features[:-1]

    # processing training data
    logger.info("Start: processing training data")
    X_train, y_train, encoder, lb = process_data(
        training_dataframe, categorical_features=cat_features, label="salary", training=True
    )
    logger.info("End: processing training data")

    # Reading model parameters
    with open(args.model_config) as fp:
        model_config = json.load(fp)
    model_config["random_state"] = args.random_state
    os.remove(args.model_config)

    # Start training process
    logger.info("Start: Training model")
    model = train_model(X_train, y_train, model_config)
    logger.info("End: Training model")

    # creating a dictionary with all models to be saved in one pkl
    models = {
        "encoder": encoder,
        "lb": lb,
        "model": model
    }

    file_name = os.path.basename(args.output_artifact)
    file_dir = os.path.dirname(args.output_artifact)
    current_dir = os.getcwd()

    # Saving models as pkl file
    with open(args.output_artifact, 'wb') as f:
        pickle.dump(models, f)


    logger.info("Start: Uploading models to the remote storage")
    os.system(f"cd {file_dir} && dvc commit {file_name}  "
              f"&& dvc add {file_name}"
              f"&& git add {file_name}.dvc"
              f"&& git commit -m tracking models "
              f"&& git push {file_name}"
              f"&& dvc push --remote {args.remote_storage} {file_name} "
              f"&& cd {current_dir}")
    logger.info("End: Uploading models to the remote storage")

    #tracking parameters and models
    with Live(resume= True, dir="../../dvclive") as live:
        live.next_step()
        live.log_params(params=model_config)
        live.log_artifact(path=args.output_artifact,
                          type="model",
                          labels=["encoder", "lb", "model"]
                          )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model training")

    parser.add_argument(
        "--input_artifact",
        type= str,
        help= "training dataset path",
        required= True
    )

    parser.add_argument(
        "--remote_storage",
        type= str,
        help= "the remote name that exists in .dvc/config",
        required=True
    )

    parser.add_argument(
        "--model_config",
        type= str,
        help= "model config json file path",
        required= True
    )
    parser.add_argument(
        "--random_state",
        type = int,
        help= "random seed for model reproducability",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type= str,
        help= "model_name.pkl",
        required= True
    )

    args = parser.parse_args()
    go(args)