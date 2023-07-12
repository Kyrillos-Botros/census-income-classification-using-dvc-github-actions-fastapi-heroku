import argparse
import logging
import pandas as pd
import dvc.api
from dvclive import Live
import io
import os
import tempfile
from sklearn.model_selection import train_test_split

logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s - %(message)s"
                    )
logger = logging.getLogger()

def go(args):

    # Reading cleaned data from remote storage
    logger.info("Start: cleaned_data reading from s3 remote storage")
    cleaned_data_bytes = dvc.api.read(
        path= args.input_artifact, # insert the path of the file that exists in the storage
        remote= args.remote_storage, # select the remote storage that exists in .dvc/config if there is more than one
        mode= "rb" # reading data as bytes
    )
    cleaned_data_byte_stream = io.BytesIO(cleaned_data_bytes)
    cleaned_dataframe = pd.read_csv(cleaned_data_byte_stream)
    logger.info("End: cleaned_data reading from s3 remote storage")

    # splitting Data
    logger.info("Start: splitting data into training and testing")
    train, test = train_test_split(cleaned_dataframe, test_size= args.test_size, random_state= args.random_state)
    logger.info("End: splitting data into training and testing")

    # uploading training and testing datasets
    logger.info("Start: Uploading files to the remote storage")
    with tempfile.TemporaryDirectory() as tmp_dir:
        for df, name in zip([train, test], ["train_data.csv", "test_data.csv"]):
            temp_path = os.path.join(tmp_dir, name)
            df.to_csv(temp_path, index=False)

        os.system(f"cd {tmp_dir}")

        # logging artifacts
        with Live(save_dvc_exp=True) as live:
            live.log_artifact(
                "train_data.csv",
                type="train_data",
                name= "train_data.csv",
                desc= "train dataset"
            )
            live.log_artifact(
                "test_data.csv",
                type="test_data",
                name= "test_data.csv",
                desc= "test dataset"
            )

        # Uploading datasets to remote storage
        os.system(f"dvc add . "
                  f"&& git add ."
                  f"&& git commit -m \"tracking training and testing data\" "
                  f"&& git push"
                  f"&& dvc push --remote {args.remote_storage}"
                  f"&& cd ..")


    logger.info("End: Uploading files to the remote storage")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="splitting data")

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