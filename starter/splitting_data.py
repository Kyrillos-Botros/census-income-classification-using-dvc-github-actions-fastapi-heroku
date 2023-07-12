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
    current_path = os.getcwd()
    for df, path in zip([train, test], [args.train_path, args.test_path]):
        df.to_csv(path, index=False)

        # logging artifacts
        with Live(resume=True) as live:
            live.next_step()
            live.log_artifact(
                path,
                type=os.path.basename(path).split(".")[0],
                desc= os.path.basename(path).split(".")[0]
            )

        # Uploading datasets to remote storage
        os.system(f" cd {os.path.dirname(path)} && dvc add {os.path.basename(path)} "
                  f"&& git add {os.path.basename(path)}.dvc "
                  f"&& git commit -m \"tracking {os.path.basename(path)}.dvc \" "
                  f"&& git push "
                  f"&& dvc push --remote {args.remote_storage} "
                  f"&& cd {current_path}")

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

    parser.add_argument(
        "--train_path",
        type= str,
        help= "train dataset path",
        required= True
    )

    parser.add_argument(
        "--test_path",
        type= str,
        help= "test dataset path",
        required= True
    )

    args = parser.parse_args()
    go(args)