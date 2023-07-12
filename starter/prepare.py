import argparse
import dvc.api
from dvclive import Live
import io
import os
import pandas as pd
import logging
import tempfile

logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s - %(message)s"
                    )
logger = logging.getLogger()

def go(args):
    # reading data from s3 storage using dvc.api
    logger.info("Start: raw_data reading from s3 remote storage")
    raw_data_bytes = dvc.api.read(
        path = args.input_artifact, # insert the path of the file that exists in the storage
        remote=args.remote_storage, # select the remote storage that exists in .dvc/config if there is more than one
        mode="rb" # reading data as bytes
    )
    raw_data_byte_stream = io.BytesIO(raw_data_bytes)
    raw_dataframe = pd.read_csv(raw_data_byte_stream)
    logger.info("End: raw_data reading from s3 remote storage")

    # Cleaning data

    ## Removing white spaces before columns names
    logger.info("Removing white spaces before columns names")
    raw_dataframe.rename(columns=lambda x: x.strip(), inplace=True)

    ## Removing white spaces from all cells
    logger.info("Removing white spaces from all cells")
    raw_dataframe = raw_dataframe.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    ## Removing duplicates
    logger.info("Removing duplicates")
    raw_dataframe.drop_duplicates(inplace=True)

    ## Remove "?" from raws
    logger.info("Removing \"?\" from raws")
    raw_dataframe.replace({"?": None}, inplace=True)
    raw_dataframe.dropna(inplace=True)

    ## Removing outliers
    logger.info("Removing outliers from data")
    idx = raw_dataframe["fnlgt"].between(args.min_fnlgt, args.max_fnlgt) & \
          raw_dataframe["capital-loss"].between(args.min_capital_loss, args.max_capital_loss) & \
          raw_dataframe["capital-gain"].between(args.min_capital_gain, args.max_capital_gain)
    raw_dataframe = raw_dataframe[idx].copy()

    # Create a temporary directory to save the output artifact
    logger.info("Start: Uploading the file to the remote storage")

    raw_dataframe.to_csv(args.output_artifact, index=False)
    file_name = os.path.basename(args.output_artifact)
    file_dir = os.path.dirname(args.output_artifact)
    current_dir = os.getcwd()

    # logging artifact
    with Live(resume= True) as live:
        live.next_step()
        live.log_artifact(path= args.output_artifact,
                          type="dataset",
                          desc="Preprocessed Data"
                          )

    os.system(f"cd {file_dir} && dvc add {file_name} "
              f"&& git add {file_name}.dvc "
              f"&& git commit -m \"tracking {file_name}.dvc \" "
              f"&& git push " 
              f"&& dvc push --remote {args.remote_storage} {file_name}"
              f"&& cd {current_dir}")

    logger.info("End: Uploading the file to the remote storage")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocessed data")

    parser.add_argument(
        "--input_artifact",
        type= str,
        help= "raw data full path name",
        required=True
    )
    parser.add_argument(
        "--remote_storage",
        type= str,
        help= "the remote name that exists in .dvc/config",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type= str,
        help= "cleaned data file name with extension",
        required=True
    )

    parser.add_argument(
        "--min_fnlgt",
        type= float,
        help= "minimum fnlgt value",
        required=True
    )

    parser.add_argument(
        "--max_fnlgt",
        type= float,
        help= "maximum fnlgt value",
        required=True
    )

    parser.add_argument(
        "--min_capital_loss",
        type= float,
        help= "capital loss minimum value",
        required=True
    )

    parser.add_argument(
        "--max_capital_loss",
        type= float,
        help= "capital loss maximum value",
        required=True
    )

    parser.add_argument(
        "--min_capital_gain",
        type= float,
        help= "capital gain minimum value",
        required=True
    )

    parser.add_argument(
        "--max_capital_gain",
        type= float,
        help= "capital gain maximum value",
        required=True
    )

    args = parser.parse_args()
    go(args)