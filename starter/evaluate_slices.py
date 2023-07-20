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
    # reading test dataframe
    testing_dataframe = pd.read_csv(args.csv_path)

    # reading model
    with open(args.model_path, "rb") as model_file:
        models = pickle.load(model_file)

    cat_dict = {}
    evaluation_dict = {}

    # Selecting category columns only
    cat_features = testing_dataframe.select_dtypes(include=['object']).columns
    cat_features = cat_features[:-1]

    # Getting the reference evaluation metrics
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
    ref_precision, ref_recall, ref_fbeta = compute_model_metrics(y_test, preds=preds)

    evaluation_dict["reference"] = {"all_categories":
                                        {"precision": ref_precision,
                                        "recall": ref_recall,
                                        "fbeta": ref_fbeta
                                        }
                                    }

    # Evaluating model for each slice
    for category in cat_features:
        category_unique_values = testing_dataframe[category].unique()
        for value in category_unique_values:
            dataset_slice = testing_dataframe.copy()
            dataset_slice[category] = value

            # processing testing data
            logger.info(f"Start: processing testing data on {category} = {value}")
            X_test, y_test, _, _ = process_data(dataset_slice,
                                                categorical_features=cat_features,
                                                label="salary",
                                                training=False,
                                                encoder=models["encoder"],
                                                lb=models["lb"]
                                                )
            logger.info(f"End: processing testing data on {category} = {value}")
            # Evaluating model
            logger.info("Start: predictions")
            preds = inference(model=models["model"], X=X_test)
            logger.info("End: predictions")
            logger.info("Start: Evaluation")
            precision, recall, fbeta = compute_model_metrics(y_test, preds=preds)
            logger.info("End: Evaluation")
            cat_dict[value] = {"precision": precision,
                               "recall": recall,
                                "fbeta": fbeta
                              }

        evaluation_dict[category] = cat_dict
        cat_dict = {}

    # create slice_output.txt and write evaluation_dict to it
    with open("slice_output.txt", "w") as f:
        for category, values_dict in evaluation_dict.items():
            for value, metrics in values_dict.items():
                f.write(f"{category} = {value}\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value}\n")
                f.write("\n")
            f.write("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model testing for each slice")

    parser.add_argument(
        "--csv_path",
        type=str,
        help="The path of testing data file",
        default="../data/test-data.csv",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="The path of models file",
        default="../model/rfmodel.pkl",
    )

    args = parser.parse_args()
    go(args)
