import os
import json
from omegaconf import DictConfig
import hydra
import warnings
warnings.filterwarnings('ignore')

@hydra.main(config_name="config", config_path="../conf", version_base=None)
def go(config: DictConfig):

    # Executing prepare.py
    os.system(f"dvc stage add -f -n prepare -d prepare.py -d ../data/census.csv "
              f"-o ../data/cleaned-data.csv "
              f"python prepare.py --input_artifact ../data/census.csv "
              f"--remote_storage {config['storage']['remote']} "
              f"--output_artifact ../data/cleaned-data.csv "
              f"--min_fnlgt {config['data']['min_fnlgt']} "
              f"--max_fnlgt {config['data']['max_fnlgt']} "
              f"--min_capital_loss {config['data']['min_capital_loss']} "
              f"--max_capital_loss {config['data']['max_capital_loss']} "
              f"--min_capital_gain {config['data']['min_capital_gain']} "
              f"--max_capital_gain {config['data']['max_capital_gain']}")

    # Executing splitting_data.py
    os.system(f"dvc stage add -f -n split -d splitting_data.py -d ../data/cleaned-data.csv "
              f"-o ../data/train-data.csv -o ../data/test-data.csv "
              f"python splitting_data.py --input_artifact ../data/cleaned-data.csv "
              f"--remote_storage {config['storage']['remote']} "
              f"--test_size {config['modeling']['test_size']} "
              f"--random_state {config['modeling']['random_state']} "
              f"--train_path ../data/train-data.csv "
              f"--test_path ../data/test-data.csv "
              )

    # Executing train_model.py
    model_config = "model_config.json"
    with open(model_config, "w+") as fp:
        json.dump(dict(config["modeling"]["random_forest"].items()), fp)

    os.system(f"dvc stage add -f -n train -d train_model.py -d ../data/train-data.csv "
              f"-o ../model/rfmodel.pkl "
              f"python train_model.py --input_artifact ../data/train-data.csv "
              f"--remote_storage {config['storage']['remote']} "
              f"--random_state {config['modeling']['random_state']} "
              f"--model_config {model_config}  "
              f"--output_artifact ../model/rfmodel.pkl"
              )

    # Executing evaluation
    os.system(f"dvc stage add -f -n evaluate -d evaluate_model.py -d ../data/test-data.csv -d ../model/rfmodel.pkl "
              f"-m ../dvclive/metrics.json "
              f" python evaluate_model.py --input_artifact ../data/test-data.csv "
              f"--remote_storage {config['storage']['remote']} "
              f"--models_path ../model/rfmodel.pkl"
              )

    os.system("git add dvc.yaml dvc.lock "
              "&& git commit -m \"tracking dvc.yaml\" "
              "&& git push")

    os.system("dvc exp run")

if __name__ == "__main__":
    go()
