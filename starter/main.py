import os
import json
import tempfile
import dvc.api
from omegaconf import DictConfig
import hydra


@hydra.main(config_name="config", config_path="../conf", version_base=None)
def go(config: DictConfig):

    os.system(
        f"dvc stage add -f -n prepare -d prepare.py -d c`"
        f"ensus.csv -p prepare.input_artifact,prepare.remote_storage,"
        f"prepare.output_artifact,prepare.min_fnlgt,prepare.max_fnlgt,prepare.min_capital_loss,"
        f"prepare.max_capital_loss,prepare.min_capital_gain,prepare.max_capital_gain -o cleaned_data.csv "
        f"python prepare.py --input_artifact census.csv "
        f"--remote_storage {config['storage']['remote']} "
        f"--output_artifact cleaned_data.csv "
        f"--min_fnlght {config['data']['min_fnlgt']} "
        f"--max_fnlght {config['data']['max_fnlgt']} "
        f"--min_capital_loss {config['data']['min_capital_loss']} "
        f"--max_capital_loss {config['data']['max_capital_loss']} "
        f"--min_capital_gain {config['data']['min_capital_gain']} "
        f"--max_capital_loss {config['data']['max_capital_gain']}")

    os.system("dvc exp run")


if __name__ == "__main__":
    go()
