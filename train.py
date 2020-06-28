from datetime import datetime
from pathlib import Path

from fire import Fire
import yaml

from utils.config import fit
import utils


def run(config: dict) -> None:
    experiment_name = (
        f"{config['model']['type']}_"
        f"{config['data']['train']['dataset_name']}_"
        f"{config['model']['loss']}_{datetime.now().strftime('%d%m%Y_%H%M%S')}"
    )
    model_path = Path(config['output_path']) / experiment_name
    model_path.mkdir(parents=True)
    with open(model_path / "config.yaml", "w") as config_file:
        yaml.dump(config, config_file)

    trainer = utils.Trainer(config=config, experiment_path=model_path)
    trainer.run()


if __name__ == "__main__":
    config = Fire(fit)
    run(config)
