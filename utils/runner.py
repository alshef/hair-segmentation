from pathlib import Path
import sys
sys.path.append("..")


class Trainer:
    def __init__(self, config: dict, experiment_path: Path) -> None:
        self.exp_path = experiment_path
        self.config = config

    def run(self) -> None:
        pass

    def train(self):
        pass

    def validate(self):
        pass
