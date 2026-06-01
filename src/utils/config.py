from pathlib import Path
import yaml


def load_config():

    config_path = (
        Path(__file__)
        .parents[2]
        / "configs"
        / "config.yaml"
    )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
