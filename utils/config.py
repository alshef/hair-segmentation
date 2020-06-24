import yaml


def update_config(config: dict, params: dict) -> dict:
    for k, v in params.items():
        *path, key = k.split('.')
        conf = config
        for p in path:
            conf = conf[p]
        conf[key] = v
    return config


def fit(**kwargs) -> dict:
    if "config" in kwargs.keys():
        config_path = kwargs["config"]
    else:
        config_path = "config_template.yaml"
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = update_config(config, kwargs)
    return config
