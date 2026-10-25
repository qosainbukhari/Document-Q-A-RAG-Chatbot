import copy

import yaml


DEFAULT_CONFIG_PATH = "configs/config.yaml"


def deep_merge(base, override):

    # Nested dicts are merged key by key; everything else is replaced.
    merged = copy.deepcopy(base)

    for key, value in (override or {}).items():

        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_config(config_path=DEFAULT_CONFIG_PATH, override_path=None):

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if override_path:

        with open(override_path, "r") as file:
            config = deep_merge(config, yaml.safe_load(file))

    return config
