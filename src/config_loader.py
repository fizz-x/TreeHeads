import yaml
from copy import deepcopy
from pathlib import Path

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def get_config(exp_name, sites_yaml="../configs/datapaths.yaml", exp_yaml="../configs/experiments.yaml"):
    sites = load_yaml(sites_yaml)
    experiments = load_yaml(exp_yaml)

    exp = deepcopy(experiments[exp_name])

    exp["exp"] = exp_name
    # tbd ?

    return sites, exp
