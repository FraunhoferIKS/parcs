import yaml
from pprint import pprint


def read_yaml_config(config_dir=None):
    with open(config_dir, 'r') as stream:
        config_loaded = yaml.safe_load(stream)
    # parse
    return config_loaded


if __name__ == '__main__':
    x = read_yaml_config(config_dir='../configs/params/default.yml')
    pprint(x)
