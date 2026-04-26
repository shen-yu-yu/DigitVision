import yaml

def load_yaml(path = "../configs/config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


