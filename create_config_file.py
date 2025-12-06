import yaml

data_yaml = {
    "path": "/data/data_v2",
    "train": "images/train",
    "val": "images/test",
    "test": "images/test",
    "names": {0: "WBC", 1: "RBC", 2: "Platelets"},
}
with open("config.yaml", "w") as f:
    yaml.dump(data_yaml, f)
