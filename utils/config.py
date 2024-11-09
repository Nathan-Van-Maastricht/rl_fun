import json


class Config:
    def __init__(self, path):
        with open(path) as file:
            self.data = json.load(file)

    def __getitem__(self, idx):
        return self.data[idx]
