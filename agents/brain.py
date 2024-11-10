import torch
import torch.nn as nn


class Brain(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Brain, self).__init__()
        self.accelerate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(),
        )
        self.turn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 11),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.accelerate(x), self.turn(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class BrainValue(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BrainValue, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.main(x)
