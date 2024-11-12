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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.turn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),
            nn.Softmax(),
        )

    def forward(self, x):
        accelerate = self.accelerate(x)
        turn = self.turn(x)
        accelerate = torch.clamp(accelerate, min=-10, max=10)
        turn = torch.clamp(turn, min=0.00001, max=0.99)

        return torch.softmax(accelerate, dim=0), torch.softmax(turn, dim=0)

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
