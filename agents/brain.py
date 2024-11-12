import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(Embedding, self).__init__()
        # self.embedding = nn.ReLU(nn.Linear(input_size, embedding_size))
        self.embedding = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
        )

    def forward(self, input):
        return self.embedding(input)


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
            nn.Softmax(),
        )

        self.turn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(),
        )

        self.embedding = Embedding(2, 64)

    def forward(self, x):
        puck_pos = self.embedding(x[0:2])
        self_pos = self.embedding(x[2:4])
        is_accelerating = torch.tensor(x[4]).unsqueeze(0)
        direction = torch.tensor(x[5]).unsqueeze(0)
        team_1_pos = self.embedding(x[6:8])
        team_2_pos = self.embedding(x[8:10])
        enemy_1_pos = self.embedding(x[10:12])
        enemy_2_pos = self.embedding(x[12:14])
        enemy_3_pos = self.embedding(x[14:16])

        vector = torch.cat(
            (
                puck_pos,
                self_pos,
                is_accelerating,
                direction,
                team_1_pos,
                team_2_pos,
                enemy_1_pos,
                enemy_2_pos,
                enemy_3_pos,
            )
        )

        accelerate = self.accelerate(vector)
        turn = self.turn(vector)
        accelerate = torch.clamp(accelerate, min=0.1, max=0.9)
        turn = torch.clamp(turn, min=0.0001, max=0.98)

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
