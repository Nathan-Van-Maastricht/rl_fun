import math

import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(Embedding, self).__init__()
        # self.embedding = nn.Sequential(
        #     nn.Linear(input_size, embedding_size),
        #     nn.ReLU(),
        #     nn.Linear(embedding_size, embedding_size),
        # )
        self.embedding = nn.Parameter(torch.FloatTensor(input_size, embedding_size))
        self.embedding.data.uniform_(
            -(1.0 / math.sqrt(embedding_size)), 1.0 / math.sqrt(embedding_size)
        )

    def forward(self, input):
        # return self.embedding(input)
        return torch.matmul(input.float().unsqueeze(0), self.embedding).squeeze(0)


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

    def forward(self, status, distances, positions):
        puck_pos = self.embedding(positions[0:2])
        self_pos = self.embedding(positions[2:4])
        team_1_pos = self.embedding(positions[4:6])
        team_2_pos = self.embedding(positions[6:8])
        enemy_1_pos = self.embedding(positions[8:10])
        enemy_2_pos = self.embedding(positions[10:12])
        enemy_3_pos = self.embedding(positions[12:14])

        if torch.any(puck_pos == 0):
            print(f"{puck_pos=}")

        vector = torch.cat(
            (
                status,
                distances,
                puck_pos,
                self_pos,
                team_1_pos,
                team_2_pos,
                enemy_1_pos,
                enemy_2_pos,
                enemy_3_pos,
            )
        )

        accelerate = self.accelerate(vector)
        turn = self.turn(vector)
        # accelerate = torch.clamp(accelerate, min=0.1, max=0.9)
        # turn = torch.clamp(turn, min=0.0001, max=0.98)

        return accelerate, turn

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
