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


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.swish = Swish(beta=1.0)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.swish(out)
        out = self.linear2(out)
        return self.swish(out + residual)


class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Brain(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Brain, self).__init__()
        # self.accelerate = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     # nn.ReLU(),
        #     nn.Linear(hidden_dim, 2 * hidden_dim),
        #     Swish(beta=1.0),
        #     nn.Linear(2 * hidden_dim, hidden_dim),
        #     Swish(beta=1.0),
        #     nn.Linear(hidden_dim, 2),
        #     nn.Softmax(),
        # )

        self.accelerate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=0),
        )

        # self.turn = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     # nn.ReLU(),
        #     nn.Linear(hidden_dim, 2 * hidden_dim),
        #     Swish(beta=1.0),
        #     nn.Linear(2 * hidden_dim, hidden_dim),
        #     Swish(beta=1.0),
        #     nn.Linear(hidden_dim, 5),
        #     nn.Softmax(),
        # )

        self.turn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 5),
            nn.Softmax(dim=0),
        )

        self.team_mate_embedding = Embedding(3, 32)
        self.enemy_embedding = Embedding(3, 32)

    def forward(self, status, distances, positions):
        # puck_pos = self.embedding(positions[0:2])
        # self_pos = self.embedding(positions[2:4])
        # team_1_pos = self.embedding(positions[4:6])
        # team_2_pos = self.embedding(positions[6:8])
        # enemy_1_pos = self.embedding(positions[8:10])
        # enemy_2_pos = self.embedding(positions[10:12])
        # enemy_3_pos = self.embedding(positions[12:14])

        # if torch.any(puck_pos == 0):
        #     print(f"{puck_pos=}")

        # vector = torch.cat(
        #     (
        #         status,
        #         distances,
        #         puck_pos,
        #         self_pos,
        #         team_1_pos,
        #         team_2_pos,
        #         enemy_1_pos,
        #         enemy_2_pos,
        #         enemy_3_pos,
        #     )
        # )
        puck_pos = torch.FloatTensor(positions[0:2])
        self_pos = torch.FloatTensor(positions[2:4])
        team_1_embedding = self.team_mate_embedding(
            torch.cat((positions[4:6], distances[0].unsqueeze(0)))
        )
        team_2_embedding = self.team_mate_embedding(
            torch.cat((positions[6:8], distances[1].unsqueeze(0)))
        )
        enemy_1_embedding = self.enemy_embedding(
            torch.cat((positions[8:10], distances[2].unsqueeze(0)))
        )
        enemy_2_embedding = self.enemy_embedding(
            torch.cat((positions[10:12], distances[3].unsqueeze(0)))
        )
        enemy_3_embedding = self.enemy_embedding(
            torch.cat((positions[12:14], distances[4].unsqueeze(0)))
        )

        vector = torch.cat(
            (
                status,
                puck_pos,
                self_pos,
                team_1_embedding,
                team_2_embedding,
                enemy_1_embedding,
                enemy_2_embedding,
                enemy_3_embedding,
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
            # nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(beta=1.0),
            nn.Linear(hidden_dim, 1),
        )

        self.team_mate_embedding = Embedding(3, 32)
        self.enemy_embedding = Embedding(3, 32)

    def forward(self, status, distances, positions):
        # puck_pos = self.embedding(positions[0:2])
        # self_pos = self.embedding(positions[2:4])
        # team_1_pos = self.embedding(positions[4:6])
        # team_2_pos = self.embedding(positions[6:8])
        # enemy_1_pos = self.embedding(positions[8:10])
        # enemy_2_pos = self.embedding(positions[10:12])
        # enemy_3_pos = self.embedding(positions[12:14])

        # if torch.any(puck_pos == 0):
        #     print(f"{puck_pos=}")

        # vector = torch.cat(
        #     (
        #         status,
        #         distances,
        #         puck_pos,
        #         self_pos,
        #         team_1_pos,
        #         team_2_pos,
        #         enemy_1_pos,
        #         enemy_2_pos,
        #         enemy_3_pos,
        #     )
        # )

        puck_pos = torch.FloatTensor(positions[0:2])
        self_pos = torch.FloatTensor(positions[2:4])
        team_1_embedding = self.team_mate_embedding(
            torch.cat((positions[4:6], distances[0].unsqueeze(0)))
        )
        team_2_embedding = self.team_mate_embedding(
            torch.cat((positions[6:8], distances[1].unsqueeze(0)))
        )
        enemy_1_embedding = self.enemy_embedding(
            torch.cat((positions[8:10], distances[2].unsqueeze(0)))
        )
        enemy_2_embedding = self.enemy_embedding(
            torch.cat((positions[10:12], distances[3].unsqueeze(0)))
        )
        enemy_3_embedding = self.enemy_embedding(
            torch.cat((positions[12:14], distances[4].unsqueeze(0)))
        )

        # vector = torch.cat((status, distances, positions))

        vector = torch.cat(
            (
                status,
                puck_pos,
                self_pos,
                team_1_embedding,
                team_2_embedding,
                enemy_1_embedding,
                enemy_2_embedding,
                enemy_3_embedding,
            )
        )

        return self.main(vector)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
