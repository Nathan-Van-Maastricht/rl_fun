import random

import torch
import torch.nn as nn
import torch.optim as optim


class DDPGAgent:
    def __init__(
        self, state_dim, action_dim, num_actions=1, hidden_dim=128, config=None
    ):
        self.config = config

        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, num_actions, hidden_dim)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, num_actions, hidden_dim)

        self.actor_optimsier = optim.AdamW(self.actor.parameters(), lr=1e-3)
        self.critic_optimiser = optim.AdamW(self.critic.parameters(), lr=1e-2)

        self.tau = 0.005

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.actor_target.to(self.device)
        self.critic_target.to(self.device)

    def select_action(self, state):
        # print(f"{state=}")
        if self.config["learn"] and random.random() < self.config["default_epsilon"]:
            action = 2 * torch.rand(1, requires_grad=True) - 1
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            action = self.actor(state)

        # print(f"{action=}")
        return action

    def update(self, experiences):
        states, actions, rewards, next_states = experiences
        # Put data into tensors on appropriate device
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.tensor(
            rewards, device=self.device, dtype=torch.float32
        ).unsqueeze(1)
        next_states = torch.stack(next_states)
        state_action_pair = torch.cat((states, actions), dim=1)

        # Compute target values
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_state_action_pair = torch.cat((next_states, next_actions), dim=1)

            target_q = self.critic_target(next_state_action_pair)
            target_q = rewards + 0.99 * target_q

        # Critic update
        q_value = self.critic(state_action_pair)

        critic_loss = (q_value - target_q).pow(2).mean()
        actor_loss = -q_value.mean()

        print(f"{actor_loss=}")
        print(f"{critic_loss=}")

        self.actor_optimsier.zero_grad()
        self.critic_optimiser.zero_grad()

        (actor_loss + critic_loss).backward()

        # for name, param in self.actor.named_parameters():
        #     print(f"{name}: grad={param.grad}")

        self.actor_optimsier.step()
        self.critic_optimiser.step()

        # Target network update
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.actor.train()

    def save(self, actor_path, critic_path):
        self.actor.save(actor_path)
        self.critic.save(critic_path)

    def load(self, actor_path, critic_path):
        self.actor.load(actor_path)
        self.critic.load(critic_path)


class Actor(nn.Module):
    def __init__(self, state_dim, out_dim, hidden_dim=128):
        super(Actor, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            # nn.Dropout(0.5),
            ResidualBlock(hidden_dim, hidden_dim),
            # nn.Dropout(0.5),
            ResidualBlock(hidden_dim, hidden_dim),
            # nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.main(state)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class Critic(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=128):
        super(Critic, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(state_dim + num_actions, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            ResidualBlock(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            ResidualBlock(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_action_pair):
        return self.main(state_action_pair)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

        self.mish = nn.Sequential(
            nn.Softplus(),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mish(x)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.mish = Mish()

    def forward(self, x):
        residual = x
        out = self.main(x)
        return self.mish(out + residual)
