import math
import random

import pygame
import torch
import torch.optim as optim

from environment.game import Game


class BrainTrainer:
    def __init__(self, agent, config, critic):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = agent.to(self.device)
        self.agent_optimiser = optim.AdamW(
            self.agent.parameters(), lr=1e-5, weight_decay=1e-4
        )

        self.critic = critic.to(self.device)
        self.critic_optimsier = optim.AdamW(
            self.critic.parameters(), lr=1e-4, weight_decay=1e-4
        )

        self.epoch = 0
        if self.config["learn"]:
            self.epsilon = self.config["default_epsilon"]
            self.epsilon_min = 0.1
            self.epsilon_decay = 2e-3
        else:
            self.epsilon = 0
            self.epsilon_min = 0
            self.epsilon_decay = 0

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

    def normalise_observation(self, status, distances, positional):
        # normalise status
        status[1] = (status[1] + math.pi) / (2 * math.pi)
        status = torch.FloatTensor(status).to(self.device)

        # normalise distances
        distances = torch.FloatTensor(distances).to(self.device)
        distances_min = torch.min(distances)
        distances_max = torch.max(distances)
        distances = (distances - distances_min) / (distances_max - distances_min)

        # normalise position
        positional = torch.FloatTensor(positional).to(self.device)
        positional_min = torch.min(positional)
        positional_max = torch.max(positional)
        positional = (positional - positional_min) / (positional_max - positional_min)

        return status, distances, positional

    def train_episode(self):
        print(f"{self.epsilon=:.4f}")
        game = Game(self.config)

        # actions = []
        rewards = []
        probabilities = []
        values = []

        frame = 0

        total_explore = 0
        total_exploit = 0

        while True:
            if self.config["visualise"]:
                pygame.event.get()
            for agent in game.agents.values():
                # observe
                status, distances, positional = game.get_observation(agent)
                status, distances, positional = self.normalise_observation(
                    status, distances, positional
                )
                value_estimate = self.critic(status, distances, positional)
                values.append(value_estimate)

                # act
                accelerating_probabilities, direction_probabilities = self.agent(
                    status, distances, positional
                )

                if random.random() < self.epsilon:
                    total_explore += 1
                    direction = torch.randint(
                        direction_probabilities.shape[0], (1,)
                    ).to(self.device)

                    accelerating = torch.randint(
                        accelerating_probabilities.shape[0], (1,)
                    ).to(self.device)

                    # direction = torch.argmin(direction_probabilities).unsqueeze(0)
                    # accelerating = torch.argmin(accelerating_probabilities).unsqueeze(0)
                else:
                    total_exploit += 1

                    direction = direction_probabilities.multinomial(1)
                    accelerating = accelerating_probabilities.multinomial(1)
                #     direction = torch.argmax(direction_probabilities).unsqueeze(0)
                #     accelerating = torch.argmax(
                #         accelerating_probabilities
                #     ).unsqueeze(0)

                agent.action(accelerating.item(), direction.item() - 20)
                probabilities.append(
                    [
                        accelerating_probabilities[accelerating],
                        direction_probabilities[direction],
                    ]
                )
                # actions.append([accelerating, direction])

            # update state
            goal_state = game.update()

            # determine rewards
            for agent in game.agents.values():
                rewards.append(self.reward(game.puck, goal_state, agent))
                agent.reward = rewards[-1]

            if self.config["visualise"]:
                game.draw()

            # input()

            frame += 1
            if frame % 125 == 0 and self.config["learn"]:
                print(f"{frame=}")
                self.update_network(
                    rewards,
                    probabilities,
                    values,
                )
                rewards = []
                probabilities = []
                values = []

            if frame == self.config["total_frames"]:
                break

        print(f"{total_explore=}")
        print(f"{total_exploit=}")
        print(f"Explore ratio: {100*total_explore/(total_explore + total_exploit):.2f}")
        print(f"Exploit ratio: {100*total_exploit/(total_explore+total_exploit):.2f}")
        print("Game done")
        if game.score[0] > game.score[1]:
            print(f"Score: \033[32m{game.score[0]} : \033[31m{game.score[1]}\033[0m")
        elif game.score[0] < game.score[1]:
            print(f"Score: \033[31m{game.score[0]} : \033[32m{game.score[1]}\033[0m")
        else:
            print(f"Score: \033[95m{game.score[0]} : {game.score[1]}\033[0m")

        if self.config["visualise"]:
            pygame.quit()

        # if self.config["learn"]:
        #     self.update_network(rewards, probabilities, values)

        self.epoch += 1

        print("finished training\n")

    def update_network(self, rewards, probabilities, values):
        print(f"{self.epoch}: Updating network")

        values = torch.cat(values)
        probabilities = torch.Tensor(probabilities).transpose(0, 1).to(self.device)

        # normalise rewards
        rewards = torch.Tensor(rewards).to(self.device)
        min_val = rewards.min()
        max_val = rewards.max()
        rewards = (rewards - min_val) / (max_val - min_val)

        rewards = torch.Tensor(self.calculate_returns(rewards)).to(self.device)

        advantage = rewards - values

        log_probabilities = torch.zeros(probabilities.shape[1]).to(self.device)
        for action_probability in probabilities:
            log_probabilities += torch.log(action_probability)
        log_probabilities[log_probabilities < -1000] = 0.0

        reinforce = advantage * log_probabilities
        actor_loss = reinforce.mean()
        critic_loss = advantage.pow(2).mean()
        total_loss = actor_loss + critic_loss + random.random() * 2.5

        print(f"{actor_loss=}")
        print(f"{critic_loss=}")

        self.agent_optimiser.zero_grad()
        self.critic_optimsier.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 2.0, norm_type=2)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 2.0, norm_type=2)
        self.agent_optimiser.step()
        self.critic_optimsier.step()

    def reward(self, puck, goal_state, agent):
        if agent.team == 0:
            target_distance = puck.distance_to_goal(1)
            own_distance = puck.distance_to_goal(0)
        else:
            target_distance = puck.distance_to_goal(0)
            own_distance = puck.distance_to_goal(1)

        distance_to_goal_reward = 100 * math.exp(
            -target_distance / 100
        ) - 100 * math.exp(-own_distance / 125)

        distance_to_puck = ((agent.x - puck.x) ** 2 + (agent.y - puck.y) ** 2) ** 0.5

        distance_to_puck_reward = 20 * math.exp(-distance_to_puck / 300) - 10
        if (
            distance_to_puck
            < self.config["puck"]["radius"] + self.config["agent"]["radius"] + 5
        ):
            distance_to_puck_reward += 50

        angle_to_puck = math.atan2(puck.y - agent.y, puck.x - agent.x)
        angle_difference = math.acos(math.cos(angle_to_puck - agent.direction))

        direction_to_puck_reward = -angle_difference

        distance_to_left_wall = agent.x
        distance_to_right_wall = self.config["field"]["width"] - agent.x
        distance_to_top = agent.y
        distance_to_bottom = self.config["field"]["height"] - agent.y

        min_distance = min(
            [
                distance_to_left_wall,
                distance_to_right_wall,
                distance_to_top,
                distance_to_bottom,
            ]
        )

        closeness_to_wall_penalty = -25 * math.exp(-min_distance / 50)

        goal_state_reward = 0
        if goal_state == 1:
            if agent.team == 0:
                goal_state_reward = -500
            else:
                goal_state_reward = 500
        elif goal_state == -1:
            if agent.team == 0:
                goal_state_reward = 500
            else:
                goal_state_reward = -500

        reward = (
            +distance_to_goal_reward
            + distance_to_puck_reward
            + direction_to_puck_reward
            + closeness_to_wall_penalty
            + goal_state_reward
        )

        return reward

    def calculate_returns(self, rewards):
        returns = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            returns.append(reward + 0.99 * cumulative_reward)
            cumulative_reward = returns[-1]

        return list(reversed(returns))

    def save_model(self, path):
        self.agent.save(path)
