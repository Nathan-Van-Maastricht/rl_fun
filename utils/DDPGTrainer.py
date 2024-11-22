import math
import random

import pygame
import torch

from agents.DDPGAgent import DDPGAgent
from environment.game import Game
from utils.config import Config


class DDPGTrainer:
    def __init__(self, agent: DDPGAgent, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent = agent

    def normalise_observation(self, status, distances, positional):
        # normalise status
        status[1] = (status[1] + math.pi) / (2 * math.pi)
        status = torch.FloatTensor(status).to(self.device)

        # normalise distances
        distances = torch.FloatTensor(distances).to(self.device)
        # distances = torch.tanh(distances)
        distances_min = torch.min(distances)
        distances_max = torch.max(distances)
        distances = (distances - distances_min) / (distances_max - distances_min)

        # normalise position
        positional = torch.FloatTensor(positional).to(self.device)
        positional_min = torch.min(positional)
        positional_max = torch.max(positional)
        positional = (positional - positional_min) / (positional_max - positional_min)

        return torch.cat((status, distances, positional))

    def train(self, number_games, start=0):
        replay_buffer = []

        for game_number in range(start, number_games + 1):
            game = Game(self.config)

            frame = 1
            while frame <= self.config["total_frames"]:
                if self.config["visualise"]:
                    pygame.event.get()

                agent_buffer = {i: [] for i in range(6)}

                for agent in game.agents.values():
                    state = self.normalise_observation(*game.get_observation(agent))
                    action = self.agent.select_action(state)
                    agent_buffer[agent.id].append(state)
                    agent_buffer[agent.id].append(action)

                for agent in game.agents.values():
                    agent.action(1, agent_buffer[agent.id][1].item())

                goal_state = game.update()

                for agent in game.agents.values():
                    reward = self.reward(game.puck, goal_state, agent)
                    agent_buffer[agent.id].append(reward)
                    agent.reward = reward

                for agent in game.agents.values():
                    agent_buffer[agent.id].append(
                        self.normalise_observation(*game.get_observation(agent))
                    )

                for agent in game.agents.values():
                    if agent.id < 3:
                        replay_buffer.append(agent_buffer[agent.id])

                if (
                    self.config["learn"]
                    and len(replay_buffer) >= self.config["mini_batch_size"]
                ):
                    print(f"Game: {game_number} Updating on frame {frame}")
                    experiences = zip(
                        *random.sample(replay_buffer, self.config["mini_batch_size"])
                    )
                    self.agent.update(experiences)
                    replay_buffer.clear()

                if self.config["visualise"]:
                    game.draw()

                frame += 1

            self.printGameScore(game.score)
            if game_number % 2 == 0 and self.config["learn"]:
                self.agent.save(
                    f"weights/a{game_number:0>5}.pth", f"weights/c{game_number:0>5}.pth"
                )

    def printGameScore(self, score):
        if score[0] > score[1]:
            print(f"Score: \033[32m{score[0]}\033[0m : \033[31m{score[1]}\033[0m")
        elif score[0] < score[1]:
            print(f"Score: \033[31m{score[0]}\033[0m : \033[32m{score[1]}\033[0m")
        else:
            print(f"Score: \033[95m{score[0]} : {score[1]}\033[0m")

    def reward(self, puck, goal_state, agent):
        if agent.team == 0:
            target_distance = puck.distance_to_goal(1)
            own_distance = puck.distance_to_goal(0)
        else:
            target_distance = puck.distance_to_goal(0)
            own_distance = puck.distance_to_goal(1)

        distance_to_goal_reward = 25 * math.exp(-target_distance / 150)
        distance_to_own_goal_penalty = -20 * math.exp(-own_distance / 150)

        distance_to_puck = ((agent.x - puck.x) ** 2 + (agent.y - puck.y) ** 2) ** 0.5

        distance_to_puck_reward = 30 * math.exp(-distance_to_puck / 300) - 12
        if (
            distance_to_puck
            < self.config["puck"]["radius"] + self.config["agent"]["radius"] + 5
        ):
            distance_to_puck_reward += 50

        angle_to_puck = math.atan2(puck.y - agent.y, puck.x - agent.x)
        angle_difference = math.acos(math.cos(angle_to_puck - agent.direction))

        rotation_penalty = -10 * (agent.last_direction**2)

        # direction_to_puck_reward = -angle_difference
        if abs(angle_difference) < 1:
            direction_to_puck_reward = 2
        else:
            direction_to_puck_reward = -1

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
            + distance_to_own_goal_penalty
            + distance_to_puck_reward
            + direction_to_puck_reward
            + closeness_to_wall_penalty
            + goal_state_reward
            + rotation_penalty
        )

        return reward
