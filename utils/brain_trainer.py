import math
import random

import pygame
import torch
import torch.optim as optim

from environment.game import Game


class BrainTrainer:
    def __init__(self, agent, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = agent.to(self.device)
        self.agent_optimiser = optim.AdamW(
            self.agent.parameters(), lr=0.001, weight_decay=1e-5
        )

        self.epoch = 0
        if self.config["learn"]:
            self.epsilon = self.config["default_epsilon"]
            self.epsilon_min = 0.1
            self.epsilon_decay = 0.006125
        else:
            self.epsilon = 0
            self.epsilon_min = 0
            self.epsilon_decay = 0

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

    def train_episode(self):
        game = Game(self.config)

        actions = []
        rewards = []
        probabilities = []

        frame = 0

        total_explore = 0
        total_exploit = 0

        while True:
            for agent in game.agents.values():
                # observe
                status, distances, positional = game.get_observation(agent)

                # normalise status
                status[1] = (status[1] + math.pi) / (2 * math.pi)
                status = torch.FloatTensor(status).to(self.device)

                # normalise distances
                distances = torch.FloatTensor(distances).to(self.device)
                distances_min = torch.min(distances)
                distances_max = torch.max(distances)
                distances = (distances - distances_min) / (
                    distances_max - distances_min
                )

                # normalise position
                positional = torch.FloatTensor(positional).to(self.device)
                positional_min = torch.min(positional)
                positional_max = torch.max(positional)
                positional = (positional - positional_min) / (
                    positional_max - positional_min
                )

                # recreate state
                # state = torch.cat((status, distances, positional))

                # act
                accelerating_probabilities, direction_probabilities = self.agent(
                    status, distances, positional
                )
                # if frame == 0 or random.random() < 0.05:
                #     print(f"{accelerating_probabilities=}")
                #     print(f"{direction_probabilities=}")

                if random.random() < self.epsilon:
                    total_explore += 1
                    # direction = torch.randint(
                    #     direction_probabilities.shape[0], (1,)
                    # ).to(self.device)

                    # accelerating = torch.randint(
                    #     accelerating_probabilities.shape[0], (1,)
                    # ).to(self.device)
                    direction = torch.argmin(direction_probabilities).unsqueeze(0)
                    accelerating = torch.argmin(accelerating_probabilities).unsqueeze(0)

                    probabilities.append(
                        [
                            accelerating_probabilities[accelerating],
                            direction_probabilities[direction],
                        ]
                    )
                    actions.append([accelerating, direction])
                else:
                    total_exploit += 1
                    # direction = direction_probabilities.multinomial(1)
                    # accelerating = accelerating_probabilities.multinomial(1)
                    direction = torch.argmax(direction_probabilities).unsqueeze(0)
                    accelerating = torch.argmax(accelerating_probabilities).unsqueeze(0)
                    probabilities.append(
                        [
                            accelerating_probabilities[accelerating],
                            direction_probabilities[direction],
                        ]
                    )
                    actions.append([accelerating, direction])

                agent.action(accelerating.item(), direction.item() - 1)

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
            if frame + 1 == self.config["total_frames"]:
                break

            if frame % 100 == 0 and self.config["learn"]:
                print(f"{frame=}")
                self.update_network(
                    actions[-100:], rewards[-100:], probabilities[-100:]
                )

        print(f"{self.epsilon=:.2f}")
        print(f"{total_explore=}")
        print(f"{total_exploit=}")
        print(f"Explore ratio: {100*total_explore/(total_explore + total_exploit):.2f}")
        print(f"Exploit ratio: {100*total_exploit/(total_explore+total_exploit):.2f}")

        pygame.quit()

        # if self.config["learn"]:
        #     self.update_network(actions, rewards, probabilities)
        self.epoch += 1

        print("finished training")
        print(f"Score: {game.score}")

    def update_network(self, actions, rewards, probabilities):
        print(f"{self.epoch}: Updating network")

        # normalise rewards
        rewards = torch.Tensor(rewards)
        min_val = rewards.min()
        max_val = rewards.max()
        rewards = (rewards - min_val) / (max_val - min_val)

        returns = self.calculate_returns(rewards)

        policy_loss = 0

        log_probabilities = 0
        for action_probabilities, action, reward in zip(
            probabilities, actions, returns
        ):
            log_probabilities = torch.log(action_probabilities[1])
            if log_probabilities > -1000:
                policy_loss -= log_probabilities * reward
            log_probabilities = torch.log(action_probabilities[0])
            if log_probabilities > -1000:
                policy_loss -= log_probabilities * reward
            policy_loss -= log_probabilities * reward

        print(f"{policy_loss=}")

        self.agent_optimiser.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0, norm_type=2)
        self.agent_optimiser.step()

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

        distance_to_puck_reward = -4 * math.log(distance_to_puck)
        if (
            distance_to_puck
            < self.config["puck"]["radius"] + self.config["agent"]["radius"] + 5
        ):
            distance_to_puck_reward += 50

        angle_to_puck = math.atan2(puck.y - agent.y, puck.x - agent.x)
        angle_difference = math.acos(math.cos(angle_to_puck - agent.direction))

        direction_to_puck_reward = -10 * angle_difference

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

        reward = (
            +distance_to_goal_reward
            + distance_to_puck_reward
            # + direction_to_puck_reward
            + closeness_to_wall_penalty
        )

        # print(f"Agent: {agent.id}, reward: {reward}")

        return reward

    def calculate_returns(self, rewards):
        returns = []
        culmulative_reward = 0
        for reward in reversed(rewards):
            returns.append(reward + 0.98 * culmulative_reward)

        return list(reversed(returns))

    def save_model(self, path):
        self.agent.save(path)
