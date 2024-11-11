import math
import random

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

    def train_episode(self):
        game = Game(self.config)

        actions = []
        rewards = []
        probabilities = []

        frame = 0

        while True:
            agents_not_used = set()
            for agent in game.agents.values():
                # observe
                state = torch.FloatTensor(game.get_observation(agent.id)).to(
                    self.device
                )

                # act
                accelerating_probabilities, direction_probabilities = self.agent(state)

                if random.random() < 0.15:
                    agents_not_used.add(agent)
                    direction = torch.randint(
                        direction_probabilities.shape[0], (1,)
                    ).to(self.device)
                    accelerating = torch.randint(
                        accelerating_probabilities.shape[0], (1,)
                    ).to(self.device)
                else:
                    direction = direction_probabilities.multinomial(1)
                    accelerating = accelerating_probabilities.multinomial(1)
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
            if self.config["visualise"]:
                game.draw()

            # determine rewards
            for agent in game.agents.values():
                if agent in agents_not_used:
                    continue
                rewards.append(self.reward(game.puck, goal_state, agent))

            frame += 1
            if frame + 1 == self.config["total_frames"]:
                break

        self.update_network(actions, rewards, probabilities)
        self.epoch += 1

        print("finished training")
        print(f"Score: {game.score}")

    def update_network(self, actions, rewards, probabilities):
        print(f"{self.epoch}: Updating network")
        # returns = self.calculate_returns(rewards)
        policy_loss = 0

        log_probabilities = 0
        for action_probabilities, action, reward in zip(
            probabilities, actions, rewards
        ):
            log_probabilities = torch.log(action_probabilities[1])
            # print(f"{action_probabilities[1]=}")
            # print(f"{log_probabilities=}")
            # print(f"{reward=}")
            policy_loss -= log_probabilities * reward
            log_probabilities = torch.log(action_probabilities[0])
            policy_loss -= log_probabilities * reward
            # print(f"{policy_loss=}")

        print(f"{policy_loss=}")

        self.agent_optimiser.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 2.0, norm_type=2)
        self.agent_optimiser.step()

    def reward(self, puck, goal_state, agent):
        print(f"{agent.id=}")
        distance_to_goal0 = puck.distance_to_goal(0)
        distance_to_goal1 = puck.distance_to_goal(1)
        # print(f"{distance_to_goal0=}")
        # print(f"{distance_to_goal1=}")
        if agent.team == 0:
            positive_goal = 100000 / (distance_to_goal1 + 100)
            negative_goal = 100000 / (distance_to_goal0 + 10)
        if agent.team == 1:
            positive_goal = 100000 / (distance_to_goal0 + 100)
            negative_goal = 100000 / (distance_to_goal1 + 10)

        distance_to_goal_reward = 100 * (positive_goal - negative_goal)

        distance_to_puck_positive = 50 * (
            ((agent.x - puck.x) ** 2 + (agent.y - puck.y) ** 2) ** 0.5
        )

        direction_to_puck = -100 * abs(
            math.atan2(puck.y - agent.y, puck.x - agent.x) - agent.direction
        )

        in_goal_penalty = 0

        if (
            (agent.x**2 + (agent.y - self.config["field"]["height"] / 2) ** 2) ** 0.5
            < self.config["field"]["goal_radius"]
        ) or (
            (
                (agent.x - self.config["field"]["width"]) ** 2
                + (agent.y - self.config["field"]["height"] / 2) ** 2
            )
            ** 0.5
            < self.config["field"]["goal_radius"]
        ):
            in_goal_penalty = -1000

        closeness_to_wall_penalty = 0
        if agent.x < 50 or agent.x > self.config["field"]["width"] - 50:
            closeness_to_wall_penalty -= 10000
        if agent.y < 50 or agent.y > self.config["field"]["height"] - 50:
            closeness_to_wall_penalty -= 10000

        reward = (
            +distance_to_goal_reward
            + direction_to_puck
            + distance_to_puck_positive
            + in_goal_penalty
            + closeness_to_wall_penalty
        ) / 1000

        print(f"Agent: {agent.id}, reward: {reward}")

        return reward

    def calculate_returns(self, rewards):
        returns = []
        culmulative_reward = 0
        for reward in reversed(rewards):
            returns.append(reward + 0.99 * culmulative_reward)

        return list(reversed(returns))

    def save_model(self, path):
        self.agent.save(path)
