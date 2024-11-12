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
            self.agent.parameters(), lr=0.00001, weight_decay=1e-5
        )

        self.epoch = 0
        self.epsilon = 0.5
        self.epsilon_decay = 0.025
        self.epsilon_min = 0.1

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
            agents_not_used = set()
            for agent in game.agents.values():
                # observe
                state = torch.FloatTensor(game.get_observation(agent.id)).to(
                    self.device
                )

                # act
                accelerating_probabilities, direction_probabilities = self.agent(state)

                if random.random() < self.epsilon:
                    total_explore += 1
                    # agents_not_used.add(agent)
                    direction = torch.randint(
                        direction_probabilities.shape[0], (1,)
                    ).to(self.device)
                    accelerating = torch.randint(
                        accelerating_probabilities.shape[0], (1,)
                    ).to(self.device)
                    probabilities.append(
                        [
                            accelerating_probabilities[accelerating],
                            direction_probabilities[direction],
                        ]
                    )
                    actions.append([accelerating, direction])
                else:
                    total_exploit += 1
                    direction = direction_probabilities.multinomial(1)
                    accelerating = accelerating_probabilities.multinomial(1)
                    probabilities.append(
                        [
                            accelerating_probabilities[accelerating],
                            direction_probabilities[direction],
                        ]
                    )
                    actions.append([accelerating, direction])

                agent.action(accelerating.item(), direction.item() - 2)

            # update state
            goal_state = game.update()
            if self.config["visualise"]:
                game.draw()

            # determine rewards
            for agent in game.agents.values():
                if agent in agents_not_used:
                    continue
                rewards.append(self.reward(game.puck, goal_state, agent))
                agent.reward = rewards[-1]

            frame += 1
            if frame + 1 == self.config["total_frames"]:
                break

        print(f"{self.epsilon=:.2f}")
        print(f"{total_explore=}")
        print(f"{total_exploit=}")
        print(f"Explore ratio: {100*total_explore/(total_explore + total_exploit):.2f}")
        print(f"Exploit ratio: {100*total_exploit/(total_explore+total_exploit):.2f}")

        pygame.quit()

        self.update_network(actions, rewards, probabilities)
        self.epoch += 1

        print("finished training")
        print(f"Score: {game.score}")

    def update_network(self, actions, rewards, probabilities):
        print(f"{self.epoch}: Updating network")
        returns = self.calculate_returns(rewards)
        policy_loss = 0

        log_probabilities = 0
        for action_probabilities, action, reward in zip(
            probabilities, actions, returns
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
        if agent.team == 0:
            target_distance = puck.distance_to_goal(1)
            own_distance = puck.distance_to_goal(0)
        else:
            target_distance = puck.distance_to_goal(0)
            own_distance = puck.distance_to_goal(1)
        positive_goal = 0
        negative_goal = 0

        if target_distance < 4 * self.config["field"]["width"] / 9:
            positive_goal = target_distance
        if own_distance < 5 * self.config["field"]["width"] / 9:
            negative_goal = own_distance

        distance_to_goal_reward = (
            ((-1) ** (negative_goal > positive_goal))
            * 10
            * (positive_goal - negative_goal) ** 2
        )

        # print(
        #     f"puck: {puck.x}, {puck.y}, distance0: {distance_to_goal0}, distance1: {distance_to_goal1}, team: {agent.team}, positive: {positive_goal}, negative: {negative_goal}, distance goal reward: {distance_to_goal_reward}"
        # )

        distance_to_puck = ((agent.x - puck.x) ** 2 + (agent.y - puck.y) ** 2) ** 0.5

        distance_to_puck_reward = -10 * distance_to_puck**2.5
        if distance_to_puck < 10:
            distance_to_goal_reward += 5000

        # direction_to_puck = -100 * abs(
        #     math.atan2(puck.y - agent.y, puck.x - agent.x) - agent.direction
        # )

        # in_goal_penalty = 0

        # if (
        #     (agent.x**2 + (agent.y - self.config["field"]["height"] / 2) ** 2) ** 0.5
        #     < self.config["field"]["goal_radius"]
        # ) or (
        #     (
        #         (agent.x - self.config["field"]["width"]) ** 2
        #         + (agent.y - self.config["field"]["height"] / 2) ** 2
        #     )
        #     ** 0.5
        #     < self.config["field"]["goal_radius"]
        # ):
        #     in_goal_penalty = -100000

        closeness_to_wall_penalty = 0
        if (agent.x < (50 + self.config["puck"]["radius"])) or (
            agent.x
            > self.config["field"]["width"] - (50 + self.config["puck"]["radius"])
        ):
            closeness_to_wall_penalty -= 10000
        if (agent.y < (50 + self.config["puck"]["radius"])) or (
            agent.y
            > self.config["field"]["height"] - (50 + self.config["puck"]["radius"])
        ):
            closeness_to_wall_penalty -= 1000000

        reward = (
            +distance_to_goal_reward
            + distance_to_puck_reward
            # direction_to_puck +
            # + in_goal_penalty
            + closeness_to_wall_penalty
        ) / 100000

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
