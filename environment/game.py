import math
import random

import pygame

from agents.agent import Agent
from environment.field import Field
from environment.puck import Puck
from utils.config import Config


class Game:
    def __init__(self, config: Config):
        self.config = config
        self.field = Field(self.config)

        self.score = [0, 0]
        self.frame = 0

        self.visualise = config["visualise"]
        if self.visualise:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (config["field"]["width"], config["field"]["height"])
            )
            pygame.display.set_caption("Hockey Field")
            self.clock = pygame.time.Clock()
            self.score_font = pygame.font.SysFont("Arial", 64)
            self.frame_font = pygame.font.SysFont("Arial", 16)

        self.reset_positions()

    def create_agents(self):
        self.agents = dict()

        # Team 0
        self.agents[0] = Agent(
            200 + (random.random() - 0.5) * 50,
            150 + (random.random() - 0.5) * 50,
            self.config,
            team=0,
            id=0,
            color=(255, 50, 50),
            direction=0,
        )
        self.agents[1] = Agent(
            200 + (random.random() - 0.5) * 50,
            300 + (random.random() - 0.5) * 50,
            self.config,
            team=0,
            id=1,
            color=(255, 50, 100),
            direction=0,
        )

        self.agents[2] = Agent(
            200 + (random.random() - 0.5) * 50,
            450 + (random.random() - 0.5) * 50,
            self.config,
            team=0,
            id=2,
            color=(255, 100, 50),
            direction=0,
        )

        # Team 1
        self.agents[3] = Agent(
            600 + (random.random() - 0.5) * 50,
            150 + (random.random() - 0.5) * 50,
            self.config,
            team=1,
            id=3,
            color=(50, 50, 255),
            direction=math.pi,
        )
        self.agents[4] = Agent(
            600 + (random.random() - 0.5) * 50,
            300 + (random.random() - 0.5) * 50,
            self.config,
            team=1,
            id=4,
            color=(100, 50, 255),
            direction=math.pi,
        )
        self.agents[5] = Agent(
            600 + (random.random() - 0.5) * 50,
            450 + (random.random() - 0.5) * 50,
            self.config,
            team=1,
            id=5,
            color=(50, 100, 255),
            direction=math.pi,
        )

        for agent in self.agents.values():
            agent.direction = self.direction_to_point(
                agent.x, agent.y, self.puck.x, self.puck.y
            )

    def distance_between_points(self, x0, y0, x1, y1):
        return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5

    def direction_to_point(self, x_start, y_start, x_terminal, y_terminal):
        return math.atan2(y_terminal - y_start, x_terminal - x_start)

    def get_observation(self, agent):
        status = []
        distances = []
        positional = []

        # status
        # is_accelerating
        status.append(agent.accelerating)

        # direction
        status.append(agent.direction)

        # distances
        # distance to puck
        distances.append(
            self.distance_between_points(agent.x, agent.y, self.puck.x, self.puck.y)
        )

        # distance to team mates
        team_mates = []
        enemy = []
        for other_agent in self.agents.values():
            if other_agent.id == agent.id:
                continue

            if other_agent.team == agent.team:
                team_mates.append(
                    self.distance_between_points(
                        agent.x, agent.y, other_agent.x, other_agent.y
                    )
                )
            else:
                enemy.append(
                    self.distance_between_points(
                        agent.x, agent.y, other_agent.x, other_agent.y
                    )
                )
        distances.extend(team_mates)
        distances.extend(enemy)

        # positional
        # puck position
        positional.append(self.puck.x)
        positional.append(self.puck.y)

        # own position
        positional.append(agent.x)
        positional.append(agent.y)

        # team and enemy lists
        team_mates = []
        enemy = []

        for other_agent in self.agents.values():
            if other_agent.id == agent.id:
                continue

            if other_agent.team == agent.team:
                team_mates.append(other_agent.x)
                team_mates.append(other_agent.y)
            else:
                enemy.append(other_agent.x)
                enemy.append(other_agent.y)
        positional.extend(team_mates)
        positional.extend(enemy)

        return status, distances, positional

    def reset_positions(self, goal_state=0):
        if goal_state == 1:
            self.score[1] += 1
        elif goal_state == -1:
            self.score[0] += 1
        # self.puck = Puck(
        #     self.config["field"]["width"] / 2,
        #     self.config["field"]["height"] / 2,
        #     self.config,
        # )
        self.puck = Puck(
            self.config["field"]["width"] / 2 + random.gauss(0, 30),
            self.config["field"]["height"] / 2 + random.gauss(0, 15),
            self.config,
        )
        self.create_agents()

        for agent in self.agents.values():
            agent.accelerating = True

    def draw_score(self):
        team0_score = self.score_font.render(str(self.score[0]), True, (255, 0, 0))
        team1_score = self.score_font.render(str(self.score[1]), True, (0, 0, 255))
        frame = self.frame_font.render(str(self.frame), True, (0, 0, 0))
        self.screen.blit(team0_score, (0, 10))
        self.screen.blit(
            team1_score,
            (self.config["field"]["width"] - 40, 10),
        )
        self.screen.blit(frame, (self.config["field"]["width"] // 2 - 20, 10))

    def update(self):
        for agent in self.agents.values():
            # agent.actions(self.puck)
            agent.update(self.agents, self.puck)

        self.puck.update()

        goal_state = self.puck.detect_goal()
        if goal_state:
            self.reset_positions(goal_state)

        self.frame += 1

        return goal_state

    def draw(self):
        self.field.draw_field(self.screen)
        self.draw_score()

        for agent in self.agents.values():
            agent.draw(self.screen)

        self.puck.draw(self.screen)

        pygame.display.flip()
        self.clock.tick(self.config["max_fps"])
        pygame.display.set_caption(f"FPS: {int(self.clock.get_fps())}")

    def run(self):
        running = True

        while running and (
            self.frame < self.config["total_frames"]
            or (self.config["tiebreak"] and self.score[0] == self.score[1])
        ):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.update()
            if self.visualise:
                self.draw()

        print(self.score)
        pygame.quit()
