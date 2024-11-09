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

        self.reset_positions()
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

    def create_agents(self):
        self.agents = []

        # Team 1
        self.agents.append(
            Agent(
                200 + (random.random() - 0.5) * 10,
                150 + (random.random() - 0.5) * 10,
                self.config,
                color=(255, 50, 50),
            )
        )
        self.agents.append(
            Agent(
                200 + (random.random() - 0.5) * 10,
                300 + (random.random() - 0.5) * 10,
                self.config,
                color=(255, 50, 100),
            )
        )
        self.agents.append(
            Agent(
                200 + (random.random() - 0.5) * 10,
                450 + (random.random() - 0.5) * 10,
                self.config,
                color=(255, 100, 50),
            )
        )

        # Team 2
        self.agents.append(
            Agent(
                600 + (random.random() - 0.5) * 10,
                150 + (random.random() - 0.5) * 10,
                self.config,
                color=(50, 50, 255),
                direction=math.pi,
            )
        )
        self.agents.append(
            Agent(
                600 + (random.random() - 0.5) * 10,
                300 + (random.random() - 0.5) * 10,
                self.config,
                color=(100, 50, 255),
                direction=math.pi,
            )
        )
        self.agents.append(
            Agent(
                600 + (random.random() - 0.5) * 10,
                450 + (random.random() - 0.5) * 10,
                self.config,
                color=(50, 100, 255),
                direction=math.pi,
            )
        )

        random.shuffle(self.agents)

    def reset_positions(self, goal_state=0):
        if goal_state == 1:
            self.score[1] += 1
        elif goal_state == -1:
            self.score[0] += 1
        self.agents = []
        self.create_agents()
        self.puck = Puck(
            self.config["field"]["width"] / 2,
            self.config["field"]["height"] / 2,
            self.config,
        )

        for agent in self.agents:
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
        for agent in self.agents:
            agent.actions(None, self.puck)
            agent.update(self.agents, self.puck)

        self.puck.update()

        goal_state = self.puck.detect_goal()
        if goal_state:
            self.reset_positions(goal_state)

    def draw(self):
        self.field.draw_field(self.screen)
        self.draw_score()

        for agent in self.agents:
            agent.draw(self.screen)

        self.puck.draw(self.screen)

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

            pygame.display.flip()
            self.clock.tick(self.config["max_fps"])
            pygame.display.set_caption(f"FPS: {int(self.clock.get_fps())}")

            self.frame += 1

        print(self.score)
        pygame.quit()
