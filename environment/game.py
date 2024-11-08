import math
import random

import pygame

from agents.agent import Agent
from environment.field import Field
from environment.puck import Puck


class Game:
    def __init__(self, width=800, height=600, goal_width=40, visualise=True):
        self.field = Field(width, height, goal_width)
        self.agents = []
        self.puck = Puck(400, 300)
        self.create_agents()
        self.score = [0, 0]
        self.frame = 0
        self.visualise = visualise
        if visualise:
            pygame.init()
            self.width = width
            self.height = height
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Hockey Field")
            self.clock = pygame.time.Clock()
            self.score_font = pygame.font.SysFont("Arial", 64)
            self.frame_font = pygame.font.SysFont("Arial", 16)

    def create_agents(self):
        # Team 1
        self.agents.append(
            Agent(
                200 + random.random() - 0.5,
                150 + random.random() - 0.5,
                color=(255, 50, 50),
            )
        )
        self.agents.append(
            Agent(
                200 + random.random() - 0.5,
                300 + random.random() - 0.5,
                color=(255, 50, 100),
            )
        )
        self.agents.append(
            Agent(
                200 + random.random() - 0.5,
                450 + random.random() - 0.5,
                color=(255, 100, 50),
            )
        )

        # Team 2
        self.agents.append(
            Agent(
                600 + random.random() - 0.5,
                150 + random.random() - 0.5,
                color=(50, 50, 255),
                direction=math.pi,
            )
        )
        self.agents.append(
            Agent(
                600 + random.random() - 0.5,
                300 + random.random() - 0.5,
                color=(100, 50, 255),
                direction=math.pi,
            )
        )
        self.agents.append(
            Agent(
                600 + random.random() - 0.5,
                450 + random.random() - 0.5,
                color=(50, 100, 255),
                direction=math.pi,
            )
        )

        random.shuffle(self.agents)

    def reset(self, goal_state):
        if goal_state == 1:
            self.score[1] += 1
        else:
            self.score[0] += 1
        self.agents.clear()
        self.create_agents()
        self.puck = Puck(400, 300)

    def draw_score(self):
        team0_score = self.score_font.render(str(self.score[0]), True, (255, 0, 0))
        team1_score = self.score_font.render(str(self.score[1]), True, (0, 0, 255))
        frame = self.frame_font.render(str(self.frame), True, (0, 0, 0))
        self.screen.blit(team0_score, (10, 10))
        self.screen.blit(team1_score, (760, 10))
        self.screen.blit(frame, (385, 10))

    def run(self):
        running = True
        for agent in self.agents:
            agent.accelerating = True

        while running and self.frame < 25000:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if self.visualise:
                self.field.draw_field(self.screen)
                self.draw_score()

            for agent in self.agents:
                if self.frame % 20 == 0:
                    # agent.direction = random.random() * math.pi * 2 - math.pi
                    if (
                        random.random() < 0.05
                        if agent.accelerating
                        else random.random() < 0.1
                    ):
                        agent.accelerating = not agent.accelerating
                agent.update(self.agents, self.puck)
                if self.visualise:
                    agent.draw(self.screen)
            self.puck.update()
            if self.visualise:
                self.puck.draw(self.screen)
            goal_state = self.field.detect_goal(self.puck)

            if goal_state:
                self.reset(goal_state)

            pygame.display.flip()
            self.clock.tick(240)
            pygame.display.set_caption(f"FPS: {int(self.clock.get_fps())}")

            self.frame += 1

        print(self.score)
        pygame.quit()
