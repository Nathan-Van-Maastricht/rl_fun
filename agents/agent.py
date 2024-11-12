import math
import random

import pygame


class Agent:
    def __init__(self, x, y, config, team, id, color=(255, 0, 0), direction=0):
        self.x = x
        self.y = y
        self.config = config
        self.radius = config["agent"]["radius"]
        self.color = color
        self.speed_x = 0
        self.speed_y = 0
        self.direction = direction  # Radians, 0 = right, pi/2 = up
        self.accelerating = False
        self.max_speed = config["agent"]["max_speed"]
        self.team = team
        self.id = id
        self.reward = 0
        self.last_direction = 0
        self.touched_puck = 0
        if self.config["visualise"]:
            self.id_font = pygame.font.SysFont("Arial", 20, bold=True)
            self.reward_font = pygame.font.SysFont("Arial", 20, bold=True)

    def draw(self, screen):
        colour = self.color
        if self.touched_puck > 0:
            colour = (0, 0, 0)

        pygame.draw.circle(screen, colour, (int(self.x), int(self.y)), self.radius)

        # Draw ID
        id = self.id_font.render(str(self.id), True, (0, 0, 0))
        screen.blit(id, (self.x, self.y - 10))

        # Draw reward
        reward = self.reward_font.render(f"{self.reward:.2f}", True, (0, 0, 0))
        screen.blit(reward, (self.x, self.y + 10))

        # Draw last direction indicator
        colour = (255, 255, 255)
        position = (self.x - 10, self.y - 10, 20, 10)
        match self.last_direction:
            case -2:
                colour = (0, 255, 0)
                position = (self.x - 10, self.y - 10, 10, 10)
            case -1:
                colour = (255, 0, 255)
                position = (self.x - 10, self.y - 10, 10, 10)
            case 1:
                colour = (255, 0, 255)
                position = (self.x, self.y - 10, 10, 10)
            case 2:
                colour = (0, 255, 0)
                position = (self.x, self.y - 10, 10, 10)
        pygame.draw.rect(
            screen,
            colour,
            position,
        )

        # Draw direction indicator
        end_x = self.x + math.cos(self.direction) * self.radius
        end_y = self.y + math.sin(self.direction) * self.radius
        direction_colour = (255, 255, 255) if self.accelerating else (0, 0, 0)
        pygame.draw.line(screen, direction_colour, (self.x, self.y), (end_x, end_y), 2)

    def actions(self, puck):
        new_dir = math.atan2(puck.y - self.y, puck.x - self.x)
        self.direction = new_dir
        if random.random() < 0.001:
            self.accelerating = not self.accelerating

    def action(self, accelerating, direction):
        self.last_direction = direction
        self.direction = (
            self.direction + direction / 57
        )  # make it roughly 1 degree of rotation
        while self.direction < -math.pi:
            self.direction += 2 * math.pi
        while self.direction > math.pi:
            self.direction -= 2 * math.pi
        self.accelerating = accelerating

    def update(self, agents, puck):
        if self.accelerating:
            self.speed_x += math.cos(self.direction)
            self.speed_y += math.sin(self.direction)
        else:
            # Slow down when not accelerating
            speed = math.sqrt(self.speed_x**2 + self.speed_y**2)
            if speed < 0.25:  # Stop when speed is negligible
                self.speed_x = 0
                self.speed_y = 0
            elif speed > 0:
                self.speed_x *= self.config["friction"]  # Friction factor
                self.speed_y *= self.config["friction"]

        # Agent-agent collision detection
        for other_agent in agents.values():
            if other_agent == self:
                continue
            dx = self.x - other_agent.x
            dy = self.y - other_agent.y
            distance = math.sqrt(dx**2 + dy**2)
            if distance < 2 * self.radius:  # Collision detected
                # Calculate collision normal
                normal_x = dx / distance
                normal_y = dy / distance

                # Resolve overlap
                overlap = 2 * self.radius - distance
                self.x += normal_x * overlap
                self.y += normal_y * overlap
                other_agent.x -= normal_x * overlap / 2
                other_agent.y -= normal_y * overlap / 2

                # Update velocities (elastic collision)
                v1n = self.speed_x * normal_x + self.speed_y * normal_y
                v2n = other_agent.speed_x * normal_x + other_agent.speed_y * normal_y
                v1t = self.speed_x - v1n * normal_x
                v2t = other_agent.speed_x - v2n * normal_x
                v1n, v2n = v2n, v1n  # Swap normal components
                self.speed_x = v1n * normal_x + v1t
                self.speed_y = v1n * normal_y + v2t
                other_agent.speed_x = v2n * normal_x + v2t
                other_agent.speed_y = v2n * normal_y

        # Agent-puck collision detection
        dx = puck.x - self.x
        dy = puck.y - self.y
        distance = math.sqrt(dx**2 + dy**2)
        self.touched_puck -= 1
        if distance < puck.radius + self.radius:  # Collision detected
            self.touched_puck = 10
            # Calculate collision normal
            normal_x = dx / distance
            normal_y = dy / distance

            # Resolve overlap
            overlap = puck.radius + self.radius - distance
            puck.x += normal_x * overlap / 2
            puck.y += normal_y * overlap / 2

            # Update velocities (elastic collision)
            v1n = puck.speed_x * normal_x + puck.speed_y * normal_y
            v2n = self.speed_x * normal_x + self.speed_y * normal_y
            v1t = puck.speed_x - v1n * normal_x
            v2t = self.speed_x - v2n * normal_x
            v1n, v2n = v2n, v1n  # Swap normal components
            puck.speed_x = v1n * normal_x + v1t
            puck.speed_y = v1n * normal_y + v2t
            self.speed_x = v2n * normal_x + v2t
            self.speed_y = v2n * normal_y

        # Limit speed
        speed = math.sqrt(self.speed_x**2 + self.speed_y**2)
        if speed > self.max_speed:
            self.speed_x = self.speed_x / speed * self.max_speed
            self.speed_y = self.speed_y / speed * self.max_speed

        self.x += self.speed_x
        self.y += self.speed_y

        # Boundary check (keep agent within field)
        field_x, field_y, field_width, field_height = (
            0,
            0,
            self.config["field"]["width"],
            self.config["field"]["height"],
        )
        if self.x - self.radius < field_x:
            self.x = field_x + self.radius
            self.speed_x *= -self.config["agent"]["reflect_energy_loss"]
        elif self.x + self.radius > field_x + field_width:
            self.x = field_x + field_width - self.radius
            self.speed_x *= -self.config["agent"]["reflect_energy_loss"]

        if self.y - self.radius < field_y:
            self.y = field_y + self.radius
            self.speed_y *= -self.config["agent"]["reflect_energy_loss"]
        elif self.y + self.radius > field_y + field_height:
            self.y = field_y + field_height - self.radius
            self.speed_y *= -self.config["agent"]["reflect_energy_loss"]
