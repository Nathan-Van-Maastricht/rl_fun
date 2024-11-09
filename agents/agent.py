import pygame
import math


class Agent:
    def __init__(self, x, y, config, color=(255, 0, 0), direction=0):
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

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

        # Draw direction indicator
        end_x = self.x + math.cos(self.direction) * self.radius
        end_y = self.y + math.sin(self.direction) * self.radius
        pygame.draw.line(screen, (255, 255, 255), (self.x, self.y), (end_x, end_y), 2)

    def update(self, agents, puck):
        new_dir = math.atan2(puck.y - self.y, puck.x - self.x)
        self.direction = new_dir
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
        for other_agent in agents:
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
        if distance < puck.radius + self.radius:  # Collision detected
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
            self.config["field"]["goal_width"],
            0,
            self.config["field"]["width"] - 2 * self.config["field"]["goal_width"],
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
