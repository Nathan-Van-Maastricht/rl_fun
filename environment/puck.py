import pygame


class Puck:
    def __init__(self, x, y, radius=10, color=(0, 0, 0)):  # White
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.speed_x = 0
        self.speed_y = 0

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

    def update(self):
        self.speed_x *= 0.999
        self.speed_y *= 0.999
        self.x += self.speed_x
        self.y += self.speed_y

        if self.detect_goal():
            return

        # Boundary check (keep puck within field)
        field_x, field_y, field_width, field_height = 40, 0, 720, 600
        if self.x - self.radius < field_x:
            self.x = field_x + self.radius
            self.speed_x *= -1
        elif self.x + self.radius > field_x + field_width:
            self.x = field_x + field_width - self.radius
            self.speed_x *= -1

        if self.y - self.radius < field_y:
            self.y = field_y + self.radius
            self.speed_y *= -1
        elif self.y + self.radius > field_y + field_height:
            self.y = field_y + field_height - self.radius
            self.speed_y *= -1

    def detect_goal(self):
        if self.y < 600 // 2 + 20 and self.y > 600 // 2 - 20:
            if self.x - self.radius < 40 + 1e-3:
                return 1
            if self.x + self.radius > 760 - 1e-3:
                return -1
        return 0
