import pygame


class Puck:
    def __init__(self, x, y, config, radius=10, color=(0, 0, 0)):
        self.x = x
        self.y = y
        self.config = config
        self.radius = radius
        self.color = color
        self.speed_x = 0
        self.speed_y = 0
        self.max_speed = self.config["puck"]["max_speed"]

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

    def update(self):
        if (speed := (self.speed_x**2 + self.speed_y**2) ** 0.5) >= self.max_speed:
            self.speed_x = self.speed_x / speed * self.max_speed
            self.speed_y = self.speed_y / speed * self.max_speed
        self.speed_x *= self.config["friction"]
        self.speed_y *= self.config["friction"]
        self.x += self.speed_x
        self.y += self.speed_y

        if self.detect_goal():
            return

        # Boundary check (keep puck within field)
        field_x, field_y, field_width, field_height = (
            0,
            0,
            self.config["field"]["width"],
            self.config["field"]["height"],
        )
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

    def distance_to_goal(self, goal_number):
        if goal_number == 0:
            return (
                (self.y - self.config["field"]["height"] // 2) ** 2 + self.x**2
            ) ** 0.5
        else:
            return (
                (self.y - self.config["field"]["height"] // 2) ** 2
                + (self.x - self.config["field"]["width"]) ** 2
            ) ** 0.5

    def detect_goal(self):
        distance_to_goal_0 = self.distance_to_goal(0)

        if distance_to_goal_0 < self.config["field"]["goal_radius"]:
            return 1

        distance_to_goal_1 = self.distance_to_goal(1)

        if distance_to_goal_1 < self.config["field"]["goal_radius"]:
            return -1

        return 0
