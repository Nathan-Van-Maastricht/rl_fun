import pygame


class Field:
    def __init__(self, config):
        self.config = config
        self.goal_width = self.config["field"]["goal_width"]
        self.goal_height = self.config["field"]["goal_height"]
        self.height = self.config["field"]["height"]
        self.width = self.config["field"]["width"]
        self.field_width = self.width - 2 * self.goal_width
        self.field_x = self.goal_width
        self.field_y = 0

    def draw_field(self, screen):
        screen.fill((0, 0, 0))  # Black background

        # Draw field
        pygame.draw.rect(
            screen,
            self.config["field"]["color"],
            (self.field_x, self.field_y, self.field_width, self.height),
        )

        # Draw goals
        pygame.draw.rect(
            screen,
            self.config["field"]["goal_color"],
            (
                0,
                self.height // 2 - self.goal_height // 2,
                self.goal_width,
                self.goal_height,
            ),
        )
        pygame.draw.rect(
            screen,
            self.config["field"]["goal_color"],
            (
                self.width - self.goal_width,
                self.height // 2 - self.goal_height // 2,
                self.goal_width,
                self.goal_height,
            ),
        )

        # Draw center line
        pygame.draw.line(
            screen,
            (255, 255, 255),
            (self.field_x + self.field_width // 2, 0),
            (self.field_x + self.field_width // 2, self.height),
        )
