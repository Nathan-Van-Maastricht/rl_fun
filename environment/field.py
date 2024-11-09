import pygame


class Field:
    def __init__(self, config):
        self.config = config
        self.height = self.config["field"]["height"]
        self.width = self.config["field"]["width"]

    def draw_field(self, screen):
        screen.fill((0, 0, 0))  # Black background

        # Draw field
        pygame.draw.rect(
            screen,
            self.config["field"]["color"],
            (0, 0, self.width, self.height),
        )

        # Draw circle goals
        pygame.draw.circle(
            screen,
            self.config["field"]["goal_color0"],
            (0, self.config["field"]["height"] // 2),
            self.config["field"]["goal_radius"],
        )
        pygame.draw.circle(
            screen,
            self.config["field"]["goal_color1"],
            (self.config["field"]["width"], self.config["field"]["height"] // 2),
            self.config["field"]["goal_radius"],
        )

        # Draw goals
        # pygame.draw.rect(
        #     screen,
        #     self.config["field"]["goal_color0"],
        #     (
        #         0,
        #         self.height // 2 - self.goal_height // 2,
        #         self.goal_width,
        #         self.goal_height,
        #     ),
        # )
        # pygame.draw.rect(
        #     screen,
        #     self.config["field"]["goal_color1"],
        #     (
        #         self.width - self.goal_width,
        #         self.height // 2 - self.goal_height // 2,
        #         self.goal_width,
        #         self.goal_height,
        #     ),
        # )

        # Draw center line
        pygame.draw.line(
            screen,
            (255, 255, 255),
            (self.width // 2, 0),
            (self.width // 2, self.height),
        )
