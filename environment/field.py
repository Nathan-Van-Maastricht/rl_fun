import pygame

class Field:
    def __init__(self, width=800, height=600, goal_width=40):
        self.goal_width = goal_width
        self.field_width = width - 2 * goal_width
        self.height = height
        self.width = width
        self.field_x = goal_width
        self.field_y = 0

    def draw_field(self, screen):
        screen.fill((0, 0, 0))  # Black background

        # Draw field
        pygame.draw.rect(screen, (0, 150, 0), (self.field_x, self.field_y, self.field_width, self.height))

        # Draw goals
        pygame.draw.rect(screen, (255, 255, 255), (0, self.height // 2 - 20, 40, 40))
        pygame.draw.rect(screen, (255, 255, 255), (self.width - 40, self.height // 2 - 20, 40, 40))

        # Draw center line
        pygame.draw.line(screen, (255, 255, 255), (self.field_x + self.field_width // 2, 0), (self.field_x + self.field_width // 2, self.height))

    def detect_goal(self, puck):
        if puck.y < self.height // 2 + 20 and puck.y > self.height // 2 - 20:
            if puck.x - puck.radius < self.goal_width + 1e-3:
                return 1
            if puck.x + puck.radius > self.field_width + self.goal_width - 1e-3:
                return -1
        return 0