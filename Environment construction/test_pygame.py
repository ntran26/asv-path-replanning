import pygame
import numpy as np

class Visualization:
    def __init__(self, width, height, path):
        self.width = width
        self.height = height
        self.path = path
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
    def draw_path(self):
        self.screen.fill((255, 255, 255))  # Fill screen with white color

        # Draw the path
        path_color = (0, 0, 0)  # Black color
        for point in self.path:
            pygame.draw.circle(self.screen, path_color, (int(point[0]), self.height - int(point[1])), 1)

        pygame.display.flip()  # Update the display

    def run_visualization(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.draw_path()
            self.clock.tick(60)  # Limit to 60 frames per second
        pygame.quit()

# Assume self.path has been defined as in the code provided
START = (5, 5)
path = []
for x in range(START[0], 100-10):
    path.append((x, START[1]))
for y in range(START[1], 100-10):
    path.append((100-10, y))

visualization = Visualization(100, 100, path)
visualization.run_visualization()