import pygame
import numpy as np

class ASVVisualization:
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
WIDTH = 900
HEIGHT = 500
START = (50, 100)
path = []
# for x in range(START[0], HEIGHT-50):
#     path.append((START[1],x))
# for y in range(START[1], WIDTH-50):
#     path.append((y,HEIGHT-50))

for x in range(START[0], WIDTH-50):
    for y in range(START[1], HEIGHT-50):
        path.append((x, HEIGHT-50))
        path.append((START[0], y))

visualization = ASVVisualization(WIDTH, HEIGHT, path)
visualization.run_visualization()