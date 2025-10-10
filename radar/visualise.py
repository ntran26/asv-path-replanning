import pygame
import numpy as np
from radar_process import RadarProcess

# Initialize and process radar data
processor = RadarProcess(path_to_data="radar/sample_data", data="polar_scan.png")
ranges_meter = processor.process()
print(ranges_meter)

# Configuration
WIDTH, HEIGHT = 600, 600
BLACK = (0,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
RED = (255,0,0)
SCALE = 1
FPS = 30

# Initialize
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LIDAR Visualisation")
clock = pygame.time.Clock()

# Parameters
lidar_range = ranges_meter
num_beam = len(lidar_range)
cx, cy = WIDTH // 2, HEIGHT // 2

# Plotting
def draw_lidar():
    screen.fill(BLACK)
    pygame.draw.circle(screen, RED, (cx,cy), 5)

    for idx, distance in enumerate(lidar_range):
        angle = 2 * np.pi * idx / num_beam
        x = int(cx + distance*SCALE*np.cos(angle))
        y = int(cy - distance*SCALE*np.sin(angle))
        pygame.draw.line(screen, GREEN, (cx, cy), (x, y), 1)

    pygame.display.update()


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    draw_lidar()
    clock.tick(FPS)

pygame.quit()