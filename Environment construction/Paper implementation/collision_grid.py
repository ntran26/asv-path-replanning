import pygame
import numpy as np
import math

# Initialize Pygame
pygame.init()

# Define screen dimensions and grid parameters
screen_width, screen_height = 600, 600
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

# Define grid parameters
grid_size = 40
observation_radius = 200
agent_position = (400, 400)

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
TRANSPARENT_RED = (255, 0, 0, 100)

# Function to generate the collision grid
def generate_grid(agent_pos, radius, grid_size):
    grid = []
    for i in range(-radius, radius, grid_size):
        for j in range(-radius, radius, grid_size):
            if math.sqrt(i**2 + j**2) < radius:
                grid.append((agent_pos[0] + i, agent_pos[1] + j))
    return grid

# Function to update grid with dynamic obstacles
def update_grid(grid, obstacles, future_steps=3):
    updated_grid = {cell: 0 for cell in grid}
    for obs in obstacles:
        x, y = obs['pos']
        updated_grid[(x, y)] = 1.0  # Current position of obstacle
        for step in range(1, future_steps + 1):
            future_x = x + step * obs['velocity'] * math.cos(obs['direction'])
            future_y = y + step * obs['velocity'] * math.sin(obs['direction'])
            updated_grid[(int(future_x), int(future_y))] = 1.0 * (0.85 ** step)
    return updated_grid

# Function to draw the grid
def draw_grid(grid, updated_grid):
    for cell, weight in updated_grid.items():
        if weight > 0:
            color = (255, 0, 0, int(255 * weight))
            pygame.draw.rect(screen, color, (*cell, grid_size, grid_size), 0)

# Function to draw a circle with transparency
def draw_transparent_circle(surface, color, position, radius):
    target_rect = pygame.Rect(position[0] - radius, position[1] - radius, radius * 2, radius * 2)
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.circle(shape_surf, color, (radius, radius), radius)
    surface.blit(shape_surf, target_rect)

# Example dynamic obstacles
obstacles = [
    {'pos': [350, 350], 'velocity': 2, 'direction': math.pi / 4},
    {'pos': [450, 450], 'velocity': 3, 'direction': -math.pi / 4}
]

# Static obstacles
static_obstacle = (300, 300)

# Objective
objective = (500, 500)

# Main loop
running = True
while running:
    screen.fill(WHITE)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Move obstacles
    for obs in obstacles:
        obs['pos'][0] += obs['velocity'] * math.cos(obs['direction'])
        obs['pos'][1] += obs['velocity'] * math.sin(obs['direction'])
    
    grid = generate_grid(agent_position, observation_radius, grid_size)
    updated_grid = update_grid(grid, obstacles)
    
    # Draw the objective
    draw_transparent_circle(screen, GREEN, objective, 10)
    
    # Draw static obstacle
    draw_transparent_circle(screen, YELLOW, static_obstacle, 10)
    
    # Draw agent
    draw_transparent_circle(screen, BLUE, agent_position, 10)
    
    # Draw the grid with updated weights
    draw_grid(grid, updated_grid)
    
    # Draw dynamic obstacles
    for obs in obstacles:
        draw_transparent_circle(screen, RED, (int(obs['pos'][0]), int(obs['pos'][1])), 10)
    
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
