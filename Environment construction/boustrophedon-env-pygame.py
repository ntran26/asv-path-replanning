import pygame
import sys
import pandas as pd

# Initialize Pygame
pygame.init()

# Define the dimensions of the 2D environment
width = 700
height = 500
step = 100

# # Define the ship dimensions and speed
# ship_width = 20
# ship_height = 40
speed = 0.2

# Define colors
white = (255,255,255)
red = (255,0,0)
blue = (0,0,255)
black = (0,0,0)
green = (0,255,0)

# Initialize Pygame
pygame.init()

# Define the original dimensions of the coverage path
original_width = 700
original_height = 500

# Define the larger scaled dimensions for the display
scaled_screen_width = 800  # Adjust as desired
scaled_screen_height = 600  # Adjust as desired

# Calculate the offsets for centering the coverage path on the scaled screen
x_offset = (scaled_screen_width - original_width) / 2
y_offset = (scaled_screen_height - original_height) / 2
# Set up the Pygame window with the larger scaled resolution
screen = pygame.display.set_mode((scaled_screen_width, scaled_screen_height))
pygame.display.set_caption('ASV Path Following Simulation')

# Initialize coverage points and ASV position
x_points = []
y_points = []
current_point_index = 0

# Implement the modified Boustrophedon algorithm
for x in range(0, width+1, step):
    for y in range(0, height+1, step):
        if x % (2*step) == 0:
            x_points.append(x)
            y_points.append(y)
        else:
            x_points.append(x)
            y_points.append(height - y)

# Create ASV class for simulating movement
class ASV:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Scale the coverage path to fit the new screen dimensions
x_points = [x + x_offset for x in x_points]
y_points = [y + y_offset for y in y_points]

# Initialize the ASV at the start point
asv = ASV(x_points[0], y_points[0])

# Main loop
while True:
    # Condition to quit the program
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill(black)  # Clear the screen

    # Draw the coverage path
    for i in range(1, len(x_points)):
        pygame.draw.line(screen, blue, (x_points[i-1], y_points[i-1]), (x_points[i], y_points[i]))

    # Draw the ASV
    pygame.draw.circle(screen, red, (asv.x, asv.y), 5)

    # Update ASV's position towards the next point
    if current_point_index < len(x_points):
        next_point = (x_points[current_point_index], y_points[current_point_index])
        dx = next_point[0] - asv.x
        dy = next_point[1] - asv.y
        distance_to_next_point = (dx ** 2 + dy ** 2) ** 0.5
        if distance_to_next_point > speed:
            unit_dx = speed * dx / distance_to_next_point
            unit_dy = speed * dy / distance_to_next_point
            asv.x += unit_dx
            asv.y += unit_dy
        else:
            current_point_index += 1

    pygame.display.flip()  # Update the display
