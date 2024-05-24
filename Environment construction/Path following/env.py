import pygame
import numpy as np
import pandas as pd

# Define colors
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (255,255,0)
CYAN = (0,255,255)

# Define map dimensions
WIDTH = 300
HEIGHT = 200
START = (50, 50)
STEP = 50
UPSCALE_WIDTH = 600
UPSCALE_HEIGHT = 400

class ASVVisualization:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        pygame.init()
        self.screen = pygame.display.set_mode((UPSCALE_WIDTH, UPSCALE_HEIGHT))
        self.surface = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()

    def draw_path(self):      
        self.surface.fill(BLACK)  # Fill the surface     
        self.step = STEP

        pts = []
        i = int(self.width / 100)
        for step in range(0,i):
            if step % 2 == 0:   # go top down
                p = (START[0]+step*100,START[1])
                pts.append(p)
                pygame.draw.circle(self.surface, WHITE, p, 5)
                # bottom horizontal line
                p = (START[0]+step*100,self.height-50)
                pts.append(p)
                pygame.draw.circle(self.surface, WHITE, p, 5)
            else:               # go bottom up
                p = (START[0]+step*100,self.height-50)
                pts.append(p)
                pygame.draw.circle(self.surface, WHITE, p, 5)
                # top horizontal line
                p = (START[0]+step*100,START[1])
                pts.append(p)
                pygame.draw.circle(self.surface, WHITE, p, 5)
        
        path = np.empty((0, 2), int)

        p0, p1, p2, p3, p4, p5 = pts[:6]

        for y in range(p0[1], p1[1]):
            new_point = np.array([[p0[0], y]])
            path = np.append(path, new_point, axis=0)
        for x in range(p1[0], p2[0]):
            new_point = np.array([[x, p1[1]]])
            path = np.append(path, new_point, axis=0)
        for y in range(p3[1], p2[1]):
            new_point = np.array([[p2[0], self.height-y]])      # invert vertical line
            path = np.append(path, new_point, axis=0)
        for x in range(p3[0], p4[0]):
            new_point = np.array([[x, p3[1]]])
            path = np.append(path, new_point, axis=0)
        
        for y in range(p4[1], p5[1]):
            new_point = np.array([[p4[0], y]])
            path = np.append(path, new_point, axis=0)

        for point in path:
            pygame.draw.circle(self.surface, GREEN, (int(point[0]), int(point[1])), 1)

        self.path = path

        num_step = 60
        # Turn right
        self.heading = 90
        self.speed = 1
        self.position = np.array(START, dtype=float) 
        self.step = 0
        pos = np.empty((0, 2), int)

        while self.step < num_step:
            self.position = np.array([self.position[0] + self.speed*np.cos(np.radians(self.heading)),
                                      self.position[1] + self.speed*np.sin(np.radians(self.heading))], dtype = float)
            pos = np.vstack([pos, self.position])
            self.step += 1
            self.heading += 2

        for point in pos:
            pygame.draw.circle(self.surface, BLUE, (int(point[0]), int(point[1])), 1)

        # Go straight
        self.heading = 90
        self.position = np.array(START, dtype=float) 
        self.step = 0
        pos = np.empty((0, 2), int)

        while self.step < num_step:
            self.position = np.array([self.position[0] + self.speed*np.cos(np.radians(self.heading)),
                                      self.position[1] + self.speed*np.sin(np.radians(self.heading))], dtype = float)
            pos = np.vstack([pos, self.position])
            self.step += 1

        for point in pos:
            pygame.draw.circle(self.surface, YELLOW, (int(point[0]), int(point[1])), 1)

        # Turn left
        self.heading = 90
        self.position = np.array(START, dtype=float) 
        self.step = 0
        pos = np.empty((0, 2), int)

        while self.step < num_step:
            self.position = np.array([
                                        self.position[0] + self.speed*np.cos(np.radians(self.heading)),
                                        self.position[1] + self.speed*np.sin(np.radians(self.heading))
                                        ], dtype = float)
            pos = np.vstack([pos, self.position])
            self.step += 1
            self.heading -= 2
        
        for point in pos:
            pygame.draw.circle(self.surface, RED, (int(point[0]), int(point[1])), 1)

        num_step = 60
        # Turn right
        self.heading = 0
        self.speed = 1
        self.position = np.array(p3, dtype=float) 
        self.step = 0
        pos = np.empty((0, 2), int)

        while self.step < num_step:
            self.position = np.array([self.position[0] + self.speed*np.cos(np.radians(self.heading)),
                                      self.position[1] + self.speed*np.sin(np.radians(self.heading))], dtype = float)
            pos = np.vstack([pos, self.position])
            self.step += 1
            self.heading += 2

        for point in pos:
            pygame.draw.circle(self.surface, BLUE, (int(point[0]), int(point[1])), 1)

        # Go straight
        self.heading = 0
        self.position = np.array(p3, dtype=float) 
        self.step = 0
        pos = np.empty((0, 2), int)

        while self.step < num_step:
            self.position = np.array([self.position[0] + self.speed*np.cos(np.radians(self.heading)),
                                      self.position[1] + self.speed*np.sin(np.radians(self.heading))], dtype = float)
            pos = np.vstack([pos, self.position])
            self.step += 1

        for point in pos:
            pygame.draw.circle(self.surface, YELLOW, (int(point[0]), int(point[1])), 1)

        # Turn left
        self.heading = 0
        self.position = np.array(p3, dtype=float) 
        self.step = 0
        pos = np.empty((0, 2), int)

        while self.step < num_step:
            self.position = np.array([
                                        self.position[0] + self.speed*np.cos(np.radians(self.heading)),
                                        self.position[1] + self.speed*np.sin(np.radians(self.heading))
                                        ], dtype = float)
            pos = np.vstack([pos, self.position])
            self.step += 1
            self.heading -= 2
        
        for point in pos:
            pygame.draw.circle(self.surface, RED, (int(point[0]), int(point[1])), 1)

    def run_visualization(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.draw_path()
            
            scaled_surface = pygame.transform.scale(self.surface, (UPSCALE_WIDTH, UPSCALE_HEIGHT))
            self.screen.blit(scaled_surface, (0, 0))

            pygame.display.update()
            self.clock.tick(60)  # Limit to 60 frames per second
        pygame.quit()

visualization = ASVVisualization(WIDTH, HEIGHT)
visualization.run_visualization()
