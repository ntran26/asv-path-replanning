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
WIDTH = 900
HEIGHT = 500
START = (50, 50)
STEP = 100

class ASVVisualization:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # self.path = path
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

    def draw_path(self):      
        self.screen.fill(BLACK)  # Fill the screen     

        pts = []
        i = int(self.width / 100)
        for step in range(0,i):
            if step % 2 == 0:   # go top down
                p = (START[0]+step*100,START[1])
                pts.append(p)
                pygame.draw.circle(self.screen, WHITE, p, 5)
                # bottom horizontal line
                p = (START[0]+step*100,self.height-50)
                pts.append(p)
                pygame.draw.circle(self.screen, WHITE, p, 5)
            else:               # go bottom up
                p = (START[0]+step*100,self.height-50)
                pts.append(p)
                pygame.draw.circle(self.screen, WHITE, p, 5)
                # top horizontal line
                p = (START[0]+step*100,START[1])
                pts.append(p)
                pygame.draw.circle(self.screen, WHITE, p, 5)
        
        path = np.empty((0, 2), int)

        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17 = pts[:18]

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
        for x in range(p5[0], p6[0]):
            new_point = np.array([[x, p5[1]]])
            path = np.append(path, new_point, axis=0)
        for y in range(p7[1], p6[1]):
            new_point = np.array([[p6[0], self.height-y]])
            path = np.append(path, new_point, axis=0)
        for x in range(p7[0], p8[0]):
            new_point = np.array([[x, p7[1]]])
            path = np.append(path, new_point, axis=0)

        for y in range(p8[1], p9[1]):
            new_point = np.array([[p8[0], y]])
            path = np.append(path, new_point, axis=0)
        for x in range(p9[0], p10[0]):
            new_point = np.array([[x, p9[1]]])
            path = np.append(path, new_point, axis=0)
        for y in range(p11[1], p10[1]):
            new_point = np.array([[p10[0], self.height-y]])
            path = np.append(path, new_point, axis=0)
        for x in range(p11[0], p12[0]):
            new_point = np.array([[x, p11[1]]])
            path = np.append(path, new_point, axis=0)

        for y in range(p12[1], p13[1]):
            new_point = np.array([[p12[0], y]])
            path = np.append(path, new_point, axis=0)
        for x in range(p13[0], p14[0]):
            new_point = np.array([[x, p13[1]]])
            path = np.append(path, new_point, axis=0)
        for y in range(p15[1], p14[1]):
            new_point = np.array([[p14[0], self.height-y]])
            path = np.append(path, new_point, axis=0)
        for x in range(p15[0], p16[0]):
            new_point = np.array([[x, p15[1]]])
            path = np.append(path, new_point, axis=0)
        
        for y in range(p16[1], p17[1]):
            new_point = np.array([[p16[0], y]])
            path = np.append(path, new_point, axis=0)

        for point in path:
            pygame.draw.circle(self.screen, GREEN, (int(point[0]), int(point[1])), 1)

        self.path = path

        num_step = 30
        # Turn right
        self.heading = 90
        self.speed = 2
        self.position = np.array(START, dtype=float) 
        self.step = 0
        pos = np.empty((0, 2), int)

        while self.step < num_step:
            self.position = np.array([self.position[0] + self.speed*np.cos(np.radians(self.heading)),
                                      self.position[1] + self.speed*np.sin(np.radians(self.heading))], dtype = float)
            # pygame.draw.circle(self.screen, BLUE, self.position, 1)
            pos = np.vstack([pos, self.position])
            self.step += 1
            self.heading += 5

        for point in pos:
            pygame.draw.circle(self.screen, BLUE, (int(point[0]), int(point[1])), 1)

        # Go straight
        self.heading = 90
        self.speed = 2
        self.position = np.array(START, dtype=float) 
        self.step = 0
        pos = np.empty((0, 2), int)

        while self.step < num_step:
            self.position = np.array([self.position[0] + self.speed*np.cos(np.radians(self.heading)),
                                      self.position[1] + self.speed*np.sin(np.radians(self.heading))], dtype = float)
            # pygame.draw.circle(self.screen, BLUE, self.position, 1)
            pos = np.vstack([pos, self.position])
            self.step += 1

        for point in pos:
            pygame.draw.circle(self.screen, YELLOW, (int(point[0]), int(point[1])), 1)

        # Turn left
        self.heading = 90
        self.speed = 2
        self.position = np.array(START, dtype=float) 
        self.step = 0
        pos = np.empty((0, 2), int)

        while self.step < num_step:
            self.position = np.array([
                                        self.position[0] + self.speed*np.cos(np.radians(self.heading)),
                                        self.position[1] + self.speed*np.sin(np.radians(self.heading))
                                        ], dtype = float)
            # pygame.draw.circle(self.screen, BLUE, self.position, 1)
            pos = np.vstack([pos, self.position])
            self.step += 1
            self.heading -= 5
        
        for point in pos:
            pygame.draw.circle(self.screen, RED, (int(point[0]), int(point[1])), 1)

        # df = pd.DataFrame(path)
        # df.to_csv("path.csv")
        pygame.display.update()  # Update the display

    def run_visualization(self):
        running = True
        index = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.draw_path()

            # # Animate the dot
            # if index < len(self.path):
            #     pygame.draw.circle(self.screen, BLUE, (self.path[index][0], self.path[index][1]), 5)
            #     index += 10

            pygame.display.update()
            # self.clock.tick(60)  # Limit to 60 frames per second
        pygame.quit()

visualization = ASVVisualization(WIDTH, HEIGHT)
visualization.run_visualization()