import pygame
import numpy as np
import pandas as pd

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
        self.screen.fill((255, 255, 255))  # Fill screen with white color      

        pts = []
        i = 9
        for step in range(0,i):
            if step % 2 == 0:   # go top down
                p = (START[0]+step*100,START[1])
                pts.append(p)
                pygame.draw.circle(self.screen, (255,0,0), p, 5)

                p = (START[0]+step*100,self.height-50)
                pts.append(p)
                pygame.draw.circle(self.screen, (255,0,0), p, 5)
            else:               # go bottom up
                p = (START[0]+step*100,self.height-50)
                pts.append(p)
                pygame.draw.circle(self.screen, (255,0,0), p, 5)
 
                p = (START[0]+step*100,START[1])
                pts.append(p)
                pygame.draw.circle(self.screen, (255,0,0), p, 5)
        
        # Define a function to add new points to the path
        def add_points(path, start, end, axis):
            if axis == 'x':
                for x in range(start[0], end[0]):
                    new_point = np.array([[x, start[1]]])
                    path = np.append(path, new_point, axis=0)
            elif axis == 'y':
                for y in range(start[1], end[1]):
                    new_point = np.array([[start[0], y]])
                    path = np.append(path, new_point, axis=0)
            return path
        
        path = np.empty((0, 2), int)

        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17 = pts[:18]

        # for y in range(p0[1], p1[1]):
        #     new_point = np.array([[p0[0], y]])
        #     path = np.append(path, new_point, axis=0)
        # for x in range(p1[0], p2[0]):
        #     new_point = np.array([[x, p3[1]]])
        #     path = np.append(path, new_point, axis=0)
        # for y in range(p3[1], p2[1]):
        #     new_point = np.array([[p2[0], y]])
        #     path = np.append(path, new_point, axis=0)
        # for x in range(p3[0], p4[0]):
        #     new_point = np.array([[x, p2[1]]])
        #     path = np.append(path, new_point, axis=0)
        
        # for y in range(p4[1], p5[1]):
        #     new_point = np.array([[p4[0], y]])
        #     path = np.append(path, new_point, axis=0)
        # for x in range(p5[0], p6[0]):
        #     new_point = np.array([[x, p3[1]]])
        #     path = np.append(path, new_point, axis=0)
        # for y in range(p7[1], p6[1]):
        #     new_point = np.array([[p6[0], y]])
        #     path = np.append(path, new_point, axis=0)
        # for x in range(p7[0], p8[0]):
        #     new_point = np.array([[x, p6[1]]])
        #     path = np.append(path, new_point, axis=0)

        # Add points to the path using the function
        path = add_points(path, p0, p1, 'y')
        path = add_points(path, p1, p2, 'x')
        path = add_points(path, p3, p2, 'y')
        path = add_points(path, p3, p4, 'x')

        path = add_points(path, p4, p5, 'y')
        path = add_points(path, p5, p6, 'x')
        path = add_points(path, p7, p6, 'y')
        path = add_points(path, p7, p8, 'x')

        path = add_points(path, p8, p9, 'y')
        path = add_points(path, p9, p10, 'x')
        path = add_points(path, p11, p10, 'y')
        path = add_points(path, p11, p12, 'x')

        path = add_points(path, p12, p13, 'y')
        path = add_points(path, p13, p14, 'x')
        path = add_points(path, p15, p14, 'y')
        path = add_points(path, p15, p16, 'x')

        path = add_points(path, p16, p17, 'y')

        for point in path:
            pygame.draw.circle(self.screen, (0,0,0), (int(point[0]), self.height - int(point[1])), 1)

        df = pd.DataFrame(path)
        df.to_csv("path.csv")
        pygame.display.update()  # Update the display

    def run_visualization(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.draw_path()
            self.clock.tick(60)  # Limit to 60 frames per second
        pygame.quit()

# for x in range(START[0], HEIGHT-51):
#     path.append((x, HEIGHT-50))
    # for y in range(START[1], HEIGHT-50):
    #     path.append((START[0], y))

# for y in range(START[1], HEIGHT-STEP+1):
#     new_point = np.array([[START[0], y]])
#     path = np.append(path, new_point, axis=0)
# for x in range(START[0], START[0]+STEP+1):
#     new_point = np.array([[x, 50]])
#     path = np.append(path, new_point, axis=0)

visualization = ASVVisualization(WIDTH, HEIGHT)
visualization.run_visualization()