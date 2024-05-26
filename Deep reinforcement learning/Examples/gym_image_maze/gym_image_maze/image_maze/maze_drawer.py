import numpy as np
import cv2 as cv

class MazeDrawer:
    @classmethod
    def draw_maze(cls, maze, height, width):
        BLACK = (0,0,0)
        img = np.ones((maze.size, maze.size), dtype=np.uint8) * 255
        cv.circle(img, (maze.robot.position), maze.robot.size, BLACK, -1)
        cv.circle(img, (maze.goal), 1, BLACK, -1)
        for wall in maze.walls:
            cv.line(img, wall.a, wall.b, BLACK)
        img = cv.resize(img, (height, width), interpolation=cv.INTER_NEAREST)
        return img