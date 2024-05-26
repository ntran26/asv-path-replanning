import pyglet
import numpy as np
import ctypes
import cv2 as cv
from pyglet.window import Window, key
from gym_image_maze.image_maze.maze import Maze
from gym_image_maze.image_maze.models import Action
from gym_image_maze.image_maze.renderer import Renderer
from gym_image_maze.image_maze.maze_drawer import MazeDrawer

class MazeGame(Window):
    def __init__(self, maze: Maze, height: int, width: int):
        display = pyglet.canvas.get_display()
        screen = display.get_screens()[0]
        config = screen.get_best_config()
        context = config.create_context(None)
        super().__init__(width=width, height=height, config=config, context=context)

        self.maze = maze
        self.width = width
        self.height = height
        self.renderer = Renderer(self)
        self.alive = True

    def on_draw(self):
        self.render()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.MOTION_UP:
            self.maze.move_robot(Action.Up)
        elif symbol == key.MOTION_RIGHT:
            self.maze.move_robot(Action.Right)
        elif symbol == key.MOTION_LEFT:
            self.maze.move_robot(Action.Left)
        elif symbol == key.MOTION_DOWN:
            self.maze.move_robot(Action.Down)

    def render(self):
        self.renderer.render(MazeDrawer.draw_maze(self.maze, self.height, self.width))

    def run(self):
        while self.alive:
            self.render()
            event = self.dispatch_events()
            # Check exit condition
            if self.maze.is_robot_reach_goal():
                self.alive = False

        self.close()