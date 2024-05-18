import unittest
from envs.maze import Maze
from envs.models import Wall, Action
from envs.utilities import dist_from_point_to_line

class MazeTest(unittest.TestCase):
    def test_maze_init(self):
        maze = Maze(size=84, robot_pos=(10, 70), robot_size=2, goal_pos=(10,10))
        self.assertEqual(maze.size, 84)
        self.assertEqual(maze.robot.position, (10, 70))
        self.assertEqual(maze.robot.size, 2)
        self.assertEqual(maze.goal, (10, 10))
        
    def test_maze_init_add_borders(self):
        maze = Maze(size=10, robot_pos=(3, 5), robot_size=2, goal_pos=(2,2))
        self.assertEqual(len(maze.walls), 4)
        self.assertEqual(maze.walls[0].a, (0,0))
        self.assertEqual(maze.walls[0].b, (9,0))
        self.assertEqual(maze.walls[1].a, (0,0))
        self.assertEqual(maze.walls[1].b, (0,9))
        self.assertEqual(maze.walls[2].a, (0,9))
        self.assertEqual(maze.walls[2].b, (9,9))
        self.assertEqual(maze.walls[3].a, (9,9))
        self.assertEqual(maze.walls[3].b, (9,0))
        
    def test_maze_init_add_walls(self):
        walls = [
            Wall((0,4), (3,4)), 
            Wall((7,0), (7,3))
        ]
        maze = Maze(size=10, robot_pos=(3, 7), robot_size=2, goal_pos=(2,2), walls=walls)
        self.assertEqual(len(maze.walls), 6)
        self.assertEqual(maze.walls[4].a, (0,4))
        self.assertEqual(maze.walls[4].b, (3,4))
        self.assertEqual(maze.walls[5].a, (7,0))
        self.assertEqual(maze.walls[5].b, (7,3))
        
    def test_maze_move_robot_no_collision(self):
        maze = Maze(size=11, robot_pos=(3, 7), robot_size=2, goal_pos=(2,2))
        maze.move_robot(Action.Up)
        self.assertEqual(maze.robot.position, (3,6))
        maze.move_robot(Action.Right)
        self.assertEqual(maze.robot.position, (4,6))
        maze.move_robot(Action.Down)
        self.assertEqual(maze.robot.position, (4,7))
        maze.move_robot(Action.Left)
        self.assertEqual(maze.robot.position, (3,7))
        
    def test_maze_move_robot_prevent_border_collision(self):
        maze = Maze(size=10, robot_pos=(3, 6), robot_size=2, goal_pos=(2,2))
        maze.move_robot(Action.Down)
        self.assertEqual(maze.robot.position, (3,6))
        maze.move_robot(Action.Left)
        self.assertEqual(maze.robot.position, (3,6))
        maze.move_robot(Action.Right)
        maze.move_robot(Action.Right)
        maze.move_robot(Action.Right)
        self.assertEqual(maze.robot.position, (6,6))
        maze.move_robot(Action.Right)
        self.assertEqual(maze.robot.position, (6,6))
        maze.move_robot(Action.Up)
        maze.move_robot(Action.Up)
        maze.move_robot(Action.Up)
        self.assertEqual(maze.robot.position, (6,3))
        maze.move_robot(Action.Up)
        self.assertEqual(maze.robot.position, (6,3))
        
    def test_maze_move_robot_prevent_wall_collision(self):
        walls = [
            Wall((3,1), (6,1)), 
            Wall((1,4), (1,6)),
            Wall((4,8), (7,8)),
            Wall((8,5), (8,8))
        ]
        maze = Maze(size=10, robot_pos=(5, 5), robot_size=2, goal_pos=(2,2), walls=walls)
        maze.move_robot(Action.Down)
        self.assertEqual(maze.robot.position, (5,5))
        maze.move_robot(Action.Right)
        self.assertEqual(maze.robot.position, (5,5))
        maze.move_robot(Action.Up)
        self.assertEqual(maze.robot.position, (5,4))
        self.assertTrue(maze.collide((5,3)))
        maze.move_robot(Action.Up)
        self.assertEqual(maze.robot.position, (5,4))
        maze.move_robot(Action.Left)
        self.assertEqual(maze.robot.position, (4,4))
        self.assertTrue(maze.collide((3,4)))
        maze.move_robot(Action.Left)
        self.assertEqual(maze.robot.position, (4,4))
        
    def test_maze_to_image(self):
        walls = [
            Wall((0,5), (3,5)), 
            Wall((8,9), (8,7)),
            Wall((7,1), (8,3))
        ]
        maze = Maze(size=10, robot_pos=(3, 7), robot_size=2, goal_pos=(2,2), walls=walls)
        img = maze.to_image()
        self.assertEqual(img.shape, (10, 10))
        # Position = (y,x)
        # Draw robot
        self.assertEqual(img[7][3], 0)
        self.assertEqual(img[5][3], 0)
        self.assertEqual(img[9][3], 0)
        self.assertEqual(img[7][1], 0)
        self.assertEqual(img[7][5], 0)
        # Draw goal
        self.assertEqual(img[2][2], 0)
        # Draw walls
        self.assertEqual(img[5][0], 0)
        self.assertEqual(img[5][3], 0)
        self.assertEqual(img[9][8], 0)
        self.assertEqual(img[7][8], 0)
        self.assertEqual(img[1][7], 0)
        self.assertEqual(img[3][8], 0)
        # Draw borders
        self.assertEqual(img[0][0], 0)
        self.assertEqual(img[0][9], 0)
        self.assertEqual(img[9][0], 0)
        self.assertEqual(img[9][9], 0)
        # Resize
        img_resize = maze.to_image(30)
        self.assertEqual(img_resize.shape, (30, 30))
        
        