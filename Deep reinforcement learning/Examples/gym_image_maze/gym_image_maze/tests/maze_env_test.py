import unittest
import os
from envs.image_maze_env import ImageMazeEnv
from envs.models import *

class MazeEnvTest(unittest.TestCase):
    def test_maze_env_create_maze(self):
        env = ImageMazeEnv(os.path.join(os.getcwd(), 'gym_image_maze/tests/maze_config.json'), 100)
        maze = env.maze
        self.assertEqual(maze.size, 36)
        self.assertEqual(maze.robot.size, 2)
        self.assertEqual(maze.robot.position, (5, 30))
        self.assertEqual(maze.goal, (5,5))
        self.assertEqual(len(maze.walls), 7) # Walls and 4 borders
        self.assertEqual(maze.walls[4].a, (0, 10))
        self.assertEqual(maze.walls[4].b, (8, 10))
        self.assertEqual(maze.walls[5].a, (20, 25))
        self.assertEqual(maze.walls[5].b, (20, 35))
        self.assertEqual(maze.walls[6].a, (15, 25))
        self.assertEqual(maze.walls[6].b, (25, 25))
        
    def test_maze_env_observation(self):
        env = ImageMazeEnv(os.path.join(os.getcwd(), 'gym_image_maze/tests/blank_maze_config.json'), 100)
        maze_size = env.maze.size
        initial_state = env.reset()
        self.assertEqual(initial_state.shape, (maze_size, maze_size))
        state, _, _, _ = env.step(env.action_space.sample())
        self.assertEqual(state.shape, (maze_size, maze_size))
        
    def test_maze_env_done(self):
        env = ImageMazeEnv(os.path.join(os.getcwd(), 'gym_image_maze/tests/blank_maze_config.json'), 100)
        # 0: Left, 1: Right, 2: Up, 3: Down, 4: Stop
        env.reset() 
        env.step(0)
        env.step(0)
        _, _, done, _ = env.step(0)
        self.assertFalse(done)
        env.step(2)
        _, _, done, _ = env.step(2)
        self.assertFalse(done)
        _, _, done, _ = env.step(2)
        self.assertTrue(env.done)
        
    def test_maze_env_collide_done(self):
        env = ImageMazeEnv(os.path.join(os.getcwd(), 'gym_image_maze/tests/blank_maze_config.json'), 100)
        # 0: Left, 1: Right, 2: Up, 3: Down, 4: Stop
        env.reset() 
        _, reward, done, _ = env.step(1)
        self.assertEqual(reward, Reward.Collide.value)
        self.assertTrue(done)
        env.reset() 
        _, _, done, _ = env.step(0)
        self.assertFalse(done)
        _, _, done, _ = env.step(0)
        self.assertFalse(done)
        _, reward, done, _ = env.step(3)
        self.assertEqual(reward, Reward.Collide.value)
        self.assertTrue(done)
        
    def test_maze_env_time_limit(self):
        env = ImageMazeEnv(os.path.join(os.getcwd(), 'gym_image_maze/tests/blank_maze_config.json'), 5)
        # 0: Left, 1: Right, 2: Up, 3: Down, 4: Stop
        env.reset()
        env.step(0)
        env.step(1)
        env.step(2)
        env.step(3)
        _, reward, done, _ = env.step(3)
        self.assertTrue(done)
        self.assertNotEqual(env.maze.robot.position, env.maze.goal)
        self.assertNotEqual(reward, Reward.Goal.value)
        
    def test_maze_env_reset(self):
        env = ImageMazeEnv(os.path.join(os.getcwd(), 'gym_image_maze/tests/blank_maze_config.json'), 5)
        # 0: Left, 1: Right, 2: Up, 3: Down, 4: Stop
        env.reset()
        env.step(0)
        env.step(0)
        env.step(2)
        self.assertEqual(env.maze.robot.position, (4,5))
        self.assertEqual(env.timestep, 3)
        env.reset()
        self.assertEqual(env.timestep, 0)
        self.assertEqual(env.maze.robot.position, (6, 6))
        
    def test_maze_env_step_reward(self):
        env = ImageMazeEnv(os.path.join(os.getcwd(), 'gym_image_maze/tests/blank_maze_config.json'), 100)
        env.reset()
        # 0: Left, 1: Right, 2: Up, 3: Down, 4: Stop
        _, reward, _, _ = env.step(0)
        self.assertEqual(reward, Reward.Closer.value)
        _, reward, _, _ = env.step(2)
        self.assertEqual(reward, Reward.Closer.value)
        _, reward, _, _ = env.step(4)
        self.assertEqual(reward, Reward.Same.value)
        _, reward, _, _ = env.step(3)
        self.assertEqual(reward, Reward.Further.value)
        _, reward, _, _ = env.step(1)
        self.assertEqual(reward, Reward.Further.value)
        _, reward, _, _ = env.step(3)
        self.assertEqual(reward, Reward.Collide.value)
        _, reward, _, _ = env.step(1)
        self.assertEqual(reward, Reward.Collide.value)
        
    def test_maze_env_total_reward(self):
        env = ImageMazeEnv(os.path.join(os.getcwd(), 'gym_image_maze/tests/blank_maze_config.json'), 10)
        env.reset()
        total_reward = 0
        # 0: Left, 1: Right, 2: Up, 3: Down, 4: Stop
        _, reward, _, _ = env.step(0)
        total_reward += reward
        _, reward, _, _ = env.step(0)
        total_reward += reward
        _, reward, _, _ = env.step(0)
        total_reward += reward
        _, reward, _, _ = env.step(2)
        total_reward += reward
        _, reward, _, _ = env.step(2)
        total_reward += reward
        _, reward, _, _ = env.step(2)
        total_reward += reward
        self.assertEqual(total_reward, 1005)
        
        env.reset()
        total_reward = 0
        _, reward, _, _ = env.step(0)
        total_reward += reward
        _, reward, _, _ = env.step(2)
        total_reward += reward
        _, reward, _, _ = env.step(1)
        total_reward += reward
        _, reward, _, _ = env.step(1)
        total_reward += reward
        self.assertEqual(total_reward, -9)
        
        env.reset()
        total_reward = 0
        _, reward, _, _ = env.step(0)
        total_reward += reward
        _, reward, _, _ = env.step(0)
        total_reward += reward
        _, reward, _, _ = env.step(2)
        total_reward += reward
        _, reward, _, _ = env.step(2)
        total_reward += reward
        _, reward, _, _ = env.step(4)
        total_reward += reward
        _, reward, _, _ = env.step(4)
        total_reward += reward
        _, reward, _, _ = env.step(4)
        total_reward += reward
        _, reward, _, _ = env.step(4)
        total_reward += reward
        _, reward, _, _ = env.step(4)
        total_reward += reward
        _, reward, _, _ = env.step(4)
        total_reward += reward
        self.assertEqual(total_reward, 4)