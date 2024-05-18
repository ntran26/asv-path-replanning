import unittest
from envs.models import Robot, Action

class ModelTest(unittest.TestCase):
    def test_robot_move(self):
        robot = Robot(position=(5,5), size=2)
        robot.move(Action.Up)
        self.assertEqual(robot.position, (5, 4))
        robot.move(Action.Right)
        self.assertEqual(robot.position, (6, 4))
        robot.move(Action.Down)
        self.assertEqual(robot.position, (6, 5))
        robot.move(Action.Left)
        self.assertEqual(robot.position, (5, 5))