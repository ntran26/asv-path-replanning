# Deceptive Maze

_Deceptive Maze_ is a reinforcement learning environment developed using PyGLET and OpenAI Gym. In this environment, the agent is placed into a maze with a mission to discover the treasure(denoted as a cross in the image). The agent can move in 4 directions each turn: Up, Down, Left, and Right. A reward of +1  is given every time the it gets closer to the goal and -1 otherwise. Also, the agent will be penalized by -10 if hit the wall. The episodes end when the agent either find the treasure or fail to do it within 200 turns. The controller has to decide which move is right at a particular state from the image of the maze.

Full detail about this project can be found in [this article](https://medium.com/geekculture/developing-reinforcement-learning-environment-using-openai-gym-f510b0393eb7).

## Installation

```
python setup.py develop
```

## Using the environment

```
import gym

if __name__=='__main__':
   env = gym.make('ImageMaze-v0')
   env.reset()
   for i in range(500):
       env.render()
       observation, reward, done, _ = env.step(env.action_space.sample())
       print('Observation : ' + str(observation.shape))
       print('Reward      : ' + str(reward))
       print('Done        : ' + str(done))
       print('---------------------')
  env.close()
```

