from gym.envs.registration import register

register(
    id='ImageMaze-v0',
    entry_point='gym_image_maze.envs:ImageMazeV0',
)

register(
    id='ImageMaze-v1',
    entry_point='gym_image_maze.envs:ImageMazeV1',
)

register(
    id='ImageMaze-v2',
    entry_point='gym_image_maze.envs:ImageMazeV2',
)