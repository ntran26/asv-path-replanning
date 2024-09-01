import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

#                               -------- CONFIGURATION --------
# Define colors
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (1, 0, 0)
GREEN = (0, 1, 0)
YELLOW = (1, 1, 0)
BLUE = (0, 0, 1)

# Define map dimensions and start/goal points, number of static obstacles
WIDTH = 100
HEIGHT = 100
START = (10, 10)
GOAL = (90, 90)
NUM_STATIC_OBS = 3

# Define observation radius and grid size
RADIUS = 40
SQUARE_SIZE = 5

# Define initial heading angle, turn rate and number of steps
INITIAL_HEADING = 90
TURN_RATE = 5
SPEED = 1

# Define states
FREE_STATE = 0         # free space
COLLISION_STATE = 1    # obstacle or border
GOAL_STATE = 2         # goal point

# Define maximum steps
MAX_NUM_STEP = 150

class ASVEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, render_mode = "human"):
        super(ASVEnv, self).__init__()
        self.render_mode = render_mode

        self.width = WIDTH
        self.height = HEIGHT
        self.heading = INITIAL_HEADING
        self.turn_rate = TURN_RATE
        self.speed = SPEED
        self.start = START
        self.goal = GOAL
        self.radius = RADIUS
        self.grid_size = SQUARE_SIZE
        self.center_point = (0,0)
        self.max_num_step = MAX_NUM_STEP

        # Define action space and observation space
        # 3 possible actions: left, right, straight
        ### OR
        # 5 possible actions: left, right, straight, accelerate, decelerate
        self.action_space = spaces.Discrete(3)

        # 4 possible states for observation, and the shape = number of grids inside the observation radius
        self.observation_space = spaces.Box(low=0, high=2, shape=(193,), dtype=np.int32) 
        
        self.reset()
    
    #                           -------- HELPER FUNCTIONS --------
    
    # Create a function that sets the priority of each state in case they overlap: 
    # obstacle/collision > goal point > path > free space
    def get_priority_state(self, current_state, new_state):
        if new_state == COLLISION_STATE:
            return COLLISION_STATE
        elif new_state == GOAL_STATE and current_state != COLLISION_STATE:
            return GOAL_STATE
        elif current_state not in (COLLISION_STATE, GOAL_STATE):
            return FREE_STATE
        return current_state
    
    # Create a function that generate grid coordinates (x,y) from global map
    def generate_grid(self, radius, square_size, center):
        x = np.arange(-radius + square_size, radius, square_size)
        y = np.arange(-radius + square_size, radius, square_size)
        grid = []
        for i in x:
            for j in y:
                if np.sqrt(i ** 2 + j ** 2) <= radius:
                    grid.append((center[0] + i, center[1] + j))
        return grid
    
    # Create a function that converts each point from the global map to a grid coordinate
    def closest_multiple(self, n, mult):
        return int((n + mult / 2) // mult) * mult
    
    # Create a function to generate a dictionary, storing the grid coordinates and state
    def fill_grid(self, objects, grid_size, center):
        grid_dict = {}
        for obj in objects:
            m = obj['x']
            n = obj['y']
            state = obj['state']

            # Calculate the distance from the ASV's current position (center) to the object
            distance_to_asv = np.sqrt((m - center[0]) ** 2 + (n - center[1]) ** 2)

            # Check if the object is within the observation radius or if it's the goal (always include goal)
            if distance_to_asv <= self.radius or state == GOAL_STATE:
                m = self.closest_multiple(m, grid_size)
                n = self.closest_multiple(n, grid_size)

            if (m, n) not in grid_dict:
                grid_dict[(m, n)] = FREE_STATE
            
            grid_dict[(m, n)] = self.get_priority_state(grid_dict[(m,n)], state)
        return grid_dict
    
    def check_virtual_goal(self, position):
        position = position
        goal = self.goal
        distance_to_goal = np.sqrt((position[0] - goal[0]) ** 2 + (position[1] - goal[1]) ** 2)
        if distance_to_goal > self.radius:
            dir_x = (goal[0] - position[0])/distance_to_goal
            dir_y = (goal[1] - position[1])/distance_to_goal
            if goal[0] > position[0] and goal[1] > position[1]:
                virtual_goal_x = position[0] + 20 * dir_x
                virtual_goal_y = position[1] + 20 * dir_y
            virtual_goal = (virtual_goal_x, virtual_goal_y)
        else:
            virtual_goal = goal
        return virtual_goal
    
    #                           -------- MAP GENERATION --------

    # Create a function to generate borders around the map
    def generate_border(self, map_width, map_height):
        boundary = []
        for x in range(0, map_width + 1):
            boundary.append({'x': x, 'y': 0, 'state': COLLISION_STATE})             # lower boundary
            boundary.append({'x': x, 'y': map_height, 'state': COLLISION_STATE})    # upper boundary
        for y in range(0, map_height + 1):
            boundary.append({'x': 0, 'y': y, 'state': COLLISION_STATE})             # left boundary
            boundary.append({'x': map_width, 'y': y, 'state': COLLISION_STATE})     # right boundary
        return boundary
    
    # Create a function to generate static obstacles
    def generate_static_obstacles(self, num_obs):
        obstacles = []
        # Generate random obstacles around the map
        for _ in range(num_obs):
            x = np.random.randint(self.start[0] + 10, self.goal[0] - 10)
            y = np.random.randint(self.start[1] + 10, self.goal[1] - 10)
            obstacles.append({'x': x, 'y': y, 'state': COLLISION_STATE})
        return obstacles
    
    def generate_goal(self, goal_point):
        goal = []
        goal.append({'x': goal_point[0], 'y': goal_point[1], 'state': GOAL_STATE})
        return goal
    
    def generate_virtual_goal(self, virtual_goal):
        goal = []
        goal.append({'x': virtual_goal[0], 'y': virtual_goal[1], 'state': GOAL_STATE})
        return goal

    #                           -------- MAIN FUNCTIONS --------

    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)
        self.step_count = 0
        self.step_taken = []
        self.heading_taken = []
        self.current_heading = self.heading
        self.current_speed = self.speed
        self.position = self.start
        self.virtual_goal = self.check_virtual_goal(self.position)

        self.boundary = self.generate_border(self.width, self.height)
        self.goal_point = self.generate_goal(self.goal)
        self.virtual_goal = self.generate_virtual_goal(self.virtual_goal)
        self.obstacles = self.generate_static_obstacles(NUM_STATIC_OBS)
        self.objects_environment = self.obstacles+ self.boundary + self.goal_point
        self.grid_dict = self.fill_grid(self.objects_environment, self.grid_size, self.position)
        self.grid = self.generate_grid(self.radius, self.grid_size, self.position)
        
        self.done = False
        return self.get_observation(), {}
    
    def check_done(self, position):        
        x, y = position
        state = self.grid_dict.get((self.closest_multiple(x, self.grid_size), self.closest_multiple(y, self.grid_size)), FREE_STATE)
        
        # If the agent collide with obstacles or boundary
        if state == COLLISION_STATE:
            return True
        # If the agent reached goal
        elif state == GOAL_STATE:
            return True
        # If the total number of steps are 250 or above
        elif self.step_count >= self.max_num_step:
            return True
        return False
    
    def get_observation(self):
        current_pos = self.position
        new_grid = self.generate_grid(self.radius, self.grid_size, current_pos)
        observation = np.zeros(len(new_grid), dtype = np.int32)
        for idx, (x, y) in enumerate(new_grid):
            state = self.grid_dict.get((self.closest_multiple(x, self.grid_size), self.closest_multiple(y, self.grid_size)), FREE_STATE)
            observation[idx] = state
        return observation
    
    def step(self, action):
        if action == 0:     # go straight
            self.current_heading = self.current_heading
            self.position = (self.position[0] + self.speed * np.cos(np.radians(self.current_heading)),
                        self.position[1] + self.speed * np.sin(np.radians(self.current_heading)))
        elif action == 1:   # turn left
            self.current_heading += self.turn_rate
            self.position = (self.position[0] + self.speed * np.cos(np.radians(self.current_heading)),
                        self.position[1] + self.speed * np.sin(np.radians(self.current_heading)))
        elif action == 2:   # turn right
            self.current_heading -= self.turn_rate
            self.position = (self.position[0] + self.speed * np.cos(np.radians(self.current_heading)),
                        self.position[1] + self.speed * np.sin(np.radians(self.current_heading)))
        # elif action == 3:   # accelerate
        #     self.current_speed += 0.5
        #     if self.current_speed > 2:
        #         self.current_speed = 2
        #     self.position = (self.position[0] + self.speed * np.cos(np.radians(self.current_heading)),
        #                     self.position[1] + self.speed * np.sin(np.radians(self.current_heading)))
        # elif action == 4:   # decelerate
        #     self.current_speed -= 0.5
        #     if self.current_speed < 1:
        #         self.current_speed = 1
        #     self.position = (self.position[0] + self.speed * np.cos(np.radians(self.current_heading)),
        #                     self.position[1] + self.speed * np.sin(np.radians(self.current_heading)))

        self.virtual_goal = self.check_virtual_goal(self.position)
        virtual_goal_point = self.generate_virtual_goal(self.virtual_goal)

        # Update grid_dict with the current ASV position
        self.objects_environment = self.obstacles + self.boundary + self.goal_point + virtual_goal_point
        self.grid_dict = self.fill_grid(self.objects_environment, self.grid_size, self.position)

        self.step_count += 1
        self.step_taken.append((self.position[0], self.position[1]))
        self.heading_taken.append(self.current_heading)
        reward = self.calculate_reward(self.position)
        terminated = self.check_done(self.position)
        observation = self.get_observation()
        
        return observation, reward, terminated, False, {}
    
    def calculate_reward(self, position):
        x, y = position
        state = self.grid_dict.get((self.closest_multiple(x, self.grid_size), self.closest_multiple(y, self.grid_size)), FREE_STATE)

        reward = 0
        if state == COLLISION_STATE:
            reward -= 5
        elif state == GOAL_STATE:
            reward += 10
        elif state == FREE_STATE:
            reward = 0
        
        return reward

    def render(self, mode="human"):
        if mode == 'human':
            if not hasattr(self, 'fig'):
                self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 8))

                self.ax1.set_aspect('equal')
                self.ax1.set_title('MAP')
                self.ax1.set_xlim(-self.radius, self.width + self.radius)
                self.ax1.set_ylim(-self.radius, self.height + self.radius)

                self.ax2.set_aspect('equal')
                self.ax2.set_title('OBSERVATION')
                self.ax2.set_xlim(-self.radius, self.radius)
                self.ax2.set_ylim(-self.radius, self.radius)

                self.agent_1, = self.ax1.plot([], [], marker='^', color=BLUE)
                self.agent_2, = self.ax2.plot([], [], marker='^', color=BLUE)
                self.observation_horizon1 = plt.Circle(self.start, self.radius, color=BLUE, fill=False)
                self.observation_horizon2 = plt.Circle((0, 0), self.radius, color=BLUE, fill=False)
                self.virtual_goal_point, = self.ax1.plot([], [], marker='o', color=GREEN)

                self.ax1.add_patch(self.observation_horizon1)
                self.ax2.add_patch(self.observation_horizon2)

                self.ax1.plot(self.start[0], self.start[1], marker='o', color=BLUE)
                self.ax1.plot(self.goal[0], self.goal[1], marker='o', color=GREEN)

                for obj in self.boundary:
                    boundary_line = plt.Rectangle((obj['x'], obj['y']), 1, 1, edgecolor=BLACK, facecolor=BLACK)
                    self.ax1.add_patch(boundary_line)

                for obj in self.obstacles:
                    self.ax1.plot(obj['x'], obj['y'], marker='o', color=RED)

            self.agent_1.set_data(self.position[0], self.position[1])
            self.agent_1.set_marker((3, 0, self.current_heading - 90))
            self.virtual_goal_point.set_data(self.virtual_goal[0], self.virtual_goal[1])
            self.virtual_goal_point.set_marker("o")

            self.observation_horizon1.center = self.position

            new_grid = self.generate_grid(self.radius, self.grid_size, self.position)
            for rect in getattr(self, 'grid_patches', []):
                rect.remove()
            self.grid_patches = []

            for (cx, cy) in new_grid:
                state = self.grid_dict.get((self.closest_multiple(cx, self.grid_size), self.closest_multiple(cy, self.grid_size)), FREE_STATE)
                color = WHITE
                if state == COLLISION_STATE:
                    color = RED
                elif state == GOAL_STATE:
                    color = GREEN
                rect = plt.Rectangle((cx - self.grid_size / 2 - self.position[0], cy - self.grid_size / 2 - self.position[1]), self.grid_size, self.grid_size,
                                    edgecolor='gray', facecolor=color)
                rect.set_zorder(1)
                self.grid_patches.append(rect)
                self.ax2.add_patch(rect)

            self.agent_2.set_data(0, 0)
            self.agent_2.set_marker((3, 0, self.current_heading - 90))
            self.agent_2.set_zorder(3)

            self.observation_horizon2.center = (0, 0)
            self.observation_horizon2.set_zorder(2)

            plt.draw()
            plt.pause(0.01)
    
    #                           -------- POST PROCESSING --------
    
    def display_path(self):
        # Plot the path taken
        fig, ax = plt.subplots(1,1, figsize=(8,8))
        ax.set_aspect("equal")
        ax.set_title("Steps Taken")
        ax.set_xlim(-self.radius, self.width + self.radius)
        ax.set_ylim(-self.radius, self.height + self.radius)
        ax.plot(self.start[0], self.start[1], marker='o', color=BLUE)
        ax.plot(self.goal[0], self.goal[1], marker='o', color=GREEN)
        for obj in self.boundary:
            boundary_line = plt.Rectangle((obj['x'], obj['y']), 1, 1, edgecolor=BLACK, facecolor=BLACK)
            ax.add_patch(boundary_line)
        for obj in self.obstacles:
            ax.plot(obj['x'], obj['y'], marker='o', color=RED)
        ax.plot(self.position[0], self.position[1], marker='^', color=BLUE)
        step_x = [point[0] for point in self.step_taken]
        step_y = [point[1] for point in self.step_taken]
        ax.plot(step_x, step_y, marker='.', color=BLUE)
        plt.show()

    def env_visualisation(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

        # First plot: whole map
        ax1.set_aspect('equal')
        ax1.set_title('MAP')
        # Second plot: observation per timestep
        ax2.set_aspect('equal')
        ax2.set_title('OBSERVATION')
        ax2.set_xlim(-self.radius, self.radius)
        ax2.set_ylim(-self.radius, self.radius)

        # Plot ASV and observation circle
        self.agent_1, = ax1.plot([], [], marker='^', color=BLUE)
        self.agent_2, = ax2.plot([], [], marker='^', color=BLUE)
        observation_horizon1 = plt.Circle(self.start, self.radius, color=BLUE, fill=False)
        observation_horizon2 = plt.Circle((0, 0), self.radius, color=BLUE, fill=False)
        ax1.add_patch(observation_horizon1)
        ax2.add_patch(observation_horizon2)

        # Plot start point
        ax1.plot(self.start[0], self.start[1], marker='o', color=BLUE)

        # Plot goal point
        ax1.plot(self.goal[0], self.goal[1], marker='o', color=GREEN)

        # Plot the boundary
        for obj in self.boundary:
            boundary_line = plt.Rectangle((obj['x'], obj['y']), 1, 1, edgecolor=BLACK, facecolor=BLACK)
            ax1.add_patch(boundary_line)

        # Plot obstacles
        for obj in self.obstacles:
            ax1.plot(obj['x'], obj['y'], marker='o', color=RED)

        # Plot the grid points on the right
        self.grid = self.generate_grid(self.radius, self.grid_size, self.center_point)
        grid_patches = []

        # Initialize all grids as free space
        for (cx, cy) in self.grid:
            state = self.grid_dict.get((self.closest_multiple(cx, self.grid_size), self.closest_multiple(cy, self.grid_size)), FREE_STATE)
            color = WHITE
            rect = plt.Rectangle((cx - self.grid_size / 2 - self.center_point[0], cy - self.grid_size / 2 - self.center_point[1]), self.grid_size, self.grid_size,
                                edgecolor='gray', facecolor=color)
            grid_patches.append(rect)
            ax2.add_patch(rect)
        
        # print(len(self.step_taken))
        # print(len(self.heading_taken))

        self.asv_path = self.step_taken
        self.asv_heading = self.heading_taken

        def update(frame):
            current_pos = self.asv_path[frame]
            heading = self.asv_heading[frame]

            # Update ASV position and heading angle in first plot
            self.agent_1.set_data(current_pos[0], current_pos[1])
            self.agent_1.set_marker((3, 0, heading - 90))

            # Update the observation circle: move along with the ASV in the first plot
            observation_horizon1.center = current_pos

            # Update grid_dict with the new center point
            new_grid = self.generate_grid(self.radius, self.grid_size, current_pos)
            for rect in grid_patches:
                rect.remove()
            grid_patches.clear()

            for (cx, cy) in new_grid:
                state = self.grid_dict.get((self.closest_multiple(cx, self.grid_size), self.closest_multiple(cy, self.grid_size)), FREE_STATE)
                color = WHITE
                if state == COLLISION_STATE:
                    color = RED
                elif state == GOAL_STATE:
                    color = GREEN
                rect = plt.Rectangle((cx - self.grid_size / 2 - current_pos[0], cy - self.grid_size / 2 - current_pos[1]), self.grid_size, self.grid_size,
                                     edgecolor='gray', facecolor=color)
                rect.set_zorder(1)     # make sure the grids don't overlay other components
                grid_patches.append(rect)
                ax2.add_patch(rect)
            
            # Update ASV and observation circle after grid patches in the second plot
            self.agent_2.set_data(0, 0)
            self.agent_2.set_marker((3, 0, heading - 90))
            self.agent_2.set_zorder(3)

            observation_horizon2.center = (0, 0)
            observation_horizon2.set_zorder(2)

            return self.agent_1, self.agent_2, observation_horizon1, observation_horizon2, *grid_patches

        # Create animation and display
        ani = FuncAnimation(fig, update, frames=len(self.asv_path), blit=True, interval=200, repeat=False)
        
        # # Write to mp4 file
        # FFwriter = FFMpegWriter(fps=5)
        # ani.save("ppo_static_obs.mp4", writer=FFwriter)
        # plt.show()

# Test the environment with random actions
if __name__ == '__main__':
    env = ASVEnv()
    obs = env.reset()

    # print("Observation Space Shape", env.observation_space.shape)
    # print("Sample observation", len(env.get_observation()))

    # print("Action Space Shape", env.action_space.n)
    # print("Action Space Sample", env.action_space.sample())

    for _ in range(env.max_num_step):  # Run for 100 steps or until done
        action = env.action_space.sample()  # Take a random action
        # print(env.get_observation())
        # print(len(env.get_observation()))
        # print(env.calculate_distance_to_goal(env.position))
        # print(env.calculate_distance_to_path(env.position))
        # print(env.calculate_reward(env.position))
        # print(len(env.grid_dict))

        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            break
    env.display_path()
    env.close()
