# GRID WORLD ENVIRONMENT
# Text-based environment
# The agent starts from top left to navigate down to bottom right
# 2 magic squares that cause the agent to teleport across the board
# Purpose of magic squares: make agent learn the shortcut
# Reward function: -1 for each step, 0 for reaching destination
# State space: all states - terminal state
# State space plus: all states

import numpy as np
import matplotlib.pyplot as plt

class GridWorld(object):
    # Initialize
    def __init__(self, m, n, magicSquares):
        self.grid = np.zeros((m,n))
        self.m = m
        self.n = n
        self.stateSpace = [i for i in range(self.m*self.n)]
        self.stateSpace.remove(self.m*self.n-1)
        self.stateSpacePlus = [i for i in range(self.m*self.n)]
        self.actionSpace = {'U': -self.m, 'D': self.m,
                            'L': -1, 'R': 1}
        self.possibleActions = ['U', 'D', 'L', 'R']
        # dict with magic squares and resulting squares
        self.addMagicSquares(magicSquares)
        self.agentPosition = 0

    # Check terminal state function
    def isTerminalState(self, state):
        return state in self.stateSpacePlus and state not in self.stateSpace

    # Add magic squares function
    def addMagicSquares(self, magicSquares):
        self.magicSquares = magicSquares
        i = 2
        for square in self.magicSquares:
            x = square // self.m
            y = square % self.n
            self.grid[x][y] = i
            i += 1
            x = magicSquares[square] // self.m
            y = magicSquares[square] % self.n
            self.grid[x][y] = i
            i += 1

    # Get agent location function
    def getAgentRowAndColumn(self):
        x = self.agentPosition // self.m
        y = self.agentPosition % self.n
        return x, y

    # Update state function
    def setState(self, state):
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 0
        self.agentPosition = state
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 1

    def offGridMove(self, newState, oldState):
        # if we move into a row not in the grid
        if newState not in self.stateSpacePlus:
            return True
        # if we're trying to wrap around to next row
        elif oldState % self.m == 0 and newState  % self.m == self.m - 1:
            return True
        elif oldState % self.m == self.m - 1 and newState % self.m == 0:
            return True
        else:
            return False

    # Action function
    def step(self, action):
        agentX, agentY = self.getAgentRowAndColumn()
        resultingState = self.agentPosition + self.actionSpace[action]
        # Check if the resulting state is in magic squares => agent move to new position
        if resultingState in self.magicSquares.keys():
            resultingState = magicSquares[resultingState]

        # Design reward function
        # if the agent is not at terminal state or goal => reward = -1, otherwise reward = 0
        reward = -1 if not self.isTerminalState(resultingState) else 0

        if not self.offGridMove(resultingState, self.agentPosition):
            self.setState(resultingState)
            return resultingState, reward, self.isTerminalState(resultingState), None
        else:
            return self.agentPosition, reward, self.isTerminalState(self.agentPosition), None

    # Reset function
    def reset(self):
        self.agentPosition = 0
        self.grid = np.zeros((self.m,self.n))
        self.addMagicSquares(self.magicSquares)
        return self.agentPosition

    # Render the map
    def render(self):
        print('------------------------------------------')
        for row in self.grid:
            for col in row:
                if col == 0:
                    print('-', end='\t')
                elif col == 1:
                    print('X', end='\t')
                elif col == 2:
                    print('Ain', end='\t')
                elif col == 3:
                    print('Aout', end='\t')
                elif col == 4:
                    print('Bin', end='\t')
                elif col == 5:
                    print('Bout', end='\t')
            print('\n')
        print('------------------------------------------')

    # Function to take a random action
    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)

# Find the maximum action for a given state
def maxAction(Q, state, actions):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)
    return actions[action]

if __name__ == '__main__':
    # Set up the map
    magicSquares = {18: 54, 63: 14}
    env = GridWorld(9, 9, magicSquares)

    # Set up hyperparameters
    ALPHA = 0.1     # learning rate
    GAMMA = 1.0     # count all future rewards equally
    EPS = 1.0       # epsilon greedy action selection (EPS = 1 -> exploration)

    # Initialize Q-Learning table
    Q = {}
    for state in env.stateSpacePlus:
        for action in env.possibleActions:
            Q[state, action] = 0

    numGames = 50000
    totalRewards = np.zeros(numGames)

    # Display environment
    env.render()

    for i in range(numGames):
        if i % 5000 == 0:
            print('starting game ', i)

        done = False
        epRewards = 0               # reset episode rewards
        observation = env.reset()   # reset environment

        # Choose action and calculate reward
        while not done:
            rand = np.random.random()
            if rand < (1-EPS):
                action = maxAction(Q, observation, env.possibleActions)
            else:
                action = env.actionSpaceSample()
            observation_, reward, done, info = env.step(action)
            epRewards += reward

            # Update Q-table
            action_ = maxAction(Q, observation_, env.possibleActions)
            Q[observation,action] = Q[observation,action] + ALPHA*(reward + \
                        GAMMA*Q[observation_,action_] - Q[observation,action])
            observation = observation_

        # Decrease epsilon at the end of each episode -> greedy strategy
        if EPS - 2 / numGames > 0:
            EPS -= 2 / numGames
        else:
            EPS = 0
        totalRewards[i] = epRewards

    plt.plot(totalRewards)
    plt.show()