import json

"""
Test case is None: random start & goal points, random obstacles

Test case 0: no obstacles
Test case 1: start & goal at center, no obstacles (path following)
Test case 2: start & goal at center, single obstacle on path
Test case 3: start & goal at center, single obstacle to the left of path
Test case 4: start & goal at center, single obstacle to the right of path
Test case 5: start & goal at center, 3 obstacles scatter along the path
Test case 6: start & goal at each corner, 4 obstacles cover the path, leaving a single blank space

Test case 99: take the setup from recorded data of a random obstacles scenario (test case 0)
"""

ENV_DATA = "data/env_setup/survival_pool/env_0.json"

OBS_LENGTH = 1

class TestCase:
    def __init__(self):
        self.obs = []
        self.start_x = None
        self.start_y = None
        self.goal_x = None
        self.goal_y = None
        self.obs_size = OBS_LENGTH/2
        self.env_data = ENV_DATA

    def obstacles(self, test_case):
        if test_case == 0:
            self.obs = []
        elif test_case == 1:    # middle
            x = 5
            y = 12.5
            self.obs.append([(x-self.obs_size, y-self.obs_size), (x+self.obs_size, y-self.obs_size), 
                             (x+self.obs_size, y+self.obs_size), (x-self.obs_size, y+self.obs_size)])
        elif test_case == 2:    # left
            x = 4
            y = 12.5
            self.obs.append([(x-self.obs_size, y-self.obs_size), (x+self.obs_size, y-self.obs_size), 
                             (x+self.obs_size, y+self.obs_size), (x-self.obs_size, y+self.obs_size)])
        elif test_case == 3:    # right
            x = 6
            y = 12.5
            self.obs.append([(x-self.obs_size, y-self.obs_size), (x+self.obs_size, y-self.obs_size), 
                             (x+self.obs_size, y+self.obs_size), (x-self.obs_size, y+self.obs_size)])
        elif test_case == 4:    # 3 obstacles
            x = 5
            y = 20
            self.obs.append([(x-self.obs_size, y-self.obs_size), (x+self.obs_size, y-self.obs_size), 
                             (x+self.obs_size, y+self.obs_size), (x-self.obs_size, y+self.obs_size)])
            x = 8
            y = 10
            self.obs.append([(x-self.obs_size, y-self.obs_size), (x+self.obs_size, y-self.obs_size), 
                             (x+self.obs_size, y+self.obs_size), (x-self.obs_size, y+self.obs_size)])
            x = 3
            y = 6
            self.obs.append([(x-self.obs_size, y-self.obs_size), (x+self.obs_size, y-self.obs_size), 
                             (x+self.obs_size, y+self.obs_size), (x-self.obs_size, y+self.obs_size)])
        elif test_case == 5:    # horizontal obstacles
            x = 3.5
            y = 15
            obs_size_x = 3      # make obstacle longer horizonatally
            self.obs.append([(x-obs_size_x, y-self.obs_size), (x+obs_size_x, y-self.obs_size), 
                             (x+obs_size_x, y+self.obs_size), (x-obs_size_x, y+self.obs_size)])
            x = 9
            self.obs.append([(x-self.obs_size, y-self.obs_size), (x+self.obs_size, y-self.obs_size), 
                             (x+self.obs_size, y+self.obs_size), (x-self.obs_size, y+self.obs_size)])
        # elif test_case == 6:
        #     x = 90
        #     y = 300
        #     self.obs.append([(x-self.obs_size, y-self.obs_size), (x+self.obs_size, y-self.obs_size), 
        #                      (x+self.obs_size, y+self.obs_size), (x-self.obs_size, y+self.obs_size)])
        #     x = 310
        #     self.obs.append([(x-self.obs_size, y-self.obs_size), (x+self.obs_size, y-self.obs_size), 
        #                      (x+self.obs_size, y+self.obs_size), (x-self.obs_size, y+self.obs_size)])
        #     x = 200
        #     y = 50
        #     self.obs.append([(x-self.obs_size, y-self.obs_size), (x+self.obs_size, y-self.obs_size), 
        #                      (x+self.obs_size, y+self.obs_size), (x-self.obs_size, y+self.obs_size)])
        elif test_case == 99:
            # load data file
            with open(self.env_data, "r") as f:
                data = json.load(f)
            self.obs = data["obstacles"]
        else:
            raise ValueError("Invalid test case")

        return self.obs
    
    def position(self, test_case):
        if test_case >= 0 and test_case <= 4:
            self.start_x = 5
            self.start_y = 24
            self.goal_x = 5
            self.goal_y = 1
        elif test_case == 5:
            self.start_x = 2
            self.start_y = 24
            self.goal_x = 8
            self.goal_y = 1
        elif test_case == 99:
            # load data file
            with open(self.env_data, "r") as f:
                data = json.load(f)
            self.start_x = data["start"][0]
            self.start_y = data["start"][1]
            self.goal_x = data["goal"][0]
            self.goal_y = data["goal"][1]
        else:
            raise ValueError("Invalid test case")
        
        return self.start_x, self.start_y, self.goal_x, self.goal_y

