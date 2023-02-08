import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RubiksCubeEnv(gym.Env):
    def __init__(self):
        # Define the observation and action spaces
        # create rubixube space made with 54 slices where each slices has 0~5 colors
        self.observation_space = spaces.MultiDiscrete(6*np.ones([6,3,3]))
        self.action_space = spaces.Text(max_length=1, charset={"F","F'","B","B'","R","R'","L","L'","U","U'","D","D'"})
        self.cube = np.arange(6)[:,np.newaxis,np.newaxis]*np.ones([6,3,3])
        self.FaceDictI2C = {0:'U', 1:'L', 2:'F', 3:'R', 4:'B', 5:'D'}
        self.FaceDictC2I = {'U':0, 'L':1, 'F':2, 'R':3, 'B':4, 'D':5}
        self.Neighbors = {'F':[('L',[8,5,2]), ('U',[6,7,8]), ('R',[0,3,6]), ('D',[2,1,0])],
                          'R':[('F',[8,5,2]), ('U',[8,5,2]), ('B',[0,3,6]), ('D',[8,5,2])],
                          'B':[('R',[8,5,2]), ('U',[2,1,0]), ('L',[0,3,6]), ('D',[6,7,8])],
                          'L':[('B',[8,5,2]), ('U',[0,3,6]), ('F',[0,3,6]), ('D',[0,3,6])],
                          'U':[('L',[2,1,0]), ('B',[2,1,0]), ('R',[2,1,0]), ('F',[2,1,0])],
                          'D':[('L',[6,7,8]), ('F',[6,7,8]), ('R',[6,7,8]), ('B',[6,7,8])],
                        }
        face = np.arange(1,10).reshape((3,3))
        self.RGBDict ={0: ('W', (255, 255, 255)),
                       1: ('O', (255, 165, 0)),
                       2: ('G', (0, 128, 0)),
                       3: ('R', (255, 0, 0)),
                       4: ('B', (0, 0, 255)),
                       5: ('Y', (255, 255, 0))}
        # self.Neighbors = {'U':}
        # Initialize the state of the cube
        self.state = self._reset()
        
    def reset(self):
        # Reset the state of the cube to its initial configuration
        self.state = [0 for i in range(54)]
        return self.state

    def action_description(action):
        facel = action[0]
        assert facel in ["F","B", "R", "L", "U", "D"]
        clock = False if "'" in action else True
        return facel, clock

    def step(self, action):
        # Perform the specified action on the cube
        # ...
        target_face, clock = action_description(action)
        FaceDictC2I[target_face]
        clock = False

        
        self.cube[FaceDictC2I[target_face]] = np.rot90(self.cube[FaceDictC2I[target_face]],clock*2+1)
        queue = np.concatenate([cube[FaceDictC2I[neigh_face]].flatten()[neigh_locs] for neigh_face, neigh_locs in Neighbors[target_face]])
        if clock:
            queue = np.roll(queue,3)
        else:
            queue = np.roll(queue,-3)
        for i, (neigh_face, neigh_locs) in enumerate(Neighbors[target_face]):
            print(i)
            print(neigh_face, neigh_locs)
            temp_fl = cube[FaceDictC2I[neigh_face]].flatten()
            temp_fl
            temp_fl = cube[FaceDictC2I[neigh_face]].flatten()[neigh_locs]

            print(neigh_face, neigh_locs)


        # Compute the reward based on the new state of the cube
        # ...
        
        # Return the new state of the cube, the reward, a flag indicating whether the episode is done, and any additional information
        return self.state, reward, done, {}
    
    def _render(self, mode='human', close=False):
        # Render the current state of the cube
        # ...
