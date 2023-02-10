#%%
import numpy as np
import gymnasium as gym
import pygame
from gymnasium import spaces

class RubiksCubeEnv(gym.Env):
    def __init__(self, render_mode = None):
        self.window_size = 512

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Define the observation and action spaces
        # create rubixube space made with 54 slices where each slices has 0~5 colors
        self.observation_space = spaces.MultiDiscrete(6*np.ones([6,3,3]))
        self.action_space = spaces.Text(max_length=1, charset={"F","F'","B","B'","R","R'","L","L'","U","U'","D","D'"})
        self.cube = self.cube_initializing()
        self.scramble_history = None
        self.step_counter = 0
    
        self.correct_state = self.cube.copy()

        self.FaceDictI2C = {0:'U', 1:'L', 2:'F', 3:'R', 4:'B', 5:'D'}
        self.FaceDictC2I = {'U':0, 'L':1, 'F':2, 'R':3, 'B':4, 'D':5}
        self.Neighbors = {'F':[('L',[8,5,2]), ('U',[6,7,8]), ('R',[0,3,6]), ('D',[2,1,0])],
                          'R':[('F',[8,5,2]), ('U',[8,5,2]), ('B',[0,3,6]), ('D',[8,5,2])],
                          'B':[('R',[8,5,2]), ('U',[2,1,0]), ('L',[0,3,6]), ('D',[6,7,8])],
                          'L':[('B',[8,5,2]), ('U',[0,3,6]), ('F',[0,3,6]), ('D',[0,3,6])],
                          'U':[('L',[2,1,0]), ('B',[2,1,0]), ('R',[2,1,0]), ('F',[2,1,0])],
                          'D':[('L',[6,7,8]), ('F',[6,7,8]), ('R',[6,7,8]), ('B',[6,7,8])],
                        }
        # face = np.arange(1,10).reshape((3,3))
        self.RGBDict ={0: ('W', (255, 255, 255)),
                       1: ('O', (255, 165, 0)),
                       2: ('G', (0, 128, 0)),
                       3: ('R', (255, 0, 0)),
                       4: ('B', (0, 0, 255)),
                       5: ('Y', (255, 255, 0))}

    def cube_initializing(self):
        cube = np.arange(6,dtype=np.int8)[:,np.newaxis,np.newaxis]*np.ones([6,3,3],dtype=np.int8)
        return cube

    def scramble(self, l_scramble):
        scramble_list = [self.action_space.sample() for i in range(l_scramble)]
        for action in scramble_list:
            target_face, clock = self.action_decomposition(action)
            self.cube = self.RotatingCube(target_face, clock)
        return self.cube, scramble_list

    def reset(self, l_scramble):
        self.cube = self.cube_initializing()
        self.cube, self.scramble_history = self.scramble(l_scramble)
        self.step_counter = 0

        if self.render_mode == "human":
            self._render_frame()

        return self.cube

    def action_decomposition(self,action):
        target_face = action[0]
        assert target_face in ["F","B", "R", "L", "U", "D"]
        clockwise = False if "'" in action else True
        return target_face, clockwise

    def RotatingCube(self, target_face, clock):
        # Step 1 : Rotate Target Face 
        # if clockwise is true, rotate 3 times i.e., equal to clockwise 90 degree rotation
        # else, counterclockwise, rotate 1 time, i.e., equal to counter clockwise 90 degree rotation
        self.cube[self.FaceDictC2I[target_face]] = np.rot90(self.cube[self.FaceDictC2I[target_face]],clock*2+1)

        # Step 2 : Rotate Neighboring Face
        # Gather neighboring pieces
        queue = np.concatenate([self.cube[self.FaceDictC2I[neigh_face]].flatten()[neigh_locs] for neigh_face, neigh_locs in self.Neighbors[target_face]])
        # Rotate pieces by 3. if clockwise=True, roll 3, else, roll -3 (reverse order)
        queue = np.roll(queue, 6*clock-3)
        # Update pieces based on the queue
        for i, (neigh_face, neigh_locs) in enumerate(self.Neighbors[target_face]):
            temp_fl = self.cube[self.FaceDictC2I[neigh_face]].flatten()
            temp_fl[neigh_locs] = queue[(3*i):(3*(i+1))]
            self.cube[self.FaceDictC2I[neigh_face]] = temp_fl.reshape((3,3))
        return self.cube

    def step(self, action):
        # Perform the specified action on the cube
        target_face, clock = self.action_decomposition(action)
        self.RotatingCube(target_face, clock)
        self.step_counter += 1
        # Compute the reward based on the new state of the cube
        if (self.cube == self.correct_state).all():
            reward = 100
            terminated = True
            truncated = False
        else:
            reward = -1
            terminated = False        
            truncated = False
        info = {'count':self.step_counter}
        # Return the new state of the cube, the reward, a flag indicating whether the episode is done, and any additional information
        if self.render_mode == "human":
            self._render_frame()

        return self.cube, reward, terminated, truncated, info

    def ColoringInt2Char(self):
        text_cube = np.array([self.RGBDict[int(x)][0] for x in np.nditer(self.cube)]).reshape(6,3,3)
        return text_cube

    def print_state(self, mode = 'c'):
        if mode == 'c':
            pr_cube = self.ColoringInt2Char()
            space = '             '

        else:
            pr_cube = self.cube
            space = '       '

        print(space,pr_cube[self.FaceDictC2I['U']][0,:])
        print(space,pr_cube[self.FaceDictC2I['U']][1,:])
        print(space,pr_cube[self.FaceDictC2I['U']][2,:])
        print(pr_cube[self.FaceDictC2I['L']][0,:],pr_cube[self.FaceDictC2I['F']][0,:],pr_cube[self.FaceDictC2I['R']][0,:],pr_cube[self.FaceDictC2I['B']][0,:])
        print(pr_cube[self.FaceDictC2I['L']][1,:],pr_cube[self.FaceDictC2I['F']][1,:],pr_cube[self.FaceDictC2I['R']][1,:],pr_cube[self.FaceDictC2I['B']][1,:])
        print(pr_cube[self.FaceDictC2I['L']][2,:],pr_cube[self.FaceDictC2I['F']][2,:],pr_cube[self.FaceDictC2I['R']][2,:],pr_cube[self.FaceDictC2I['B']][2,:])
        print(space,pr_cube[self.FaceDictC2I['D']][0,:])
        print(space,pr_cube[self.FaceDictC2I['D']][1,:])
        print(space,pr_cube[self.FaceDictC2I['D']][2,:])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()


    
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

#%%
pygame.init()
pygame.display.init()
window_size = 512
window =pygame.display.set_mode((window_size, 2*window_size))
clock = pygame.time.Clock()

canvas = pygame.Surface((window_size, 2*window_size))
canvas.fill((255, 255, 255))
pix_square_size = (window_size / 12)
# pygame.draw.rect(canvas, ()
#             canvas,
#             (255, 0, 0),
#             pygame.Rect(
#                 pix_square_size * self._target_location,
#                 (pix_square_size, pix_square_size),
#             ),
#         )

for x in range(13):
    pygame.draw.line(canvas, 0, (0, pix_square_size * x), (window_size, pix_square_size * x), width = 3)
    pygame.draw.line(canvas, 0, (pix_square_size * x, 0), (pix_square_size * x, window_size),width=3)
pygame.event.pump()
pygame.display.update()
clock.tick(4)
#%%
import pygame

background_colour = (255,255,255)
(width, height) = (300, 200)

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Tutorial 1')
screen.fill(background_colour)

pygame.display.flip()

running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
#%%
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            # window = pygame.display.set_mode((100,100))
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        # canvas = pygame.Surface((100,100))
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )