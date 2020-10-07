import os
import random
import time

import gym
import numpy as np 
from gym import spaces
from gym.utils import seeding, EzPickle
from PIL import Image

from gym.envs.classic_control import rendering

# Sleep after each render
# This allows us to control the actual FPS
RENDER_SLEEP_TIME = 0.01

# Maximum number of steps allowed by each game
# A game terminates when this number is reached 
MAX_NUM_STEPS = 1000

# Rewards for colision
REWARD_COLISION = -1

# Reward for a regular move (no colision, no goal)
REWARD_MOVE = 0

# Base reward should be equal to the max number of steps.
# The final reward is the base reward subtracted by the
# number of total steps in that game.
BASE_REWARD_GOAL = 100

# Base reward for stepping into a cell that has been visited
# The actual reward is multiplied by the number of visits.
BASE_REWARD_PER_VISIT = -0.05

# Radius of the agent vision
# For r=1, agent can see surrounding 3x3 area
# For r=2, agent can see surrounding 5x5 area
# For r=3, agent can see surrounding 7x7 area
# ...
VISION_RADIUS = 2

# For now, we fix the position of the goal
# That makes it easy for a beginner.
DEFAULT_GOAL_POS = (21, 18)

# Display parameters

# Window size
DISP_WIN_WIDTH = 800
DISP_WIN_HEIGHT = 600

# Window size in world coordinates
DISP_WORLD_SIZE = 100
DISP_WORLD_WIDTH = DISP_WORLD_SIZE * DISP_WIN_WIDTH // DISP_WIN_HEIGHT
DISP_WORLD_HEIGHT = DISP_WORLD_SIZE

def load_maze_image(image_file):
    image = np.array(Image.open(image_file)).astype(bool)
    assert image.ndim == 2
    return image

def generate_padded_mmap(mmap, vr):
    """ Pads length of 'vr' on each side of a 2D array
    The array is supposed to have a shape of (h, w).
    """
    assert mmap.ndim == 2
    h, w = mmap.shape
    pw,ph=w+2*vr,h+2*vr
    pmmap=np.zeros((ph,pw), mmap.dtype)
    pmmap[vr:-vr,vr:-vr]=mmap
    return pmmap, pmmap[vr:-vr,vr:-vr]

default_maze_image_file = os.path.join(
    os.path.dirname(__file__), 'data/map.png')
default_maze_image = load_maze_image(default_maze_image_file)



class MazeEnv(gym.Env, EzPickle):
    """ Gym interface
    """
    
    def __init__(self):
        EzPickle.__init__(self)
        core = MazeCore()
        disp = MazeDisp(core)
        self.core = core
        self.disp = disp
        
        fsize = (2*VISION_RADIUS+1)**2
        self.observation_space = spaces.Box(-1, 1, shape=(fsize,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
    
    def step(self, action):
        return self.core.step(action)
    
    def reset(self):
        return self.core.reset()
    
    def render(self, mode='human'):
        return self.disp.render()
    
    def close(self):
        self.disp.close()

class MazeDisp(object):
    
    def __init__(self, core):
        self.core = core
        bsize = min(DISP_WORLD_WIDTH / core.ncols, DISP_WORLD_HEIGHT / core.nrows)
        self.offset_x = (DISP_WORLD_WIDTH - core.ncols * bsize) // 2
        self.offset_y = (DISP_WORLD_HEIGHT - core.nrows * bsize) // 2
        self.bsize = bsize
        self.viewer = None
        
    def render(self):
        if self.viewer is None:
            print('({}, {}) - ({}, {})'.format(
                DISP_WIN_WIDTH, DISP_WIN_HEIGHT,
                DISP_WORLD_WIDTH, DISP_WORLD_HEIGHT,
            ))
            self.viewer = rendering.Viewer(DISP_WIN_WIDTH, DISP_WIN_HEIGHT)
            self.viewer.set_bounds(0, DISP_WORLD_WIDTH, 0, DISP_WORLD_HEIGHT)
        
        for r in range(self.core.nrows):
            for c in range(self.core.ncols):
                tl, br = self._cal_cell_corners_(r, c)
                polygon = [tl, (br[0], tl[1]), br, (tl[0], br[1])]
                
                if self.core.is_wall((r, c)):
                    color = (0.2, 0.2, 0.2)
                else:
                    x = 1 - min(1.0, self.core.get_cell((r, c))['num_visits'] / 10.0)
                    color = (1.0, x, x)
                self.viewer.draw_polygon(polygon, color=color)
        
        t = rendering.Transform(translation=self._cal_cell_center_(*self.core.goal_pos))
        self.viewer.draw_circle(1, 8, color=(0.9,0.1,0.1)).add_attr(t)
        
        t = rendering.Transform(translation=self._cal_cell_center_(*self.core.player_pos))
        self.viewer.draw_circle(1, 8, color=(0.2,0.8,0.2)).add_attr(t)
        
        if RENDER_SLEEP_TIME > 0:
            time.sleep(RENDER_SLEEP_TIME)
        
        mode='human'
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
    def _cal_cell_corners_(self, r, c):
        """ Calculates the top-left and the bottom-right coordinates of a cell
        Returns:
          ((x,y), (x,y)) 
        """
        assert type(r) == int
        assert type(c) == int
        
        x = self.offset_x + c * self.bsize
        y = self.offset_y + (self.core.nrows - r) * self.bsize
        tl = (x, y)
        br = (x + self.bsize, y - self.bsize)
        return tl, br
    
    def _cal_cell_center_(self, r, c):
        """ Calculates the center coordinates of a cell
        """
        tl, br = self._cal_cell_corners_(r, c)
        return ((tl[0]+br[0])/2, (tl[1]+br[1])/2)
        

class MazeCore(object):
    """ Logical component of the maze game
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """
        """
        vr = VISION_RADIUS
        vd = 2*vr+1
        player_pos = (0, 0)
        h, w = default_maze_image.shape
        
        mmap_dtype = [('wall', 'bool'),('num_visits', 'int')]
        mmap_init_val = np.array((False, 0), mmap_dtype)
        self.mmap_dtype = mmap_dtype
        self.mmap_init_val = mmap_init_val
        
        mmap = np.zeros((h, w), dtype=mmap_dtype)
        mmap['wall'] = default_maze_image
        mmap['num_visits'] = 0
        self._mmap_padded, self._mmap = generate_padded_mmap(mmap, vr)
        self._vision_radius = vr
        self._vr = vr # vision radius
        self._vd = 2*vr+1 # vision diameter
        self._w = w
        self._h = h
        self.ncols = self._w
        self.nrows = self._h
        self.goal_pos = self.find_empty_pos(lambda pos : not self.is_wall(pos))
        self.player_pos = self.generate_init_player_pos()
        self.num_steps = 0
        vision = self.generate_observation(self.player_pos)
        return vision
    
    def is_wall(self, pos):
        return self._mmap['wall'][pos[0], pos[1]]
    
    def is_player(self, pos):
        return self.player_pos == pos
    
    def is_goal(self, pos):
        return self.goal_pos == pos
    
    def generate_init_player_pos(self):
        range_radius = 8
        rmin = max(0, self.goal_pos[0] - range_radius)
        rmax = min(self.nrows, self.goal_pos[0] + range_radius)
        cmin = max(0, self.goal_pos[1] - range_radius)
        cmax = min(self.ncols, self.goal_pos[1] + range_radius)
        rc_minmax = (rmin, rmax, cmin, cmax)
        is_empty = lambda pos : not (self.is_wall(pos) or self.is_goal(pos))
        return self.find_empty_pos(is_empty, rc_minmax=rc_minmax)
    
    def find_empty_pos(self, is_empty, rc_minmax=None, max_tries=500):
        """ Finds a pos
        """
        if rc_minmax is not None:
            rmin = max(0, rc_minmax[0])
            rmax = min(self.nrows, rc_minmax[1])
            cmin = max(0, rc_minmax[2])
            cmax = min(self.ncols, rc_minmax[3])
        else:
            rmin = 0
            cmin = 0
            rmax = self.nrows
            cmax = self.ncols
        for i in range(max_tries):
            r = np.random.randint(rmin, rmax)
            c = np.random.randint(cmin, cmax)
            pos = (r, c)
            if not is_empty(pos): continue
            return pos
        raise RuntimeError("Couldn't find an empty place.")
    
    def set_player_pos(self, pos):
        """ Sets the position of the player
        The position must be valid
        """
        assert not self.is_wall(pos) and not self.is_goal(pos), pos
        self.player_pos = pos
    
    def is_in_vision(self, vision_pos, target_pos):
        dr = abs(vision_pos[0] - target_pos[0])
        dc = abs(vision_pos[1] - target_pos[1])
        vr = self._vr
        return dr <= vr and dc <= vr
    
    def generate_observation(self, pos):
        """ 
        """
        r = pos[0]
        c = pos[1]
        vd = self._vd
        vr = self._vr
        vision = np.zeros((vd,vd), np.float32)
        vision -= self._mmap_padded['wall'][r:r+vd,c:c+vd].astype(np.float32)
        dr = self.goal_pos[0] - pos[0]
        dc = self.goal_pos[1] - pos[1]
        if abs(dr) <= vr and abs(dc) <= vr:
            vision[vr+dr,vr+dc] = 1
        return vision.flatten()
    
    def get_cell(self, pos):
        return self._mmap[pos[0], pos[1]]
    
    def step(self, action):
        assert action >= 0 and action < 4
        move = ((0,1),(1,0),(0,-1),(-1,0))[int(action)]
        expected_pos = (self.player_pos[0]+move[0],
                        self.player_pos[1]+move[1])
        
        step_reward = 0.0
        game_end = False
        if self.is_wall(expected_pos):
            step_reward += REWARD_COLISION
        else:
            self.player_pos = expected_pos
            step_reward += REWARD_MOVE
        self.num_steps += 1
        
        step_reward += self._reward_cell_visit()
        self.get_cell(self.player_pos)['num_visits'] += 1
        
        if self.is_goal(self.player_pos):
            discount = 1.0 - float(self.num_steps) / MAX_NUM_STEPS
            step_reward += BASE_REWARD_GOAL * discount
            game_end = True
        
        vision = self.generate_observation(self.player_pos)
        if self.num_steps >= BASE_REWARD_GOAL:
            game_end = True
        return vision, step_reward, game_end, {}
    
    def _reward_cell_visit(self):
        num_visits = self.get_cell(self.player_pos)['num_visits']
        return num_visits * BASE_REWARD_PER_VISIT
    
    def print_map(self):
        """
        """
        lines = []
        for mrow in self._mmap:
            lines.append(['#' if v else ' ' for v in mrow])
        lines[self.player_pos[0]][self.player_pos[1]] = '@'
        lines[self.goal_pos[0]][self.goal_pos[1]] = 'x'
        for line in lines:
            print(' '.join(line))

