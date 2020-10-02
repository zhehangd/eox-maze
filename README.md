# Maze Environment for Gym and Spinning Up

# Install 

copy `eox_maze.py` to `gym/envs/box2d/`. In `gym/envs/box2d/__init__.py` add `from gym.envs.box2d.eox_maze import MazeEnv` after other import statements. In `gym/envs/__init__.py` add 
```python
register(
    id='EoxMaze-v0',
    entry_point='gym.envs.box2d:MazeEnv',
    max_episode_steps=200,
    reward_threshold=100,
)
```
