# Maze Environment for Gym and Spinning Up

# Install 

copy `eox_maze.py` to `gym/envs/box2d/`. In `gym/envs/box2d/__init__.py` add `from gym.envs.box2d.eox_maze import MazeEnv` after other import statements. In `gym/envs/__init__.py` add 
```python
register(
    id='EoxMaze-v0',
    entry_point='gym.envs.box2d:MazeEnv',
    max_episode_steps=1000,
    reward_threshold=100,
)
```

# Train and Test

Train
```bash
python train.py
```

Plot traning progress
```bash
python -m spinup.run plot [-s 1000] <data-dir/>
```

Test traning result
```bash
python test_policy.py  <data-dir>
```
I have to make a local copy of test_policy.py otherwise it can't find the local modules.
# Test

```bash
python -m unittest discover -s test
```

