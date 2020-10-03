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
python -m spinup.run ppo --hid "[32,32]" --env EoxMaze-v0 --exp_name eox-x3 --gamma 0.999 --epochs 200000 --steps_per_epoch 1000
```

Plot
```bash
python -m spinup.run plot -s 1000 <spinningup>/data/eox-x3/eox-x3_s0
```

Test
```bash
python -m spinup.run test_policy <spinningup>/data/eox-x3/eox-x3_s0
```
