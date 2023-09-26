# Setup for Super Mario Bros Gym

## test environment
```
CPU : Intel i5
RAM : 32G
GPU : NVIDIA GTX 1660 6GB
OS : Ubuntu22.04 on Windows11 WSL
Python3.9.18 Anaconda Virtual environment
OpenAI Gym Version : 0.26.2
```

## installation
```
conda create -n "env_name" python=3.9
conda activate env_name
python -m pip install -U pip
python -m pip install gym
python -m pip install gym-super-mario-bros
```

## Test Code
```python
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

env = gym.make('SuperMarioBros-v0',  apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True
env.reset()
for step in range(5000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if done:
       state = env.reset()
env.close()
```
if this not works, try belows.


## edit time_limit.py scripts
### file path
```
vi /home/ubuntu/anaconda3/envs/gym_env_origin/lib/python3.9/site-packages/gym/wrappers/time_limit.py
```

### replace "step" function to belows
```python
    def step(self, action):
        """Steps through the environment and if the number of steps elapsed exceeds ``max_episode_steps`` then truncate.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, terminated, truncated, info)`` with `truncated=True`
            if the number of steps elapsed >= max episode steps

        """

        truncated = False
        result = self.env.step(action)

        if (len(result) == 4):
            observation, reward, terminated, info = self.env.step(action)
        elif (len(result) == 5):
            observation, reward, terminated, truncated, info = self.env.step(action)
        else:
            raise Exception('Tupel length of step return was neither 4 nore 5. Stop run.')

        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        return observation, reward, terminated, truncated, info
```

## install libstdcxx-ng
```
conda install -c conda-forge libstdcxx-ng
# if not work after installation, use this option
# LIBGL_ALWAYS_SOFTWARE=1
```
