# for env
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

# for preprocessing & visualize
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np

# for RL
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import deque

# custom scripts
from experience_replay import ExperienceReplay
from models import Fnet, Gnet, Qnetwork

# preprocessing resize & gray-scale
def downscale_obs(obs, new_size=(42,42), to_gray = True):
    if to_gray:
        return resize(obs, new_size, anti_aliasing=True).max(axis=2)
    else:
        return resize(obs, new_size, anti_aliasing=True)

# state to torch tensor
def prepare_state(state):
    return torch.from_numpy(downscale_obs(state)).float()

# merge 3 frames into a tensor using deque
def prepare_multi_state(dq, new_state):
    if len(dq) == 0:
        for _ in range(dq.maxlen):
            dq.append(new_state)
    else:
        dq.append(new_state)
    ts = torch.from_numpy(np.array([prepare_state(s) for s in dq]))
    return ts

# extract action from qvalue
# e-greedy + softmax
def policy(qvalues, eps = None):
    if eps is not None:
        if torch.rand(1) < eps:
            return torch.randint(low=0, high=12,size=(1,))
        else:
            return torch.argmax(qvalues)
    else:
        return torch.multinomial(F.softmax(F.normalize(qvalues)), num_samples = 1)

# hyperparameters
params = {
    'batch_size':128,
    'beta' : 0.2,
    'lambda': 0.1,
    'eta' : 1.0
    'gamma' : 0.2,
    'max_episode_len':100,
    'min_progress':15,
    'action_repeats':6,
    'frames_per_state':3
}

# model instances, optimizer, loss functions
replay = ExperienceReplay(N=1024, batch_size=params['batch_size'])
Qmodel = Qnetwork()
encoder = Phi()
forward_model = Fnet()
inverse_model = Gnet()
forward_loss = nn.MSELoss(reduction='none')
inverse_loss = nn.CrossEntropyLoss(reduction='none')
qloss = nn.MSELoss()
all_model_params = list(Qmodel.parameters()) + list(encoder.parameters())
all_model_params += list(forward_model.parameters()) + list(inverse_model.parameters())
opt = optim.Adam(lr=0.001, params = all_model_params)

# if SHOW, render the original frame and gray-scaled frame
SHOW = True
if SHOW:
    fig, ax = plt.subplots(1,2)
    fig.show()

env = gym.make('SuperMarioBros-v0',  apply_api_compatibility=True, render_mode="rgb_array")
env = JoypadSpace(env, COMPLEX_MOVEMENT)
#print(env.action_space)
done = True
env.reset()

dq = deque(maxlen=3)

for step in range(5000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    multi_state = prepare_multi_state(dq, obs)

    # for visualize
    if SHOW:
        ax[0].clear()
        ax[1].clear()
        ax[0].imshow(env.render())
        ax[1].imshow(downscale_obs(env.render()))
        fig.canvas.draw()
        fig.canvas.flush_events()
    if done:
       state = env.reset()
env.close()
