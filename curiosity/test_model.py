import sys
import os
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
from models import Phi, Fnet, Gnet, Qnetwork

def policy(qvalues, eps = None):
    if eps is not None:
        if torch.rand(1) < eps:
            return torch.randint(low=0, high=12,size=(1,))
        else:
            return torch.argmax(qvalues)
    else:
        return torch.multinomial(F.softmax(F.normalize(qvalues)), num_samples = 1)

def downscale_obs(obs, new_size=(42,42), to_gray = True):
    if to_gray:
        return resize(obs, new_size, anti_aliasing=True).max(axis=2)
    else:
        return resize(obs, new_size, anti_aliasing=True)

def prepare_state(state):
    return torch.from_numpy(downscale_obs(state)).float()

def prepare_multi_state(dq, new_state):
    if len(dq) == 0:
        for _ in range(dq.maxlen):
            dq.append(new_state.copy())
    else:
        dq.append(new_state.copy())
    ts = torch.from_numpy(np.array([prepare_state(s) for s in dq])).unsqueeze(dim=0)
    return ts

model_path= 'models.pt'
if len(sys.argv) == 2:
    model_path = sys.argv[1]

if not os.path.exists(model_path):
    print('no model file: ' + model_path)
    exit(1)

Qmodel = Qnetwork()
encoder = Phi()
forward_model = Fnet()
inverse_model = Gnet()

checkpoints = torch.load(model_path)
Qmodel.load_state_dict(checkpoints['q_model_state_dict'])

SHOW = True
if SHOW:
    fig, ax = plt.subplots(1,2)
    fig.show()

env = gym.make('SuperMarioBros-v0',  apply_api_compatibility=True, render_mode="rgb_array")
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env.reset()
dq = deque(maxlen=3)
multi_state = prepare_multi_state(dq, env.render())
epochs = 10
max_eplen = 1000
epi_len = []
action = 0
for i in range(epochs):
    print(f"{i} epoch start")
    count = 0
    try:
        while 1:
            count+=1
            qval_pred = Qmodel(multi_state)
            action = int(policy(qval_pred,0.15))
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
            if done or count > max_eplen or (count > 500 and info['x_pos'] < 150):
                epi_len.append(info['x_pos'])
                obs,_ = env.reset()
                dq.clear()
                multi_state = prepare_multi_state(dq, obs)
                break
    except:
        epi_len.append(info['x_pos'])
        obs,_ = env.reset()
        dq.clear()
        multi_state = prepare_multi_state(dq, obs)

env.close()
plt.close()
plt.plot(epi_len)
plt.show()
