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
            dq.append(new_state.copy())
    else:
        dq.append(new_state.copy())
    ts = torch.from_numpy(np.array([prepare_state(s) for s in dq])).unsqueeze(dim=0)
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

# define loss function
# total loss = (beta * forward_loss) +
#              (1-beta) * inverse loss +
#              (lambda * q_loss)
def loss_fn(forward_loss, inverse_loss, q_loss, params):
    beta = params['beta']
    lmda = params['lambda']
    loss_ = beta * forward_loss + \
            (1-beta) * inverse_loss
    loss = loss_.sum() / loss_.flatten().shape[0]
    loss += lmda * q_loss
    return loss

#reset env and state deque
def reset_env(env, dq):
    env.reset()
    dq.clear()
    ts = prepare_multi_state(dq, env.render())
    return ts

# loss internal curiosity
def ICM(state1, action, state2, encoder, forward_model, inverse_model,\
        forward_loss_fn, inverse_loss_fn, \
        forward_scale = 1., inverse_scale = 1e4):
    state1_hat = encoder(state1)
    state2_hat = encoder(state2)
    state2_hat_pred = forward_model(state1_hat.detach(), action.detach())
    forward_pred_err = forward_loss_fn(state2_hat_pred, state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
    pred_action = inverse_model(state1_hat, state2_hat)
    inverse_pred_err = inverse_loss_fn(pred_action, action.detach().flatten()).unsqueeze(dim=1)
    return forward_scale * forward_pred_err, inverse_scale * inverse_pred_err


def minibatch_train(replay, models, loss_fns, params, use_extrinsic = True):
    encoder = models['encoder']
    qmodel = models['qmodel']
    inverse_model = models['inverse']
    forward_model = models['forward']

    forward_loss_fn = loss_fns['forward_loss']
    inverse_loss_fn = loss_fns['inverse_loss']
    q_loss_fn = loss_fns['q_loss']
    state1_batch, action_batch, reward_batch, state2_batch = replay.get_batch()
    action_batch = action_batch.view(action_batch.shape[0],1)
    reward_batch = reward_batch.view(reward_batch.shape[0],1)

    forward_err, inverse_err = ICM(state1_batch, action_batch, state2_batch, \
                                   encoder, forward_model, inverse_model,\
                                   forward_loss_fn, inverse_loss_fn)
    i_reward = 1. / params['eta'] * forward_err
    reward = i_reward.detach()
    if use_extrinsic:
        reward += reward_batch
    qvals = qmodel(state2_batch)
    reward += params['gamma'] * torch.max(qvals)
    reward_pred = qmodel(state1_batch)
    reward_target = reward_pred.clone()
    indices = torch.stack( ( torch.arange(action_batch.shape[0]), action_batch.squeeze() ), dim =0 )
    indices = indices.tolist()
    reward_target[indices] = reward.squeeze()
    q_loss = 1e5 * q_loss_fn(F.normalize(reward_pred), F.normalize(reward_target.detach()))
    return forward_err, inverse_err, q_loss


# hyperparameters
params = {
    'batch_size':128,
    'beta' : 0.2,
    'lambda': 0.1,
    'eta' : 1.0,
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
load_model_path = 'models.pt'
if len(sys.argv) == 2:
    load_model_path = sys.argv[1]
if not os.path.exists(load_model_path):
    print('no model file exist, so train from scratch')
else:
    checkpoints = torch.load(load_model_path)
    Qmodel.load_state_dict(checkpoints['q_model_state_dict'])
    forward_model.load_state_dict(checkpoints['forward_model_state_dict'])
    inverse_model.load_state_dict(checkpoints['inverse_model_state_dict'])
    encoder.load_state_dict(checkpoints['encoder_state_dict'])
    print('model loaded from the path, '+ load_model_path)
os.makedirs('model_cpt/', exist_ok=True)
forward_loss = nn.MSELoss(reduction='none')
inverse_loss = nn.CrossEntropyLoss(reduction='none')
qloss = nn.MSELoss()
all_model_params = list(Qmodel.parameters()) + list(encoder.parameters())
all_model_params += list(forward_model.parameters()) + list(inverse_model.parameters())
opt = optim.Adam(lr=0.001, params = all_model_params)
#opt = optim.RMSprop(lr=0.001, params = all_model_params)

models = {
        'encoder':encoder,
        'forward':forward_model,
        'inverse':inverse_model,
        'qmodel':Qmodel
}
loss_fns = {
        'forward_loss':forward_loss,
        'inverse_loss':inverse_loss,
        'q_loss':qloss
}

env = gym.make('SuperMarioBros-v0',  apply_api_compatibility=True, render_mode="rgb_array")
env = JoypadSpace(env, COMPLEX_MOVEMENT)
#print(env.action_space)

epochs = 5000
env.reset()
dq = deque(maxlen=params['frames_per_state'])
ts_state1 = prepare_multi_state(dq,env.render())
eps = 0.15
losses = []
episode_len = 0
switch_to_eps_greedy = 1000
e_reward = 0.
last_x_pos = 0
ep_lengths = []

ts_state2 = None
for i in range(epochs):
    if i % 500 == 499:
        torch.save({
            'q_model_state_dict':Qmodel.state_dict(),
            'forward_model_state_dict':forward_model.state_dict(),
            'inverse_model_state_dict':inverse_model.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            'opt_state_dict':opt.state_dict()
            }, "model_cpt/models_%05d.pt" % i)
    if i % 100 == 0:
        print(f'\n{i} epoch start')
    done = False
    opt.zero_grad()
    episode_len += 1
    q_val_pred = Qmodel(ts_state1)
    if i > switch_to_eps_greedy:
        action = int(policy(q_val_pred,eps))
    else:
        action = int(policy(q_val_pred))
    for j in range(params['action_repeats']):
        try:
            state2, e_reward_, term, trunc, info = env.step(action)
            last_x_pos = info['x_pos']
            done = term or trunc
            if done:
                ts_state1 = reset_env(env, dq)
                #print("done and call reset 1")
                break
            e_reward += e_reward_
            ts_state2 = prepare_multi_state(dq, state2)
        except:
            #print("done and call reset except")
            ts_state1 = reset_env(env, dq)
            break

    replay.add_memory(ts_state1, action, e_reward, ts_state2)
    e_reward = 0
    if episode_len > params['max_episode_len']:
        if (info['x_pos'] - last_x_pos) < params['min_progress']:
            done = True
        else:
            last_x_pos = info['x_pos']
    if done:
        #print("done and call reset 2")
        ep_lengths.append(info['x_pos'])
        print(info['x_pos'])
        ts_state1 = reset_env(env, dq)
        last_x_pos = 0
        episode_len = 0
    else:
        ts_state1 = ts_state2
    if len(replay.memory) < params['batch_size']:
        continue
    forward_pred_err, inverse_pred_err, q_loss =  \
            minibatch_train(replay, models, loss_fns, params, False)
    loss = loss_fn(forward_pred_err, inverse_pred_err, q_loss, params)
    #print(ep_lengths[-1])
    if i % 200 == 199:
        losses.append(loss_list)
    loss.backward()
    opt.step()
    loss_list = ( q_loss.mean().detach().numpy(), forward_pred_err.flatten().mean().detach().numpy(), inverse_pred_err.flatten().mean().detach().numpy())
    print(loss_list)


torch.save({
    'q_model_state_dict':Qmodel.state_dict(),
    'forward_model_state_dict':forward_model.state_dict(),
    'inverse_model_state_dict':inverse_model.state_dict(),
    'encoder_state_dict': encoder.state_dict(),
    'opt_state_dict':opt.state_dict()
    }, "models.pt")


losses_ = np.array(losses)
plt.figure(figsize=(8,6))
plt.plot(np.log(losses_[:,0]), label='Q_loss')
plt.plot(np.log(losses_[:,1]), label='forward loss')
plt.plot(np.log(losses_[:,2]), label='inverse loss')
plt.legend()
plt.show()

plt.plot(ep_lengths)
plt.show()
