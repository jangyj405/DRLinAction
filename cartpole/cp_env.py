import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def discount_reward(rewards, gamma = 0.99):
    lenr = len(rewards)
    disc_return = torch.pow(gamma, torch.arange(lenr).float())*rewards
    disc_return /= disc_return.max()
    return disc_return

def loss_fn(preds, r):
    return -1 * torch.sum(r*torch.log(preds))

n_input = 4
l1_unit = 150
n_output = 2
learning_rate = 9e-4
gamma = 0.99

model = nn.Sequential(nn.Linear(n_input, l1_unit),
                      nn.LeakyReLU(),
                      nn.Linear(l1_unit, n_output),
                      nn.Softmax(dim=1)
        )
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

MAX_DUR = 200
MAX_EPISODES = 500
scores = []

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)
for epi in range(MAX_EPISODES):
    curr_state = env.reset()[0]
    done = False
    transitions = []
    for trans in range(MAX_DUR):
        state_tensor = torch.from_numpy(np.array(curr_state, dtype=np.float32).reshape([1,-1]))
        act_prob = model(state_tensor)
        action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy()[0])
        prev_state = curr_state
        curr_state, _, term, trun, info = env.step(action)
        transitions.append([prev_state, action, trans+1])
        if term or trun:
            break
    
    ep_len = len(transitions)
    scores.append(ep_len)

    reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,))
    disc_reward = discount_reward(reward_batch, gamma)
    state_batch = torch.Tensor([s for (s,a,r) in transitions])
    action_batch = torch.Tensor([a for (s,a,r) in transitions])
    pred_batch = model(state_batch)
    prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1,1)).squeeze()
    loss = loss_fn(prob_batch, disc_reward)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

env.close()
plt.plot(scores)
plt.show()

avgs =[]
num_for_avg = 10
for i in range(num_for_avg):
    s = scores[i*MAX_EPISODES//num_for_avg:(i+1)*MAX_EPISODES//num_for_avg]
    avg = sum(s)/len(s)
    avgs.append(avg)
plt.plot(avgs)
plt.show()

