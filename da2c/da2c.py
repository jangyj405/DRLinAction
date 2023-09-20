import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
import gymnasium as gym
import torch.multiprocessing as mp

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4,25);
        self.l2 = nn.Linear(25,50);
        self.actor_lin1 = nn.Linear(50,2)
        self.l3 = nn.Linear(50,25)
        self.critic_lin1 = nn.Linear(25,1)

    def forward(self, x):
        x = F.normalize(x, dim = 0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y), dim=0)
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1)
        return actor, critic

def worker(t, worker_model:nn.Module, counter, params):
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    worker_opt = optim.Adam(lr = 1e-4, params=worker_model.parameters())
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards = run_episode(worker_env, worker_model)
        actor_loss, critic_loss, epilen = update_params(worker_opt, values, logprobs, rewards)
        counter.value += 1

def run_episode(worker_env, worker_model):
    state = torch.from_numpy(worker_env.env.state).float()
    values, logprobs,rewards = [],[],[]
    done = False
    j = 0
    while done == False:
        j += 1
        policy, value = worker_model(state)
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, _, term, trunc, info = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()
        done = term or trunc
        if done:
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0
        rewards.append(reward)
    return values, logprobs, rewards


