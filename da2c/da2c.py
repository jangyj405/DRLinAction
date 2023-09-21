import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
import gymnasium as gym
import torch.multiprocessing as mp
import matplotlib.pyplot as plt


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
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic

def worker(t, worker_model:nn.Module, counter, params):
    sum_epilen = 0
    arr = []
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    worker_opt = optim.Adam(lr = 1e-5, params=worker_model.parameters())
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards = run_episode(worker_env, worker_model)
        actor_loss, critic_loss, epilen = update_params(worker_opt, values, logprobs, rewards)
        counter.value += 1
        sum_epilen += epilen
        if i%10 == 9:
            arr.append(sum_epilen/10)
            sum_epilen = 0
            print(f'{t} worker - {i} epochs')
    plt.plot(arr)
    plt.savefig(f'{t}.png')

def run_episode(worker_env, worker_model):
    #state = torch.from_numpy(worker_env.env.state).float()
    state = torch.from_numpy(worker_env.reset()[0]).float()
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
            #worker_env.reset()
            state = torch.from_numpy(worker_env.reset()[0]).float()
        else:
            reward = 1.0
        rewards.append(reward)
    return values, logprobs, rewards

def update_params(worker_opt, values, logprobs, rewards, clc=1.0, gamma=0.95):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    Returns = []
    ret_ = torch.Tensor([0])
    for r in range(rewards.shape[0]):
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)
    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns, dim=0)
    actor_loss = -1 * logprobs * (Returns - values.detach())
    critic_loss = torch.pow(values-Returns,2)
    loss = actor_loss.sum() + clc*critic_loss.sum()
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss, len(rewards)

if __name__ == "__main__":
    MasterNode = ActorCritic()
    MasterNode.share_memory()
    processes = []
    params = {
        'epochs':3000,
        'n_workers':7
    }
    counter = mp.Value('i',0)
    for i in range(params['n_workers']):
        p = mp.Process(target=worker, args=(i, MasterNode, counter, params))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    for p in processes:
        p.terminate()

    print(counter.value, processes[1].exitcode)
