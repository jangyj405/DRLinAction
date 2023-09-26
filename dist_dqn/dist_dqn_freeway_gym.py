import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import gym
from collections import deque
import random
from random import shuffle
def update_dist(r, support, probs, lim=(-10,10), gamma = 0.8):
    nsup = probs.shape[0]
    vmin, vmax = lim[0],lim[1]
    dz = (vmax-vmin)/(nsup-1.)
    bj = np.round((r-vmin)/dz)
    bj = int(np.clip(bj,0, nsup-1))
    m = probs.clone()
    j = 1
    for i in range(bj,1,-1):
        m[i] += np.power(gamma,j) * m[i-1]
        j += 1
    j = 1
    for i in range(bj, nsup-1,1):
        m[i] += np.power(gamma,j) * m[i-1]
        j += 1
    m /= m.sum()
    return m

def dist_dqn(x, theta, aspace = 3):
    dim0, dim1,dim2,dim3 = 128,100,25,51
    t1 = dim0*dim1
    t2 = dim2 * dim1
    theta1 = theta[0:t1].reshape(dim0,dim1)
    theta2 = theta[t1:t1+t2].reshape(dim1,dim2)
    l1 = x @ theta1
    l1 = torch.selu(l1)
    l2 = l1 @ theta2
    l2 = torch.selu(l2)
    l3 = []
    for i in range(aspace):
        step = dim2 * dim3
        theta5_dim = t1 + t2 + i + step
        theta5 = theta[theta5_dim:theta5_dim+step].reshape(dim2,dim3)
        l3_ = l2 @ theta5
        l3.append(l3_)
    l3 = torch.stack(l3, dim = 1)
    l3 = F.softmax(l3, dim=2)
    return l3.squeeze()

def get_target_dist(dist_batch, action_batch, reward_batch, support,
                    lim = (-10,10), gamma=0.8):
    nsup = support.shape[0]
    vmin, vmax = lim[0], lim[1]
    dz = (vmax-vmin)/(nsup-1.)
    target_dist_batch = dist_batch.clone()
    for i in range(dist_batch.shape[0]):
        dist_full = dist_batch[i]
        action = int(action_batch[i].item())
        dist = dist_full[action]
        r = reward_batch[i]
        if r != -1:
            target_dist = torch.zeros(nsup)
            bj = np.round((r-vmin)/dz)
            bj = int(np.clip(bj,0, nsup-1))
            target_dist[bj] = 1.
        else:
            target_dist = update_dist(r,support, dist, lim=lim, gamma = gamma)
        target_dist_batch[i, action, :] = target_dist
    return target_dist_batch

def lossfn(x,y):
    loss = torch.Tensor([0.])
    loss.requires_grad = True
    for i in range(x.shape[0]):
        loss_ = -1 * torch.log(x[i].flatten(start_dim = 0)) @ \
                y[i].flatten(start_dim = 0)
        loss = loss + loss_
    return loss

def test_dist_dqn_functions_one_step():
    torch.manual_seed(42)
    aspace = 3
    tot_params = 128 * 100 + 25 * 100 + aspace*25*51
    theta = torch.randn(tot_params)/10.
    theta.requires_grad = True
    theta_2 = theta.detach().clone()

    vmin, vmax = -10, 10
    gamma = 0.9
    lr = 0.00001
    update_rate = 75
    support = torch.linspace(-10,10,51)
    state = torch.randn(2,128)/10.
    action_batch = torch.Tensor([0,2])
    reward_batch = torch.Tensor([0,10])
    losses = []
    pred_batch = dist_dqn(state, theta, aspace = aspace)
    target_dist = get_target_dist(pred_batch, action_batch, reward_batch,
                                   support, lim = (vmin, vmax), gamma = gamma)
    plt.plot((target_dist.flatten(start_dim = 1)[0].data.numpy()), color='red', label='target')
    plt.plot((pred_batch.flatten(start_dim = 1)[0].data.numpy()), color='green', label='pred')
    plt.legend()
    plt.show()

def test_dist_dqn_simul():
    torch.manual_seed(42)
    aspace = 3
    tot_params = 128 * 100 + 25 * 100 + aspace*25*51
    theta = torch.randn(tot_params)/10.
    theta.requires_grad = True
    theta_2 = theta.detach().clone()

    vmin, vmax = -10, 10
    gamma = 0.9
    lr = 0.00001
    update_rate = 75
    support = torch.linspace(-10,10,51)
    state = torch.randn(2,128)/10.
    action_batch = torch.Tensor([0,2])
    losses = []
    for i in range(1000):
        reward_batch = torch.Tensor([0,8]) + torch.randn(2)/10.
        pred_batch = dist_dqn(state, theta, aspace = aspace)
        pred_batch2 = dist_dqn(state,theta_2, aspace=aspace)
        target_dist = get_target_dist(pred_batch2, action_batch, reward_batch,
                                      support, lim=(vmin,vmax), gamma=gamma)
        loss = lossfn(pred_batch, target_dist.detach())
        losses.append(loss.item())
        loss.backward()
        with torch.no_grad():
            theta -= lr * theta.grad
        theta.requires_grad = True

        if i% update_rate == 0:
            theta_2 = theta.detach().clone()
    fig, ax = plt.subplots(1,2)
    ax[0].plot((target_dist.flatten(start_dim = 1)[0].data.numpy()), color='red', label='target')
    ax[0].plot((pred_batch.flatten(start_dim = 1)[0].data.numpy()), color='green', label='pred')
    ax[1].plot(losses)
    plt.show()

def preproc_state(state):
    p_state = torch.from_numpy(state).unsqueeze(dim = 0).float()
    p_state = F.normalize(p_state, dim=1)
    return p_state
    
def get_action(dist, support):
    actions = []
    for b in range(dist.shape[0]):
        expectations = [support @ dist[b,a,:] for a in range(dist.shape[1])]
        action = int(np.argmax(expectations))
        actions.append(action)
    actions = torch.Tensor(actions).int()
    return actions

def test_model(theta, aspace, support):
    env = gym.make("Freeway-ram-v0", render_mode="human")
    env.env.get_action_meanings()
    fig, ax = plt.subplots(1,3)
    for i in range(3):
        ax[i].set_ylim([0., 1.])
    fig.show()
    while 1:
        done = False
        state = preproc_state(env.reset()[0])
        while not done:
            ax[0].clear()
            ax[1].clear()
            ax[2].clear()
            
            for i in range(3):
                ax[i].set_ylim([0., 1.])
            pred = dist_dqn(state, theta, aspace = aspace)
            action = get_action(pred.unsqueeze(dim = 0).detach(),support).item()
            state2, reward, term, trunc, info = env.step(action)
            ax[0].bar(support,pred.detach().numpy()[0])
            ax[1].bar(support,pred.detach().numpy()[1])
            ax[2].bar(support,pred.detach().numpy()[2])
            fig.canvas.draw()
            fig.canvas.flush_events()
            #a = random.randint(0,2)
            #state2, reward, term, trunc, info = env.step(a)
            done = term or trunc
            state2 = preproc_state(state2)
            state = state2

    

def freeway():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    env = gym.make("Freeway-ram-v0")
    aspace = 3
    env.env.get_action_meanings()
    
    vmin, vmax = -10, 10
    replay_size = 200
    batch_size = 50
    nsup = 51
    dz = (vmax-vmin)/(nsup-1)
    support = torch.linspace(vmin, vmax, nsup)

    replays = deque(maxlen = replay_size)
    lr = 0.0001
    gamma = 0.1
    epochs = 5000
    eps = 0.20
    eps_min = 0.05
    priority_level = 5
    update_freq = 25

    tot_params = 128*100 + 25 * 100 + aspace * 25*51
    theta = torch.randn(tot_params)/10.
    theta.requires_grad = True
    theta_2 = theta.detach().clone()
    
    #test_model(theta, aspace, support)
    losses = []
    cum_rewards = []
    renders = []
    state = preproc_state(env.reset()[0])

    for i in range(epochs):
        pred = dist_dqn(state, theta, aspace = aspace)
        if i < replay_size or np.random.rand(1) < eps:
            action = np.random.randint(aspace)
        else:
            action = get_action(pred.unsqueeze(dim = 0).detach(),support).item()
        state2, reward, term, trunc, info = env.step(action)
        done = term or trunc
        state2 = preproc_state(state2)
        if reward == 1:
            cum_rewards.append(1)
            print("clear")
        reward = 10 if reward == 1 else reward
        reward = -10 if done else reward
        reward = -1 if reward == 0 else reward
        exp = (state,action,reward, state2)
        replays.append(exp)

        if reward == 10:
            for e in range(priority_level):
                replays.append(exp)

        shuffle(replays)
        state = state2

        if len(replays) == replay_size:
            indx = np.random.randint(low = 0, high = len(replays),size = batch_size)
            exps = [replays[j] for j in indx]
            state_batch = torch.stack([ex[0] for ex in exps], dim=1).squeeze()
            action_batch = torch.Tensor([ex[1] for ex in exps])
            reward_batch = torch.Tensor([ex[2] for ex in exps])
            state2_batch = torch.stack([ex[3] for ex in exps], dim=1).squeeze()

            pred_batch = dist_dqn(state_batch.detach(), theta, aspace = aspace)
            pred2_batch = dist_dqn(state2_batch.detach(), theta_2, aspace = aspace)
            target_dist = get_target_dist(pred2_batch, action_batch, reward_batch,
                                          support, lim=(vmin, vmax), gamma = gamma)
            loss = lossfn(pred_batch, target_dist.detach())
            losses.append(loss.item())
            loss.backward()
            with torch.no_grad():
                theta -= lr * theta.grad
            theta.requires_grad = True

        if i%update_freq == 0:
            theta_2 = theta.detach().clone()
        if i > 100 and eps > eps_min:
            dec = 1./np.log2(i)
            dec /= 1e3
            eps -= dec
        if i % 100 == 0:
            print(f"current epochs : {i}, current eps : {eps}")
        if done:
            state = preproc_state(env.reset()[0])
            done = False
    print("cum_rewards len = " + str(len(cum_rewards)))
    plt.plot(losses)
    plt.show()
    test_model(theta, aspace, support)

if __name__=="__main__":
    #test_dist_dqn_functions_one_step()
    #test_dist_dqn_simul()
    freeway()



