# import packages
import math
from collections import deque
from random import shuffle

import matplotlib.pyplot as plt
from magent2.gridworld import GridWorld
from scipy.spatial.distance import cityblock
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, obs_space,hidden_layer, action_space):
        super().__init__()
        self.l1 = nn.Linear(obs_space, hidden_layer)
        self.l2 = nn.Linear(hidden_layer, action_space)

    def forward(self, x):
        y = F.elu(self.l1(x))
        y = F.tanh(self.l2(y))
        return y

def gen_params(N, size):
    ret = []
    for i in range(N):
        vec = torch.randn(size)/10.
        vec.require_grad = True
        ret.append(vec)
    return ret

# return neighbors in radius(r)
def get_neighbors(j, pos_list, r=6):
    neighbors = []
    pos_j = pos_list[j]
    for i,pos in enumerate(pos_list):
        if i == j:
            continue
        dist = cityblock(pos, pos_j)
        if dist < r:
            neighbors.append(i)
    return neighbors

def get_onehot(a,l=21):
    x = torch.zeros(l)
    x[a] = 1.
    return x

def get_scalar(v):
    return torch.argmax(v)

def get_mean_field(j,pos_list, act_list, r=7, l=21):
    neighbors = get_neighbors(j, pos_list, r = r)
    mean_field = torch.zeros(l)
    for k in neighbors:
        act_ = act_list[k]
        act = get_onehot(act_)
        mean_field += act
    tot = mean_field.sum()
    mean_field = mean_field / tot if tot > 0 else mean_field
    return mean_field

def infer_acts(qmodel, obs, pos_list, acts, act_space=21, num_iter = 5, temp=0.5):
    N = acts.shape[0]
    mean_fields = torch.zeros(N,act_space)
    acts_ = acts.clone()
    qvals = torch.zeros(N, act_space)
    states = torch.zeros(N, 359)

    for i in range(num_iter):
        for j in range(N):
            mean_fields[j] = get_mean_field(j, pos_list, acts_)
        states = torch.cat((torch.flatten(obs,1),mean_fields), dim=1)
        qvals = qmodel(states.detach())
        acts_ = torch.multinomial(F.softmax(qvals.detach() / 0.5, dim=1), 1)
    return acts_, mean_fields, qvals

def init_mean_field(N, act_space = 21):
    mean_fields = torch.abs(torch.rand(N, act_space))
    for i in range(mean_fields.shape[0]):
        mean_fields[i] = mean_fields[i] / mean_fields[i].sum()
    return mean_fields

def train(batch_size, replay, qmodel,opt, gamma=0.5 ):
    opt.zero_grad()
    ids = np.random.randint(low = 0, high=len(replay), size = batch_size)
    exps = [replay[idx] for idx in ids]
    losses = []
    jobs = torch.stack([ex[0] for ex in exps]).detach()
    jacts = torch.stack([ex[1] for ex in exps]).detach()
    jrewards = torch.stack([ex[2] for ex in exps]).detach()
    jmeans = torch.stack([ex[3] for ex in exps]).detach()
    vs = torch.stack([ex[4] for ex in exps]).detach()
    qs = []
    for h in range(batch_size):
        states = torch.cat((torch.flatten(jobs[h],0), jmeans[h]), dim=0)
        qs.append(qmodel(states).detach())

    qvals = torch.stack(qs)
    target = qvals.clone().detach()
    target[:,jacts] = jrewards.reshape((-1,1)) + gamma * torch.max(vs, dim=1)[0].reshape((-1,1))
    loss = ((qvals - target.detach())**2).sum()
    loss.requires_grad = True
    losses.append(loss.detach().item())
    loss.backward()
    opt.step()
    return np.array(losses).mean()

def team_step(team, qmodel,acts, env):
    obs, features = env.get_observation(team)
    ids = env.get_agent_id(team)
    obs_small = torch.from_numpy(obs[:,:,:,[1,4]])
    agent_pos = env.get_pos(team)
    acts, act_means, qvals = infer_acts(qmodel, obs_small, agent_pos, acts)
    return acts, act_means, qvals, obs_small, ids

def add_to_replay(replay, obs_small, acts, rewards, act_means, qnext):
    for j in range(rewards.shape[0]):
        exp = (obs_small[j], acts[j], rewards[j], act_means[j], qnext[j])
        replay.append(exp)
    return replay

map_size = 30
env = GridWorld('battle', map_size = map_size)
env.set_render_dir('MAgent/build/render')

team1, team2 = env.get_handles()

hid_layer = 25
in_size = 359
act_space = 21
#layers = [(in_size, hid_layer), (hid_layer, act_space)]
#params = gen_params(2, in_size*hid_layer + hid_layer*act_space)
team1_net = QNetwork(in_size, hid_layer, act_space)
team2_net = QNetwork(in_size, hid_layer, act_space)

checkpoints = torch.load('model.pt')
team1_net.load_state_dict(checkpoints['team1'])
team2_net.load_state_dict(checkpoints['team2'])
all_params = list(team1_net.parameters()) + list(team2_net.parameters())

opt = torch.optim.RMSprop(params = all_params, lr = 0.001)


def reset_env(env, team1, team2):
    width = height = map_size
    n1 = n2 = 16
    gap = 1
    epochs = 100
    replay_size = 70
    batch_size = 25

    side1 = int(math.sqrt(n1)) * 2
    pos1 = []
    for x in range(width // 2 - gap - side1, width//2 - gap, 2):
        for y in range((height - side1)//2, (height-side1)//2+side1,2):
            pos1.append([x,y,0])

    side2 = int(math.sqrt(n2)) * 2
    pos2 = []
    for x in range(width // 2 + gap, width//2 + gap + side2, 2):
        for y in range((height - side2)//2, (height-side2)//2+side2,2):
            pos2.append([x,y,0])

    env.reset()
    env.add_agents(team1, method='custom', pos=pos1)
    env.add_agents(team2, method='custom', pos=pos2)
    return pos1, pos2





width = height = map_size
n1 = n2 = 16
gap = 1
epochs = 100
replay_size = 70
batch_size = 25

side1 = int(math.sqrt(n1)) * 2
pos1 = []
for x in range(width // 2 - gap - side1, width//2 - gap, 2):
    for y in range((height - side1)//2, (height-side1)//2+side1,2):
        pos1.append([x,y,0])

side2 = int(math.sqrt(n2)) * 2
pos2 = []
for x in range(width // 2 + gap, width//2 + gap + side2, 2):
    for y in range((height - side2)//2, (height-side2)//2+side2,2):
        pos2.append([x,y,0])

env.reset()
env.add_agents(team1, method='custom', pos=pos1)
env.add_agents(team2, method='custom', pos=pos2)

N1 = env.get_num(team1)
N2 = env.get_num(team2)

step_ct = 0
acts_1 = torch.randint(low=0, high=act_space, size = (N1,))
acts_2 = torch.randint(low=0, high=act_space, size = (N2,))

replay1 = deque(maxlen=replay_size)
replay2 = deque(maxlen=replay_size)

qnext1 = torch.zeros(N1)
qnext2 = torch.zeros(N2)

act_means1 = init_mean_field(N1,act_space)
act_means2 = init_mean_field(N2,act_space)

rewards1 = torch.zeros(N1)
rewards2 = torch.zeros(N2)

losses1 = []
losses2 = []


'''
obs, features = env.get_observation(team1)
print(obs.shape)
obs = torch.from_numpy(obs[:,:,:,[1,4]])
obs = torch.flatten(obs, 1)
print(obs.shape)
actions = torch.zeros(16, act_space)
states = torch.cat((obs, actions), dim=1)
print(states.shape)

qvals = team1_net(states)
actions = F.softmax(qvals.detach()/0.5, dim=1)
print(actions.shape)
select_action = torch.multinomial(actions, 1)
print(select_action.shape)
print(get_onehot(select_action[0]).shape)
'''
fig, ax = plt.subplots(3,1)
fig.show()
loss_sum1 = 0
loss_sum2 = 0
rewards_sum_arr_1 = []
rewards_sum_arr_2 = []
rewards_sum_1 = 0
rewards_sum_2 = 0
for i in range(epochs):
    print(f'{i} epoch starts')
    done = False
    '''
    pos1, pos2 = reset_env(env, team1, team2)
    N1 = env.get_num(team1)
    N2 = env.get_num(team2)

    step_ct = 0
    acts_1 = torch.randint(low=0, high=act_space, size = (N1,))
    acts_2 = torch.randint(low=0, high=act_space, size = (N2,))
    '''
    while not done:
        acts_1, act_means1, qvals1, obs_small_1, ids_1 = \
                team_step(team1, team1_net, acts_1, env)
        env.set_action(team1, acts_1.detach().numpy().astype(np.int32))

        acts_2, act_means2, qvals2, obs_small_2, ids_2 = \
                team_step(team2, team2_net, acts_2, env)
        env.set_action(team2, acts_2.detach().numpy().astype(np.int32))
        
        done = env.step()

        _,_,qnext1,_,ids_1 = team_step(team1, team1_net, acts_1, env)
        _,_,qnext2,_,ids_2 = team_step(team2, team2_net, acts_2, env)

        env.render()
        rewards1 = torch.from_numpy(env.get_reward(team1)).float()
        rewards2 = torch.from_numpy(env.get_reward(team2)).float()

        replay1 = add_to_replay(replay1, obs_small_1, acts_1, rewards1, act_means1, qnext1)
        replay2 = add_to_replay(replay2, obs_small_2, acts_2, rewards2, act_means2, qnext2)
        shuffle(replay1)
        shuffle(replay2)

        ids_1_ = list(zip(np.arange(ids_1.shape[0]), ids_1))
        ids_2_ = list(zip(np.arange(ids_2.shape[0]), ids_2))

        env.clear_dead()

        ids_1 = env.get_agent_id(team1)
        ids_2 = env.get_agent_id(team2)

        ids_1_ = [i for (i,j) in ids_1_ if j in ids_1]
        ids_2_ = [i for (i,j) in ids_2_ if j in ids_2]

        acts_1 = acts_1[ids_1_]
        acts_2 = acts_2[ids_2_]

        step_ct += 1
        rewards_sum_1 += rewards1.sum()
        rewards_sum_2 += rewards2.sum()
        if step_ct % 250==249:
            losses1.append(loss_sum1)
            losses2.append(loss_sum2)
            loss_sum1 = 0
            loss_sum2 = 0
            rewards_sum_arr_1.append(rewards_sum_1)
            rewards_sum_1 = 0
            rewards_sum_arr_2.append(rewards_sum_2)
            rewards_sum_2 = 0
            break
        if len(replay1) > batch_size and len(replay2) > batch_size:
            loss1 = train(batch_size, replay1, team1_net,opt)
            loss2 = train(batch_size, replay2, team2_net,opt)
            loss_sum1 += loss1
            loss_sum2 += loss2

        ax[0].clear() 
        ax[1].clear()
        ax[2].clear()
        img = np.c_[env.get_global_minimap(30,30), np.zeros(shape=(30,30,1))]
        img *= 255.
        img = np.clip(img, 0, 1)
        ax[0].imshow(img)
        ax[1].plot(losses1)
        ax[1].plot(losses2)
        ax[2].plot(rewards_sum_arr_1)
        ax[2].plot(rewards_sum_arr_2)
        fig.canvas.draw()
        fig.canvas.flush_events()

torch.save({'team1' : team1_net.state_dict(),
            'team2' : team2_net.state_dict()}, 'model.pt')
