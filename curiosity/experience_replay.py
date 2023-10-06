from random import shuffle
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np

class ExperienceReplay:
    def __init__(self, N=1024, batch_size = 128, shuffle_counter = 500):
        self.N = N
        self.batch_size = batch_size
        self.memory = []
        self.counter = 0
        self.shuffle_counter = shuffle_counter

    def add_memory(self, state1, action, reward, state2):
        self.counter += 1
        if self.counter % self.shuffle_counter == 0:
            shuffle(self.memory)
            self.counter = 0

        if len(self.memory) < self.N:
            self.memory.append((state1, action, reward, state2))
        else:
            rand_idx = np.random.randint(0, self.N-1)
            self.memory[rand_idx] = (state1, action, reward, state2)

    def get_batch(self):
        if len(self.memory) < self.batch_size:
            batch_size = len(self.memory)
        else:
            batch_size = self.batch_size
        if len(self.memory) < 1:
            print("No data in Memory")
            return None

        ind = np.random.choice(np.arange(len(self.memory)), batch_size, replace = False)
        batch = [self.memory[i] for i in ind]
        state1_batch = torch.stack([x[0].squeeze(dim = 0) for x in batch])
        action_batch = torch.Tensor([x[1] for x in batch]).long()
        reward_batch = torch.Tensor([x[2] for x in batch])
        state2_batch = torch.stack([x[3].squeeze(dim = 0) for x in batch])
        return state1_batch, action_batch, reward_batch, state2_batch




