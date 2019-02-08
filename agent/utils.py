import random
import copy
from collections import namedtuple, deque

import torch
import numpy as np

class ReplayBuffer:
    """ Storage for experience tuples """

    def __init__(self, action_size, buffer_size, batch_size, device, seed):
        """ Initialize a random access replay buffer

        Params
        ======
            action_size (int): dimension of the action space
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each batch to retrieve during sampling
            seed (int): random seed
        """
        self.seed = random.seed(seed)

        self.action_size = action_size

        self.storage = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """ Adds an experience to the buffer """
        for i in range(state.shape[0]):
            new_exp = self.experience(state[i], action[i], reward[i], next_state[i], done[i])
            self.storage.append(new_exp)

    def sample(self):
        """ Randomly samples a batch from the memory """
        samples = random.sample(self.storage, k=self.batch_size)

        states = torch.from_numpy(np.vstack([s.state for s in samples if s is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([s.action for s in samples if s is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([s.reward for s in samples if s is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([s.next_state for s in samples if s is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([s.done for s in samples if s is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """ Returns the length of the current buffer """
        return len(self.storage)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state
