import numpy as np
import random

from .model import Actor, Critic
from .utils import ReplayBuffer, OUNoise

import torch
import torch.optim as optim
import torch.nn.functional as F

class AgentConfig():
    def __init__(self):
        self.seed = 0
        self.buffer_size = int(1e6)
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 1e-3

        self.actor_hidden_sizes = [400, 300]
        self.lr_actor = 1e-4
        self.critic_hidden_sizes = [400, 300]
        self.lr_critic = 1e-3
        self.critic_weight_decay = 0.

        self.mu = 0.
        self.theta=0.15
        self.sigma=0.2

    def __repr__(self):
        return "Agent Config:\n\tbuffer size: {}\tbatch size: {}\n\tgamma: {}\ttau: {}\n\tactor lr: {}\tcritic lr: {}\n\tmu: {}\ttheta: {}\tsigma: {}".format(self.buffer_size, self.batch_size, self.gamma, self.tau, self.lr_actor, self.lr_critic, self.mu, self.theta, self.sigma);

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():

    def __init__(self, state_size, action_size, config):
        """ Initializes the DQN Agent.

        Params
        ======
            state_size (int): dimensions in the state space
            action_size (int): action space dimension
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        self.seed = random.seed(config.seed)

        self.actor_online = Actor(state_size, action_size,
                                  self.config.seed,
                                  hidden_sizes=self.config.actor_hidden_sizes).to(device)
        self.actor_target = Actor(state_size, action_size,
                                  self.config.seed,
                                  hidden_sizes=self.config.actor_hidden_sizes).to(device)
        self.actor_target.load_state_dict(self.actor_online.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_online.parameters(), lr=self.config.lr_actor)

        self.critic_online = Critic(state_size, action_size,
                                    self.config.seed,
                                    hidden_sizes=self.config.critic_hidden_sizes).to(device)
        self.critic_target = Critic(state_size, action_size,
                                    self.config.seed,
                                    hidden_sizes=self.config.critic_hidden_sizes).to(device)
        self.critic_target.load_state_dict(self.critic_online.state_dict())
        self.critic_optimizer = optim.Adam(self.critic_online.parameters(),
                                           lr=self.config.lr_critic,
                                           weight_decay=self.config.critic_weight_decay)

        self.replay = ReplayBuffer(action_size, self.config.buffer_size,
                                   self.config.batch_size, device, self.config.seed)

        self.noise = OUNoise(action_size, self.config.seed,
                             self.config.mu, self.config.theta, self.config.sigma)

    def step(self, state, action, reward, next_state, done):
        """ Learning step for the agent """

        # save the step in the replay buffer
        self.replay.add(state, action, reward, next_state, done)

        # learn update_every time steps
        if len(self.replay) > self.config.batch_size:
            experiences = self.replay.sample()
            self.learn(experiences, self.config.gamma)


    def act(self, state, add_noise=True):
        """ Returns actions for the state based on the current policy

        Params
        ======
            state (array_like): current state of the environment
            add_noise (bool): add OUNoise to the output
        """
        # move the state to the device
        state = torch.from_numpy(state).float().to(device)

        # get the action values from the online network
        self.actor_online.eval()
        with torch.no_grad():
            action_values = self.actor_online(state).cpu().data.numpy()
        self.actor_online.train()

        if add_noise:
            action_values += self.noise.sample()

        return np.clip(action_values, -1, 1)


    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        next_actions = self.actor_target(next_states)

        q_targets_next = self.critic_target(next_states, next_actions.detach())

        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        q_expected = self.critic_online(states, actions)

        # compute and minimize the loss
        critic_loss = F.mse_loss(q_expected, q_targets.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.critic_online.parameters(), 1)
        self.critic_optimizer.step()

        online_actions = self.actor_online(states)
        actor_loss = -self.critic_online(states, online_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update the target network
        self.soft_update(self.actor_online, self.actor_target, self.config.tau)
        self.soft_update(self.critic_online, self.critic_target, self.config.tau)

    def soft_update(self, online_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_online + (1 - τ)*θ_target

        Params
        ======
            online_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
