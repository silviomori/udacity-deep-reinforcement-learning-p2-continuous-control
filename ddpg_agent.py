import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

_batch_size = 64
_buffer_size = int(1e5)
_lr_actor = 0.001
_lr_critic = 0.001
_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed).to(_device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(_device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=_lr_actor)

        # Critic Network
        self.critic_local = Critic(state_size, action_size, random_seed).to(_device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(_device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=_lr_critic)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(random_seed)
    
    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(_device)
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).data.numpy()
        self.actor_local.train()
        
        return action
    
    def step(self, states, actions, rewards, next_states):
        """Save experience in replay buffer"""
        # Save experience
        self.replay_buffer.add(states, actions, rewards, next_states)
    
    def learn(self, experiences, gamma):
        # TODO
        pass


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            seed (int): random seed
        """
        self.memory = deque(maxlen=_buffer_size)
        self.experience = namedtuple('Experience', field_names=['states', 'actions', 'rewards', 'next_states'])
        self.seed = random.seed(seed)
    
    def add(self, states, actions, rewards, next_states):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=_batch_size)
        
        states = torch.from_numpy(np.vstack([e.states for e in experiences if e is not None])).float().to(_device)
        actions = torch.from_numpy(np.vstack([e.actions for e in experiences if e is not None])).long().to(_device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(_device)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None])).float().to(_device)
        
        return (states, actions, rewards, next_states)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
