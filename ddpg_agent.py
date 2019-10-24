import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

_batch_size = 64
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

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(_device)
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).data.numpy()
        self.actor_local.train()
        
        return action
    
    def learn(self, experiences, gamma):
        # TODO
        pass