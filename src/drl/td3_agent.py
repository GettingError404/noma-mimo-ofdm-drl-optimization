"""
src/drl/td3_agent.py

Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent
For NOMA-MIMO-OFDM Resource Allocation

Reference: "Addressing Function Approximation Error in Actor-Critic Methods"
Fujimoto et al., ICML 2018
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, capacity=100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network: Maps states to actions."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[512, 256, 128]):
        """
        Initialize actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(Actor, self).__init__()
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())  # Actions in [-1, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass."""
        return self.network(state)


class Critic(nn.Module):
    """Critic network: Estimates Q-value Q(s,a)."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[512, 256]):
        """
        Initialize critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(Critic, self).__init__()
        
        # Q1 network
        self.q1_layers = nn.ModuleList()
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            self.q1_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        self.q1_output = nn.Linear(input_dim, 1)
        
        # Q2 network (twin)
        self.q2_layers = nn.ModuleList()
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            self.q2_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        self.q2_output = nn.Linear(input_dim, 1)
    
    def forward(self, state, action):
        """
        Forward pass through both Q networks.
        
        Returns:
            q1, q2: Q-values from both networks
        """
        # Concatenate state and action
        sa = torch.cat([state, action], dim=1)
        
        # Q1 forward
        q1 = sa
        for layer in self.q1_layers:
            q1 = F.relu(layer(q1))
        q1 = self.q1_output(q1)
        
        # Q2 forward
        q2 = sa
        for layer in self.q2_layers:
            q2 = F.relu(layer(q2))
        q2 = self.q2_output(q2)
        
        return q1, q2
    
    def q1_forward(self, state, action):
        """Forward pass through Q1 only (for actor update)."""
        sa = torch.cat([state, action], dim=1)
        
        q1 = sa
        for layer in self.q1_layers:
            q1 = F.relu(layer(q1))
        q1 = self.q1_output(q1)
        
        return q1


class TD3Agent:
    """
    Twin Delayed DDPG Agent
    
    Key features:
    - Twin Q-networks to reduce overestimation
    - Delayed policy updates
    - Target policy smoothing
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize TD3 agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            tau: Soft update coefficient
            policy_noise: Std of noise added to target policy
            noise_clip: Clip range for policy noise
            policy_delay: Delay between policy updates
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # Training step counter
        self.total_steps = 0
    
    def select_action(self, state, noise_scale=0.1):
        """
        Select action using current policy with exploration noise.
        
        Args:
            state: Current state
            noise_scale: Scale of exploration noise
            
        Returns:
            action: Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        # Add exploration noise
        noise = np.random.normal(0, noise_scale, size=action.shape)
        action = np.clip(action + noise, -1, 1)
        
        return action
    
    def update(self, batch_size=256):
        """
        Update actor and critic networks.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            actor_loss, critic_loss: Loss values
        """
        if len(self.replay_buffer) < batch_size:
            return None, None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # ============ UPDATE CRITIC ============
        with torch.no_grad():
            # Target policy smoothing: add noise to target action
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)
            
            # Compute target Q-values (take minimum of twin Q-networks)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # TD target
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ============ DELAYED POLICY UPDATE ============
        actor_loss = None
        if self.total_steps % self.policy_delay == 0:
            # Actor loss: maximize Q1(s, actor(s))
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            
            actor_loss = actor_loss.item()
        
        self.total_steps += 1
        
        return actor_loss, critic_loss.item()
    
    def _soft_update(self, source, target):
        """
        Soft update target network: θ_target = τ*θ + (1-τ)*θ_target
        
        Args:
            source: Source network
            target: Target network
        """
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, filepath):
        """Save model parameters."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"✅ Model loaded from {filepath}")


def test_td3_agent():
    """Test TD3 agent."""
    print("="*70)
    print("Testing TD3 Agent")
    print("="*70)
    
    # Create agent
    state_dim = 41  # Example for K=10 users
    action_dim = 20  # Power + pairing for K=10
    
    agent = TD3Agent(state_dim, action_dim)
    
    print(f"\n✅ Agent Created:")
    print(f"   State dim: {state_dim}")
    print(f"   Action dim: {action_dim}")
    print(f"   Device: {agent.device}")
    
    # Test action selection
    state = np.random.randn(state_dim)
    action = agent.select_action(state)
    
    print(f"\n✅ Action Selection:")
    print(f"   Input state shape: {state.shape}")
    print(f"   Output action shape: {action.shape}")
    print(f"   Action range: [{action.min():.3f}, {action.max():.3f}]")
    
    # Test replay buffer
    for _ in range(300):
        s = np.random.randn(state_dim)
        a = np.random.randn(action_dim)
        r = np.random.randn()
        s_next = np.random.randn(state_dim)
        d = 0
        agent.replay_buffer.push(s, a, r, s_next, d)
    
    print(f"\n✅ Replay Buffer:")
    print(f"   Buffer size: {len(agent.replay_buffer)}")
    
    # Test update
    actor_loss, critic_loss = agent.update(batch_size=256)
    
    print(f"\n✅ Network Update:")
    print(f"   Actor loss: {actor_loss}")
    print(f"   Critic loss: {critic_loss:.6f}")
    
    # Test save/load
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_model.pth')
        agent.save(filepath)
        agent.load(filepath)
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_td3_agent()