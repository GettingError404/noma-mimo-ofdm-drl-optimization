"""
experiments/train_drl.py

Training Script for TD3 Agent on NOMA-MIMO-OFDM Environment
"""

import numpy as np
import torch
import sys
import os
from datetime import datetime
from tqdm import tqdm
import json

sys.path.append('.')

from configs.system_params import get_scenario_config
from src.environment.noma_mimo_ofdm_env import NomaMimoOfdmEnv
from src.drl.td3_agent import TD3Agent


class TrainingLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir='results/training_logs'):
        """Initialize logger."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_{timestamp}.json')
        
        self.episode_rewards = []
        self.episode_se = []
        self.episode_ee = []
        self.actor_losses = []
        self.critic_losses = []
    
    def log_episode(self, episode, reward, se, ee):
        """Log episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_se.append(se)
        self.episode_ee.append(ee)
    
    def log_update(self, actor_loss, critic_loss):
        """Log update losses."""
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
        if critic_loss is not None:
            self.critic_losses.append(critic_loss)
    
    def save(self):
        """Save logs to file."""
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_se': self.episode_se,
            'episode_ee': self.episode_ee,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Training logs saved to {self.log_file}")
    
    def print_stats(self, window=100):
        """Print recent statistics."""
        if len(self.episode_rewards) < window:
            return
        
        recent_rewards = self.episode_rewards[-window:]
        recent_se = self.episode_se[-window:]
        recent_ee = self.episode_ee[-window:]
        
        print(f"\n📊 Last {window} episodes:")
        print(f"   Avg Reward: {np.mean(recent_rewards):.4f} ± {np.std(recent_rewards):.4f}")
        print(f"   Avg SE: {np.mean(recent_se):.4f} ± {np.std(recent_se):.4f} bps/Hz")
        print(f"   Avg EE: {np.mean(recent_ee):.4e} ± {np.std(recent_ee):.4e} bits/J")


def train_td3(
    env,
    agent,
    num_episodes=1000,
    max_steps=100,
    batch_size=256,
    start_steps=1000,
    update_freq=1,
    eval_freq=50,
    save_freq=100,
    save_dir='results/models',
    log_dir='results/training_logs'
):
    """
    Train TD3 agent.
    
    Args:
        env: Gymnasium environment
        agent: TD3 agent
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        batch_size: Batch size for updates
        start_steps: Random exploration steps before training
        update_freq: Steps between network updates
        eval_freq: Episodes between evaluation
        save_freq: Episodes between saving checkpoints
        save_dir: Directory to save models
        log_dir: Directory to save logs
    """
    os.makedirs(save_dir, exist_ok=True)
    logger = TrainingLogger(log_dir)
    
    print("="*70)
    print("Training TD3 Agent on NOMA-MIMO-OFDM Environment")
    print("="*70)
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {agent.device}")
    print("="*70)
    
    total_steps = 0
    best_reward = -np.inf
    
    # Training loop
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_se = 0
        episode_ee = 0
        episode_steps = 0
        
        for step in range(max_steps):
            # Select action
            if total_steps < start_steps:
                # Random exploration
                action = env.action_space.sample()
            else:
                # Policy with exploration noise
                action = agent.select_action(state, noise_scale=0.1)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            
            # Update metrics
            episode_reward += reward
            episode_se += info['SE']
            episode_ee += info['EE']
            episode_steps += 1
            total_steps += 1
            
            # Update networks
            if total_steps >= start_steps and total_steps % update_freq == 0:
                actor_loss, critic_loss = agent.update(batch_size)
                logger.log_update(actor_loss, critic_loss)
            
            state = next_state
            
            if done:
                break
        
        # Average metrics over episode
        episode_se /= episode_steps
        episode_ee /= episode_steps
        
        # Log episode
        logger.log_episode(episode, episode_reward, episode_se, episode_ee)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} | "
                  f"Steps: {episode_steps} | "
                  f"Reward: {episode_reward:.4f} | "
                  f"SE: {episode_se:.4f} | "
                  f"EE: {episode_ee:.4e}")
        
        # Evaluation
        if episode % eval_freq == 0:
            logger.print_stats(window=eval_freq)
        
        # Save checkpoint
        if episode % save_freq == 0:
            checkpoint_path = os.path.join(save_dir, f'td3_ep{episode}.pth')
            agent.save(checkpoint_path)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = os.path.join(save_dir, 'td3_best.pth')
            agent.save(best_path)
    
    # Save final model
    final_path = os.path.join(save_dir, 'td3_final.pth')
    agent.save(final_path)
    
    # Save training logs
    logger.save()
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print(f"Best reward: {best_reward:.4f}")
    print(f"Total steps: {total_steps}")
    print("="*70)


def main():
    """Main training function."""
    # Configuration
    SCENARIO = 'small_cell'  # or 'macro_cell'
    NUM_EPISODES = 1000
    SEED = 42
    
    # Set random seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Create environment
    print("Creating environment...")
    env = NomaMimoOfdmEnv(scenario=SCENARIO, seed=SEED)
    
    print(f"✅ Environment created:")
    print(f"   State dim: {env.observation_space.shape[0]}")
    print(f"   Action dim: {env.action_space.shape[0]}")
    
    # Create agent
    print("\nCreating TD3 agent...")
    agent = TD3Agent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2
    )
    
    print(f"✅ Agent created:")
    print(f"   Device: {agent.device}")
    
    # Train
    print("\nStarting training...")
    train_td3(
        env=env,
        agent=agent,
        num_episodes=NUM_EPISODES,
        max_steps=100,
        batch_size=256,
        start_steps=1000,
        update_freq=1,
        eval_freq=50,
        save_freq=100
    )


if __name__ == "__main__":
    main()