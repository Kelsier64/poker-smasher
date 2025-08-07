"""
Transformer Self-Play Training for Poker Agents

This script implements transformer-based self-play training where multiple
transformer agents learn by playing against each other.
"""

import numpy as np
import random
import os
import time
import json
from typing import List, Dict, Any, Optional
from poker_env import PokerEnv, Action
from transformer_agent import TransformerAgent, TransformerConfig
import matplotlib.pyplot as plt

# Fallback imports for when torch is not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not found. Please install with: pip install torch")


class TransformerSelfPlayTrainer:
    """Trainer for transformer-based self-play"""
    
    def __init__(self, num_players: int = 3, config: TransformerConfig = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for transformer training")
        
        self.num_players = num_players
        self.config = config or TransformerConfig()
        
        # Environment
        self.env = PokerEnv(
            num_players=num_players, 
            initial_chips=1000, 
            small_blind=10, 
            big_blind=20
        )
        
        # Agents
        self.agents = []
        for i in range(num_players):
            agent = TransformerAgent(self.config)
            self.agents.append(agent)
        
        # Training metrics
        self.episode_rewards = {i: [] for i in range(num_players)}
        self.win_rates = {i: [] for i in range(num_players)}
        self.training_stats = []
        
        # Directories
        self.save_dir = "saved_transformer_agents"
        os.makedirs(self.save_dir, exist_ok=True)
    
    def get_legal_actions(self) -> List[int]:
        """Get legal actions for current player"""
        return [Action.FOLD.value, Action.CALL.value, Action.RAISE.value, Action.ALL_IN.value]
    
    def train(self, num_episodes: int = 5000, 
             render_every: int = 500, 
             save_every: int = 1000,
             evaluate_every: int = 500):
        """Main training loop"""
        
        print(f"Starting transformer self-play training with {self.num_players} agents...")
        print(f"Config: {self.config.__dict__}")
        
        best_performance = -float('inf')
        episode_wins = {i: 0 for i in range(self.num_players)}
        
        for episode in range(num_episodes):
            # Reset environment
            observation, info = self.env.reset()
            
            # Episode tracking
            episode_rewards = {i: 0 for i in range(self.num_players)}
            episode_steps = 0
            
            # Game loop
            while True:
                current_player = self.env.current_player
                if current_player >= len(self.agents):
                    break
                
                agent = self.agents[current_player]
                legal_actions = self.get_legal_actions()
                
                # Get action from agent
                action = agent.get_action(observation, legal_actions)
                
                # Step environment
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                
                # Update reward tracking
                episode_rewards[current_player] += reward
                
                # Update agent with experience
                experience = {
                    'observation': observation,
                    'action': action,
                    'reward': reward,
                    'next_observation': next_observation,
                    'done': terminated or truncated
                }
                agent.update(experience)
                
                observation = next_observation
                episode_steps += 1
                
                if terminated or truncated:
                    break
            
            # Track episode results
            for i, reward in episode_rewards.items():
                self.episode_rewards[i].append(reward)
            
            # Track wins (player with highest reward)
            winner = max(episode_rewards, key=episode_rewards.get)
            episode_wins[winner] += 1
            
            # Logging and evaluation
            if episode % render_every == 0:
                self._log_progress(episode, episode_rewards, episode_wins, render_every)
            
            if episode % evaluate_every == 0 and episode > 0:
                self._evaluate_agents(episode)
            
            # Save models periodically
            if episode % save_every == 0 and episode > 0:
                current_performance = np.mean([np.mean(self.episode_rewards[i][-100:]) 
                                             for i in range(self.num_players)])
                
                if current_performance > best_performance:
                    best_performance = current_performance
                    self._save_agents(episode, is_best=True)
                else:
                    self._save_agents(episode, is_best=False)
        
        # Final save
        self._save_agents(num_episodes, is_best=True)
        
        print("\n=== Training Complete ===")
        self._print_final_stats()
        
        return self.agents
    
    def _log_progress(self, episode: int, episode_rewards: Dict, episode_wins: Dict, window: int):
        """Log training progress"""
        print(f"\n=== Episode {episode} ===")
        print(f"Episode rewards: {episode_rewards}")
        
        # Recent average rewards
        recent_rewards = []
        for i in range(self.num_players):
            recent = self.episode_rewards[i][-window:] if len(self.episode_rewards[i]) >= window else self.episode_rewards[i]
            avg_reward = np.mean(recent) if recent else 0
            recent_rewards.append(avg_reward)
        print(f"Average rewards (last {window}): {[f'{r:.3f}' for r in recent_rewards]}")
        
        # Win rates
        total_games = sum(episode_wins.values())
        win_rates = [episode_wins[i] / total_games if total_games > 0 else 0 for i in range(self.num_players)]
        print(f"Win rates: {[f'{r:.3f}' for r in win_rates]}")
        
        # Training stats from first agent
        stats = self.agents[0].get_training_stats()
        print(f"Training stats (Agent 0):")
        print(f"  Steps: {stats['steps_done']}, Memory: {stats['memory_size']}")
        print(f"  Epsilon: {stats['epsilon']:.4f}, Avg Loss: {stats['avg_loss']:.6f}")
        print(f"  Avg Q-values: {stats['avg_q_values']:.3f}")
    
    def _evaluate_agents(self, episode: int):
        """Evaluate agents performance"""
        # Calculate win rates over recent episodes
        recent_window = min(500, episode)
        
        for i in range(self.num_players):
            if len(self.episode_rewards[i]) >= recent_window:
                recent_rewards = self.episode_rewards[i][-recent_window:]
                win_rate = len([r for r in recent_rewards if r > 0]) / len(recent_rewards)
                self.win_rates[i].append(win_rate)
        
        # Store training statistics
        stats = {
            'episode': episode,
            'agent_stats': [agent.get_training_stats() for agent in self.agents],
            'avg_rewards': [np.mean(self.episode_rewards[i][-100:]) if len(self.episode_rewards[i]) >= 100 
                          else np.mean(self.episode_rewards[i]) for i in range(self.num_players)]
        }
        self.training_stats.append(stats)
    
    def _save_agents(self, episode: int, is_best: bool = False):
        """Save trained agents"""
        suffix = "_best" if is_best else f"_episode_{episode}"
        
        for i, agent in enumerate(self.agents):
            filename = f"transformer_agent_{i}{suffix}.pt"
            filepath = os.path.join(self.save_dir, filename)
            agent.save(filepath)
        
        # Save training statistics
        stats_file = os.path.join(self.save_dir, f"training_stats{suffix}.json")
        with open(stats_file, 'w') as f:
            json.dump({
                'episode_rewards': self.episode_rewards,
                'win_rates': self.win_rates,
                'training_stats': self.training_stats,
                'config': self.config.__dict__
            }, f, indent=2)
        
        print(f"Agents and stats saved with suffix: {suffix}")
    
    def _print_final_stats(self):
        """Print final training statistics"""
        print("\nFinal Training Statistics:")
        
        for i in range(self.num_players):
            total_reward = sum(self.episode_rewards[i])
            avg_reward = np.mean(self.episode_rewards[i]) if self.episode_rewards[i] else 0
            final_stats = self.agents[i].get_training_stats()
            
            print(f"\nAgent {i}:")
            print(f"  Total reward: {total_reward:.3f}")
            print(f"  Average reward: {avg_reward:.3f}")
            print(f"  Training steps: {final_stats['steps_done']}")
            print(f"  Episodes: {final_stats['episode_count']}")
            print(f"  Final epsilon: {final_stats['epsilon']:.4f}")
            print(f"  Memory size: {final_stats['memory_size']}")
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training curves"""
        if not TORCH_AVAILABLE:
            print("Matplotlib plots require PyTorch environment")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        axes[0, 0].set_title('Episode Rewards')
        for i in range(self.num_players):
            # Moving average
            window = 100
            if len(self.episode_rewards[i]) >= window:
                rewards = np.array(self.episode_rewards[i])
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(moving_avg, label=f'Agent {i}')
        axes[0, 0].legend()
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Reward')
        
        # Plot win rates
        axes[0, 1].set_title('Win Rates')
        for i in range(self.num_players):
            if self.win_rates[i]:
                axes[0, 1].plot(self.win_rates[i], label=f'Agent {i}')
        axes[0, 1].legend()
        axes[0, 1].set_xlabel('Evaluation Period')
        axes[0, 1].set_ylabel('Win Rate')
        
        # Plot training loss (first agent)
        if self.agents[0].training_losses:
            axes[1, 0].set_title('Training Loss (Agent 0)')
            losses = self.agents[0].training_losses
            window = 50
            if len(losses) >= window:
                moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
                axes[1, 0].plot(moving_avg)
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
        
        # Plot epsilon decay
        axes[1, 1].set_title('Exploration (Epsilon)')
        if self.training_stats:
            episodes = [stat['episode'] for stat in self.training_stats]
            epsilons = [stat['agent_stats'][0]['epsilon'] for stat in self.training_stats]
            axes[1, 1].plot(episodes, epsilons)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training curves saved to {save_path}")
        else:
            plt.show()


def load_and_test_transformer_agents(agent_files: List[str], num_test_games: int = 100):
    """Load and test transformer agents"""
    if not TORCH_AVAILABLE:
        print("PyTorch is required for loading transformer agents")
        return
    
    print(f"Loading {len(agent_files)} transformer agents...")
    
    agents = []
    for i, filepath in enumerate(agent_files):
        if os.path.exists(filepath):
            try:
                agent = TransformerAgent.load(filepath)
                agents.append(agent)
            except Exception as e:
                print(f"Error loading agent from {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")
    
    if len(agents) < 2:
        print("Need at least 2 agents for testing")
        return
    
    # Test the agents
    env = PokerEnv(num_players=len(agents), initial_chips=1000, small_blind=10, big_blind=20)
    test_rewards = {i: [] for i in range(len(agents))}
    
    print(f"\nTesting agents over {num_test_games} games...")
    
    for game in range(num_test_games):
        observation, info = env.reset()
        game_rewards = {i: 0 for i in range(len(agents))}
        
        while True:
            current_player = env.current_player
            if current_player >= len(agents):
                break
            
            agent = agents[current_player]
            legal_actions = [Action.FOLD.value, Action.CALL.value, Action.RAISE.value, Action.ALL_IN.value]
            
            # Set agent to evaluation mode (no exploration)
            original_epsilon = agent.config.epsilon_start
            agent.config.epsilon_start = 0.0
            agent.config.epsilon_end = 0.0
            
            action = agent.get_action(observation, legal_actions)
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Restore original epsilon
            agent.config.epsilon_start = original_epsilon
            
            game_rewards[current_player] += reward
            
            if terminated or truncated:
                break
        
        for i, reward in game_rewards.items():
            test_rewards[i].append(reward)
        
        if game % 20 == 0:
            print(f"Game {game}/{num_test_games}")
    
    # Print results
    print("\n=== Test Results ===")
    for i, agent in enumerate(agents):
        avg_reward = np.mean(test_rewards[i])
        win_rate = len([r for r in test_rewards[i] if r > 0]) / len(test_rewards[i])
        print(f"Agent {i}: Avg reward = {avg_reward:.3f}, Win rate = {win_rate:.3f}")


def main():
    """Main training function"""
    if not TORCH_AVAILABLE:
        print("This script requires PyTorch. Please install with:")
        print("pip install torch")
        return
    
    # Check for existing agents
    save_dir = "saved_transformer_agents"
    agent_files = [
        os.path.join(save_dir, "transformer_agent_0_best.pt"),
        os.path.join(save_dir, "transformer_agent_1_best.pt"),
        os.path.join(save_dir, "transformer_agent_2_best.pt")
    ]
    
    agents_exist = all(os.path.exists(f) for f in agent_files)
    
    if agents_exist:
        print("Found existing transformer agents!")
        print("1. Load and test existing agents")
        print("2. Train new agents (will overwrite existing)")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            load_and_test_transformer_agents(agent_files)
            return
    
    # Training configuration
    config = TransformerConfig(
        state_dim=106,  # Poker environment observation size
        action_dim=4,   # Number of actions
        sequence_length=16,  # Shorter sequences for poker
        hidden_dim=128,  # Smaller model for faster training
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        learning_rate=3e-4,
        batch_size=32,
        gamma=0.99,
        epsilon_start=0.9,
        epsilon_end=0.05,
        epsilon_decay=2000,
        memory_size=10000,
        target_update=100
    )
    
    # Create trainer
    trainer = TransformerSelfPlayTrainer(num_players=3, config=config)
    
    # Train agents
    print("Starting transformer self-play training...")
    start_time = time.time()
    
    trained_agents = trainer.train(
        num_episodes=10000,
        render_every=1000,
        save_every=2000,
        evaluate_every=500
    )
    
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    
    # Plot training curves
    try:
        trainer.plot_training_curves(os.path.join(save_dir, "training_curves.png"))
    except Exception as e:
        print(f"Could not save plots: {e}")
    
    # Test the trained agents
    print("\nTesting final trained agents...")
    best_agent_files = [f.replace('.pt', '_best.pt') for f in agent_files]
    load_and_test_transformer_agents(best_agent_files, num_test_games=50)


if __name__ == "__main__":
    main()
