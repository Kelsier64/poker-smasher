#!/usr/bin/env python3
"""
Poker Smasher - Poker Environment for Reinforcement Learning

A comprehensive Texas Hold'em poker environment for training RL agents.
"""

from poker_env import PokerEnv
from train import train_agents, play_single_game, RandomAgent, SimpleHeuristicAgent, QLearningAgent


def main():
    """Main entry point for the poker environment"""
    
    print("🃏 Welcome to Poker Smasher! 🃏")
    print("A Texas Hold'em poker environment for RL training\n")
    
    # Demo the environment
    print("1. Creating poker environment...")
    env = PokerEnv(num_players=3, initial_chips=1000, small_blind=10, big_blind=20)
    
    print("2. Testing environment reset...")
    observation, info = env.reset()
    print(f"   Observation shape: {observation.shape}")
    print(f"   Info: {info}")
    
    print("\n3. Rendering initial game state...")
    env.render()
    
    print("4. Creating different agent types...")
    agents = [
        QLearningAgent(env.observation_space.shape[0], env.action_space.n),
        SimpleHeuristicAgent(),
        RandomAgent()
    ]
    
    print("5. Playing a quick demonstration game...")
    play_single_game(agents, render=True)
    
    print("\n6. Starting training session...")
    print("   (Training 500 episodes with 3 different agent types)")
    
    trained_agents, scores = train_agents(num_episodes=500, render_every=100)
    
    print("\n✅ Demo complete!")
    print("\nTo train your own agents:")
    print("  python train.py")
    print("\nTo use the environment in your code:")
    print("  from poker_env import PokerEnv")
    print("  env = PokerEnv()")


if __name__ == "__main__":
    main()

