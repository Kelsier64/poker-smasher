"""
Examples of using the poker environment for different scenarios
"""

import numpy as np
from poker_env import PokerEnv
from train import RandomAgent, SimpleHeuristicAgent, QLearningAgent


def example_basic_usage():
    """Basic environment usage example"""
    print("=== Basic Environment Usage ===")
    
    # Create environment
    env = PokerEnv(num_players=3, initial_chips=1000)
    
    # Reset and get initial observation
    observation, info = env.reset()
    print(f"Observation shape: {observation.shape}")
    print(f"Initial info: {info}")
    
    # Take a few random actions
    for step in range(5):
        action = np.random.choice([0, 1, 2])  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step + 1}: Action {action}, Reward {reward:.3f}")
        if terminated or truncated:
            print("Game ended")
            break
    
    print()


def example_agent_comparison():
    """Compare different agent types"""
    print("=== Agent Comparison ===")
    
    agents = [
        ('Random', RandomAgent()),
        ('Heuristic', SimpleHeuristicAgent()),
        ('Q-Learning', QLearningAgent(120, 3, epsilon=0.1))
    ]
    
    # Play multiple games
    results = {name: [] for name, _ in agents}
    
    for game in range(10):
        env = PokerEnv(num_players=len(agents))
        observation, info = env.reset()
        
        game_rewards = {name: 0 for name, _ in agents}
        
        while True:
            current_player = env.current_player
            if current_player >= len(agents):
                break
                
            agent_name, agent = agents[current_player]
            action = agent.get_action(observation, [0, 1, 2])
            
            obs, reward, terminated, truncated, info = env.step(action)
            game_rewards[agent_name] += reward
            
            if terminated or truncated:
                break
            
            observation = obs
        
        for name in game_rewards:
            results[name].append(game_rewards[name])
    
    # Print results
    for name, rewards in results.items():
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print(f"{name:12s}: {avg_reward:6.3f} ± {std_reward:5.3f}")
    
    print()


def example_custom_agent():
    """Example of creating a custom agent"""
    print("=== Custom Agent Example ===")
    
    class ConservativeAgent:
        """Always folds unless very strong hand"""
        
        def get_action(self, observation, legal_actions):
            # Extract hole cards from observation
            hole_cards = observation[:52]
            card_indices = np.where(hole_cards == 1.0)[0]
            
            if len(card_indices) >= 2:
                # Check if we have a pair
                rank1 = (card_indices[0] % 13) + 2
                rank2 = (card_indices[1] % 13) + 2
                
                # Only play if we have a pair of 10s or better
                if rank1 == rank2 and rank1 >= 10:
                    return 1 if 1 in legal_actions else 2  # Call or raise
            
            return 0 if 0 in legal_actions else 1  # Fold or call
        
        def update(self, experience):
            pass  # No learning
    
    # Test the conservative agent
    env = PokerEnv(num_players=2)
    agents = [ConservativeAgent(), RandomAgent()]
    
    observation, info = env.reset()
    
    print("Playing Conservative vs Random agent...")
    step = 0
    while step < 20:  # Limit steps for demo
        current_player = env.current_player
        if current_player >= len(agents):
            break
            
        agent = agents[current_player]
        action = agent.get_action(observation, [0, 1, 2])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        action_names = {0: "FOLD", 1: "CALL", 2: "RAISE"}
        print(f"Player {current_player} ({type(agent).__name__}): {action_names[action]}")
        
        if terminated or truncated:
            print("Game ended")
            break
        
        observation = obs
        step += 1
    
    print()


def example_environment_configuration():
    """Show different environment configurations"""
    print("=== Environment Configuration Examples ===")
    
    # High stakes game
    high_stakes = PokerEnv(
        num_players=6,
        initial_chips=10000,
        small_blind=100,
        big_blind=200,
        max_raises=5
    )
    
    # Heads-up tournament
    heads_up = PokerEnv(
        num_players=2,
        initial_chips=1500,
        small_blind=25,
        big_blind=50,
        max_raises=3
    )
    
    # Short-handed cash game
    short_handed = PokerEnv(
        num_players=4,
        initial_chips=2000,
        small_blind=5,
        big_blind=10,
        max_raises=4
    )
    
    configs = [
        ("High Stakes", high_stakes),
        ("Heads-Up", heads_up),
        ("Short-Handed", short_handed)
    ]
    
    for name, env in configs:
        obs, info = env.reset()
        print(f"{name:15s}: {env.num_players} players, ${env.initial_chips} stacks, ${env.small_blind}/${env.big_blind} blinds")
    
    print()


def example_observation_analysis():
    """Analyze the observation space"""
    print("=== Observation Space Analysis ===")
    
    env = PokerEnv(num_players=3)
    observation, info = env.reset()
    
    print(f"Total observation size: {len(observation)}")
    
    # Break down observation components
    components = [
        ("Hole cards", 52),
        ("Community cards", 52),
        ("Pot size", 1),
        ("Player stacks", env.num_players),
        ("Current bets", env.num_players),
        ("Position", 1),
        ("Betting round", 1),
        ("Active players", 1),
        ("Amount to call", 1)
    ]
    
    idx = 0
    for name, size in components:
        section = observation[idx:idx + size]
        if size == 1:
            print(f"{name:20s}: {section[0]:.3f}")
        else:
            non_zero = np.count_nonzero(section)
            print(f"{name:20s}: {non_zero}/{size} non-zero values")
        idx += size
    
    print()


def main():
    """Run all examples"""
    print("🃏 Poker Environment Examples 🃏\n")
    
    example_basic_usage()
    example_agent_comparison()
    example_custom_agent()
    example_environment_configuration()
    example_observation_analysis()
    
    print("All examples completed! 🎉")


if __name__ == "__main__":
    main()
