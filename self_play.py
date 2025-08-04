#!/usr/bin/env python3
"""
Self-play testing script for the poker environment.
Allows manual testing and watching AI agents play against each other.
"""

import numpy as np
from poker_env import PokerEnv, Action
from train import RandomAgent, SimpleHeuristicAgent, QLearningAgent


def get_human_action(env, player_idx):
    """Get action from human player via console input"""
    player = env.players[player_idx]
    to_call = max(0, env.current_bet - player.current_bet)
    
    print(f"\n--- Your turn (Player {player_idx}) ---")
    print(f"Your chips: ${player.chips}")
    print(f"Your current bet: ${player.current_bet}")
    print(f"Amount to call: ${to_call}")
    print(f"Pot: ${env.pot}")
    print(f"Your cards: {' '.join([str(card) for card in player.hole_cards])}")
    
    # Show valid actions
    print("\nAvailable actions:")
    print("0 - Fold")
    
    if to_call == 0:
        print("1 - Check")
    else:
        print(f"1 - Call (${to_call})")
    
    can_raise = (env.raises_this_round < env.max_raises and 
                 player.chips > to_call and 
                 to_call < player.chips)
    
    if can_raise:
        min_raise_total = to_call + env.min_raise
        if min_raise_total <= player.chips:
            print(f"2 - Raise (minimum ${env.min_raise} more)")
        else:
            print("2 - Raise (not available)")
    else:
        print("2 - Raise (not available)")
    
    if player.chips > 0:
        print(f"3 - All-in (${player.chips})")
    else:
        print("3 - All-in (not available)")
    
    # Get user input
    while True:
        try:
            action = int(input("\nEnter your action (0-3): "))
            if action in [0, 1, 2, 3]:
                if action == 2 and not can_raise:
                    print("Cannot raise in this situation. Choose 0 (fold), 1 (call/check), or 3 (all-in).")
                    continue
                if action == 3 and player.chips == 0:
                    print("Cannot go all-in with no chips. Choose 0 (fold) or 1 (call/check).")
                    continue
                return action
            else:
                print("Invalid action. Please enter 0, 1, 2, or 3.")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting game...")
            return 0  # Fold and exit


def play_human_vs_ai():
    """Play as a human against AI agents"""
    print("🃏 Human vs AI Poker Game 🃏")
    print("You are Player 0. The AI agents will play automatically.\n")
    
    # Create environment with human as player 0
    env = PokerEnv(num_players=4, initial_chips=1000, small_blind=10, big_blind=20)
    
    # Create AI agents for other players
    ai_agents = [
        SimpleHeuristicAgent(),
        RandomAgent(), 
        QLearningAgent(env.observation_space.shape[0], env.action_space.n, epsilon=0.2)
    ]
    
    observation, info = env.reset()
    env.render()
    
    step_count = 0
    max_steps = 100  # Prevent infinite loops
    
    while not env._is_terminal() and step_count < max_steps:
        current_player = env.current_player
        
        if current_player == 0:
            # Human player's turn
            action = get_human_action(env, current_player)
        else:
            # AI agent's turn
            ai_agent = ai_agents[current_player - 1]
            legal_actions = [0, 1, 2, 3]  # All actions are legal in our environment
            action = ai_agent.get_action(observation, legal_actions)
            
            player = env.players[current_player]
            action_name = ["Fold", "Call/Check", "Raise", "All-in"][action]
            print(f"\nPlayer {current_player} ({type(ai_agent).__name__}): {action_name}")
            
            if action == 1:  # Call/Check
                to_call = max(0, env.current_bet - player.current_bet)
                if to_call == 0:
                    print(f"  -> Checks")
                else:
                    print(f"  -> Calls ${to_call}")
            elif action == 2:  # Raise
                print(f"  -> Raises")
            elif action == 3:  # All-in
                print(f"  -> Goes all-in with ${player.chips + player.current_bet}")
        
        # Execute the action
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
            
        # Show game state after significant actions
        if env.betting_round != (step_count // 10):  # New betting round
            env.render()
        
        step_count += 1
    
    # Show final results
    print("\n" + "="*50)
    print("GAME OVER")
    print("="*50)
    env.render()
    
    # Show chip changes
    print("\nFinal chip counts:")
    for i, player in enumerate(env.players):
        change = player.chips - env.initial_chips
        change_str = f"+${change}" if change >= 0 else f"-${abs(change)}"
        player_type = "You" if i == 0 else type(ai_agents[i-1]).__name__
        print(f"  Player {i} ({player_type}): ${player.chips} ({change_str})")


def watch_ai_vs_ai():
    """Watch AI agents play against each other"""
    print("🤖 AI vs AI Poker Game 🤖")
    print("Watching AI agents play against each other...\n")
    
    # Create environment
    env = PokerEnv(num_players=4, initial_chips=1000, small_blind=10, big_blind=20)
    
    # Create different AI agents
    agents = [
        SimpleHeuristicAgent(),
        RandomAgent(),
        QLearningAgent(env.observation_space.shape[0], env.action_space.n, epsilon=0.3),
        SimpleHeuristicAgent()
    ]
    
    agent_names = [type(agent).__name__ for agent in agents]
    
    observation, info = env.reset()
    env.render()
    
    step_count = 0
    max_steps = 200
    
    while not env._is_terminal() and step_count < max_steps:
        current_player = env.current_player
        agent = agents[current_player]
        
        legal_actions = [0, 1, 2, 3]  # All actions are legal in our environment
        action = agent.get_action(observation, legal_actions)
        
        # Show what the agent is doing
        player = env.players[current_player]
        action_name = ["Fold", "Call/Check", "Raise", "All-in"][action]
        print(f"\nPlayer {current_player} ({agent_names[current_player]}): {action_name}")
        
        if action == 1:  # Call/Check
            to_call = max(0, env.current_bet - player.current_bet)
            if to_call == 0:
                print(f"  -> Checks")
            else:
                print(f"  -> Calls ${to_call}")
        elif action == 2:  # Raise
            print(f"  -> Raises")
        elif action == 3:  # All-in
            print(f"  -> Goes all-in")
        
        # Execute action
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
        
        # Pause between actions for readability
        input("Press Enter to continue...")
        
        step_count += 1
    
    # Show final results
    print("\n" + "="*50)
    print("GAME OVER")
    print("="*50)
    env.render()
    
    print("\nFinal chip counts:")
    for i, player in enumerate(env.players):
        change = player.chips - env.initial_chips
        change_str = f"+${change}" if change >= 0 else f"-${abs(change)}"
        print(f"  Player {i} ({agent_names[i]}): ${player.chips} ({change_str})")


def test_environment_basic():
    """Basic environment testing without manual input"""
    print("🧪 Basic Environment Test 🧪")
    print("Testing the environment with random actions...\n")
    
    env = PokerEnv(num_players=3, initial_chips=1000, small_blind=10, big_blind=20)
    observation, info = env.reset()
    
    print("Initial state:")
    env.render()
    
    step_count = 0
    max_steps = 50
    
    while not env._is_terminal() and step_count < max_steps:
        # Take random action
        action = np.random.choice([0, 1, 2, 3])
        
        print(f"\nStep {step_count + 1}: Player {env.current_player} takes action {action} ({['Fold', 'Call/Check', 'Raise', 'All-in'][action]})")
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print("Game terminated!")
            break
        
        step_count += 1
        
        # Show state every few steps
        if step_count % 5 == 0:
            print(f"\nState after {step_count} steps:")
            env.render()
    
    print("\nFinal state:")
    env.render()
    
    print(f"\nTest completed after {step_count} steps.")


def main():
    """Main menu for self-play testing"""
    print("🃏 Poker Environment Self-Play Testing 🃏")
    print("="*50)
    
    while True:
        print("\nChoose a testing mode:")
        print("1 - Play as human vs AI agents")
        print("2 - Watch AI agents play against each other")
        print("3 - Basic environment test (random actions)")
        print("4 - Exit")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                play_human_vs_ai()
            elif choice == "2":
                watch_ai_vs_ai()
            elif choice == "3":
                test_environment_basic()
            elif choice == "4":
                print("Thanks for testing! 🃏")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        print("\n" + "="*50)


if __name__ == "__main__":
    main()
