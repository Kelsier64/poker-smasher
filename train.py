"""
Training script for poker RL agents

This script demonstrates how to train a simple RL agent on the poker environment.
Includes random agent, simple heuristic agent, and placeholder for advanced RL algorithms.
"""

import numpy as np
import random
import pickle
import os
from typing import List, Tuple, Dict, Any
from poker_env import PokerEnv, Action
from abc import ABC, abstractmethod


class PokerAgent(ABC):
    """Abstract base class for poker agents"""
    
    @abstractmethod
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        """Get action given observation and legal actions"""
        pass
    
    @abstractmethod
    def update(self, experience: Dict[str, Any]) -> None:
        """Update agent with experience"""
        pass


class RandomAgent(PokerAgent):
    """Random poker agent for baseline"""
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        return random.choice(legal_actions)
    
    def update(self, experience: Dict[str, Any]) -> None:
        pass  # Random agent doesn't learn


class SimpleHeuristicAgent(PokerAgent):
    """Simple heuristic poker agent based on hand strength and pot odds"""
    
    def __init__(self):
        self.hand_rankings = self._create_hand_rankings()
    
    def _create_hand_rankings(self) -> Dict[Tuple[int, int], float]:
        """Create preflop hand strength rankings (simplified)"""
        rankings = {}
        
        # Premium hands
        premium = [(14, 14), (13, 13), (12, 12), (11, 11), (10, 10)]
        for hand in premium:
            rankings[hand] = 0.9
            rankings[hand[::-1]] = 0.9
        
        # Strong hands
        strong = [(14, 13), (14, 12), (14, 11), (13, 12), (13, 11), (12, 11)]
        for hand in strong:
            rankings[hand] = 0.8
            rankings[hand[::-1]] = 0.8
        
        # Medium pairs
        for rank in range(9, 6, -1):
            rankings[(rank, rank)] = 0.7
        
        # Suited connectors and medium hands
        medium_suited = [(14, 10), (13, 10), (12, 10), (11, 10), (10, 9)]
        for hand in medium_suited:
            rankings[hand] = 0.6
        
        # Small pairs
        for rank in range(6, 1, -1):
            rankings[(rank, rank)] = 0.5
        
        return rankings
    
    def _extract_hole_cards(self, observation: np.ndarray) -> Tuple[int, int]:
        """Extract hole card ranks from observation"""
        # First 52 elements are hole cards (one-hot encoded)
        hole_cards = observation[:52]
        card_indices = np.where(hole_cards == 1.0)[0]
        
        if len(card_indices) >= 2:
            # Convert indices back to ranks
            rank1 = (card_indices[0] % 13) + 2
            rank2 = (card_indices[1] % 13) + 2
            return (max(rank1, rank2), min(rank1, rank2))
        
        return (2, 2)  # Default if can't extract
    
    def _get_hand_strength(self, observation: np.ndarray) -> float:
        """Get hand strength based on hole cards"""
        hole_cards = self._extract_hole_cards(observation)
        return self.hand_rankings.get(hole_cards, 0.3)  # Default weak hand
    
    def _get_pot_odds(self, observation: np.ndarray) -> float:
        """Calculate pot odds from observation"""
        # Extract pot size and amount to call from observation
        pot_size = observation[104] * 6000  # Denormalize (assuming 6 players * 1000 chips)
        amount_to_call = observation[-1] * 1000  # Denormalize
        
        if amount_to_call == 0:
            return float('inf')  # Free to call
        
        return pot_size / amount_to_call
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        hand_strength = self._get_hand_strength(observation)
        pot_odds = self._get_pot_odds(observation)
        
        # Decision logic based on hand strength and pot odds
        if hand_strength >= 0.9:
            # Premium hand - consider all-in
            if Action.ALL_IN.value in legal_actions and random.random() < 0.3:
                return Action.ALL_IN.value
            elif Action.RAISE.value in legal_actions:
                return Action.RAISE.value
            else:
                return Action.CALL.value
        elif hand_strength >= 0.8:
            # Strong hand - raise or call
            if Action.RAISE.value in legal_actions:
                return Action.RAISE.value
            else:
                return Action.CALL.value
        
        elif hand_strength >= 0.6:
            # Medium hand - call if good pot odds
            if pot_odds > 3.0 and Action.CALL.value in legal_actions:
                return Action.CALL.value
            elif Action.RAISE.value in legal_actions and random.random() < 0.3:
                return Action.RAISE.value  # Occasional bluff
            else:
                return Action.FOLD.value if Action.FOLD.value in legal_actions else Action.CALL.value
        
        else:
            # Weak hand - mostly fold unless great pot odds
            if pot_odds > 5.0 and Action.CALL.value in legal_actions:
                return Action.CALL.value
            else:
                return Action.FOLD.value if Action.FOLD.value in legal_actions else Action.CALL.value
    
    def update(self, experience: Dict[str, Any]) -> None:
        pass  # Heuristic agent doesn't learn


class QLearningAgent(PokerAgent):
    """Simple Q-Learning agent for poker (placeholder for advanced RL)"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1, 
                 epsilon: float = 0.1, discount: float = 0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount = discount
        
        # Simple tabular Q-learning (would need function approximation for real use)
        self.q_table = {}
        self.last_state = None
        self.last_action = None
    
    def _state_to_key(self, observation: np.ndarray) -> str:
        """Convert observation to discrete state key (simplified)"""
        # This is a very simplified state representation
        # In practice, you'd want more sophisticated state abstraction
        
        # Extract key features
        hand_strength = self._get_simplified_hand_strength(observation)
        pot_odds = min(int(observation[-1] * 10), 10)  # Discretize pot odds
        betting_round = int(observation[-4] * 3)
        
        return f"{hand_strength}_{pot_odds}_{betting_round}"
    
    def _get_simplified_hand_strength(self, observation: np.ndarray) -> int:
        """Get simplified hand strength category"""
        # Extract hole cards (very simplified)
        hole_cards = observation[:52]
        num_cards = int(np.sum(hole_cards))
        
        if num_cards >= 2:
            return min(int(np.sum(hole_cards[:26]) * 5), 4)  # 0-4 strength categories
        return 0
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        state_key = self._state_to_key(observation)
        
        # Initialize Q-values if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in range(self.action_size)}
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action = random.choice(legal_actions)
        else:
            # Choose best legal action
            legal_q_values = {action: self.q_table[state_key][action] for action in legal_actions}
            action = max(legal_q_values, key=legal_q_values.get)
        
        # Store for learning
        self.last_state = state_key
        self.last_action = action
        
        return action
    
    def update(self, experience: Dict[str, Any]) -> None:
        """Update Q-values based on experience"""
        if self.last_state is None or self.last_action is None:
            return
        
        reward = experience.get('reward', 0)
        next_observation = experience.get('next_observation')
        done = experience.get('done', False)
        
        current_q = self.q_table[self.last_state][self.last_action]
        
        if done or next_observation is None:
            target = reward
        else:
            next_state_key = self._state_to_key(next_observation)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = {action: 0.0 for action in range(self.action_size)}
            
            next_max_q = max(self.q_table[next_state_key].values())
            target = reward + self.discount * next_max_q
        
        # Q-learning update
        self.q_table[self.last_state][self.last_action] = current_q + self.learning_rate * (target - current_q)
    
    def save(self, filepath: str) -> None:
        """Save the Q-learning agent to a file"""
        agent_data = {
            'q_table': self.q_table,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'discount': self.discount
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        print(f"Agent saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'QLearningAgent':
        """Load a Q-learning agent from a file"""
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        # Create agent with loaded parameters
        agent = cls(
            state_size=agent_data['state_size'],
            action_size=agent_data['action_size'],
            learning_rate=agent_data['learning_rate'],
            epsilon=agent_data['epsilon'],
            discount=agent_data['discount']
        )
        
        # Load the Q-table
        agent.q_table = agent_data['q_table']
        print(f"Agent loaded from {filepath}, Q-table size: {len(agent.q_table)}")
        
        return agent


def get_legal_actions(env: PokerEnv) -> List[int]:
    """Get legal actions for current player"""
    # For simplicity, assume all actions are always legal
    # In practice, you might want to check if player can actually raise, etc.
    return [Action.FOLD.value, Action.CALL.value, Action.RAISE.value, Action.ALL_IN.value]


def train_self_play(num_episodes: int = 2000, render_every: int = 200, num_players: int = 3):
    """Train RL agents through self-play"""
    
    env = PokerEnv(num_players=num_players, initial_chips=1000, small_blind=10, big_blind=20)
    
    # Create multiple RL agents for self-play
    # Start with higher exploration, then decay epsilon over time
    agents = []
    for i in range(num_players):
        agent = QLearningAgent(
            env.observation_space.shape[0], 
            env.action_space.n, 
            learning_rate=0.1,
            epsilon=0.3,  # Start with higher exploration
            discount=0.95
        )
        agents.append(agent)
    
    scores = {i: [] for i in range(len(agents))}
    epsilon_decay = 0.995  # Decay exploration over time
    min_epsilon = 0.05
    
    print(f"Starting self-play training with {num_players} RL agents...")
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        
        episode_rewards = {i: 0 for i in range(len(agents))}
        episode_steps = 0
        
        while True:
            current_player = env.current_player
            if current_player >= len(agents):
                break
                
            agent = agents[current_player]
            legal_actions = get_legal_actions(env)
            
            action = agent.get_action(observation, legal_actions)
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards[current_player] += reward
            
            # Update the agent that just acted
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
        
        # Record scores
        for i, reward in episode_rewards.items():
            scores[i].append(reward)
        
        # Decay epsilon for all agents
        for agent in agents:
            agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)
        
        # Render occasionally
        if episode % render_every == 0:
            print(f"\nEpisode {episode}")
            print(f"Episode rewards: {episode_rewards}")
            print(f"Average scores (last 100): {[np.mean(scores[i][-100:]) if len(scores[i]) >= 100 else np.mean(scores[i]) for i in range(len(agents))]}")
            print(f"Current epsilon: {agents[0].epsilon:.4f}")
            print(f"Q-table sizes: {[len(agent.q_table) for agent in agents]}")
            
            if episode % (render_every * 2) == 0 and episode > 0:
                env.render()
    
    # Final results
    print("\n=== Self-Play Training Complete ===")
    for i, agent in enumerate(agents):
        avg_score = np.mean(scores[i]) if scores[i] else 0
        print(f"Agent {i}: Average score = {avg_score:.3f}, Q-table size = {len(agent.q_table)}")
    
    # Save the trained agents
    save_dir = "saved_agents"
    os.makedirs(save_dir, exist_ok=True)
    
    for i, agent in enumerate(agents):
        save_path = os.path.join(save_dir, f"self_play_agent_{i}.pkl")
        agent.save(save_path)
    
    return agents, scores


def train_agents(num_episodes: int = 1000, render_every: int = 100):
    """Train multiple agents against each other"""
    
    env = PokerEnv(num_players=3, initial_chips=1000, small_blind=10, big_blind=20)
    
    # Create different types of agents
    agents = [
        QLearningAgent(env.observation_space.shape[0], env.action_space.n, epsilon=0.2),
        SimpleHeuristicAgent(),
        RandomAgent()
    ]
    
    scores = {i: [] for i in range(len(agents))}
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        
        episode_rewards = {i: 0 for i in range(len(agents))}
        episode_steps = 0
        
        while True:
            current_player = env.current_player
            if current_player >= len(agents):
                break
                
            agent = agents[current_player]
            legal_actions = get_legal_actions(env)
            
            action = agent.get_action(observation, legal_actions)
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards[current_player] += reward
            
            # Update learning agents
            if hasattr(agent, 'update'):
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
        
        # Record scores
        for i, reward in episode_rewards.items():
            scores[i].append(reward)
        
        # Render occasionally
        if episode % render_every == 0:
            print(f"\nEpisode {episode}")
            print(f"Episode rewards: {episode_rewards}")
            print(f"Average scores (last 100): {[np.mean(scores[i][-100:]) for i in range(len(agents))]}")
            if episode % (render_every * 5) == 0:
                env.render()
    
    # Final results
    print("\n=== Training Complete ===")
    for i, agent in enumerate(agents):
        avg_score = np.mean(scores[i])
        agent_type = type(agent).__name__
        print(f"Agent {i} ({agent_type}): Average score = {avg_score:.3f}")
    
    return agents, scores


def play_single_game(agents: List[PokerAgent], render: bool = True):
    """Play a single game with given agents"""
    
    env = PokerEnv(num_players=len(agents), initial_chips=1000, small_blind=10, big_blind=20)
    observation, info = env.reset()
    
    if render:
        print("=== Starting New Game ===")
        env.render()
    
    step = 0
    while True:
        current_player = env.current_player
        if current_player >= len(agents):
            break
            
        agent = agents[current_player]
        legal_actions = get_legal_actions(env)
        
        action = agent.get_action(observation, legal_actions)
        
        if render:
            action_names = {0: "FOLD", 1: "CALL", 2: "RAISE", 3: "ALL_IN"}
            print(f"Player {current_player} chooses: {action_names[action]}")
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        if render:
            env.render()
        
        step += 1
        
        if terminated or truncated:
            break
    
    if render:
        print("=== Game Complete ===")
    
    return info


def load_and_test_agents(agent_files: List[str]):
    """Load saved agents and test them against each other"""
    print(f"Loading {len(agent_files)} saved agents...")
    
    loaded_agents = []
    for i, filepath in enumerate(agent_files):
        if os.path.exists(filepath):
            try:
                agent = QLearningAgent.load(filepath)
                loaded_agents.append(agent)
            except Exception as e:
                print(f"Error loading agent from {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")
    
    if len(loaded_agents) < 2:
        print("Need at least 2 agents to test. Creating additional agents...")
        while len(loaded_agents) < 3:
            loaded_agents.append(RandomAgent())
    
    # Test the loaded agents
    print("\nTesting loaded agents...")
    evaluation_scores = {i: [] for i in range(len(loaded_agents))}
    env = PokerEnv(num_players=len(loaded_agents), initial_chips=1000, small_blind=10, big_blind=20)
    
    for episode in range(100):  # Test games
        observation, info = env.reset()
        episode_rewards = {i: 0 for i in range(len(loaded_agents))}
        
        while True:
            current_player = env.current_player
            if current_player >= len(loaded_agents):
                break
                
            agent = loaded_agents[current_player]
            legal_actions = [Action.FOLD.value, Action.CALL.value, Action.RAISE.value, Action.ALL_IN.value]
            
            action = agent.get_action(observation, legal_actions)
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards[current_player] += reward
            
            if terminated or truncated:
                break
        
        for i, reward in episode_rewards.items():
            evaluation_scores[i].append(reward)
    
    print("\n=== Loaded Agents Test Results ===")
    for i, agent in enumerate(loaded_agents):
        agent_type = type(agent).__name__
        avg_score = np.mean(evaluation_scores[i])
        q_table_size = len(agent.q_table) if hasattr(agent, 'q_table') else 0
        print(f"Agent {i} ({agent_type}): Average score = {avg_score:.3f}, Q-table size = {q_table_size}")
    
    return loaded_agents


if __name__ == "__main__":
    # Check if saved agents exist
    save_dir = "saved_agents"
    saved_agent_files = [
        os.path.join(save_dir, "self_play_agent_0.pkl"),
        os.path.join(save_dir, "self_play_agent_1.pkl"),
        os.path.join(save_dir, "self_play_agent_2.pkl")
    ]
    
    agents_exist = all(os.path.exists(f) for f in saved_agent_files)
    
    if agents_exist:
        print("Found saved agents! Choose an option:")
        print("1. Load and test existing agents")
        print("2. Train new agents (will overwrite existing ones)")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            print("Loading and testing existing agents...")
            loaded_agents = load_and_test_agents(saved_agent_files)
            
            print("\nPlaying demonstration game with loaded agents...")
            play_single_game(loaded_agents[:3], render=True)
            exit()
    
    print("Training poker agents through self-play...")
    
    # Train agents through self-play
    trained_agents, training_scores = train_self_play(num_episodes=20000, render_every=2000)
    
    print("\nNow testing against different agent types...")
    
    # Test the best trained agent against other types
    best_agent = trained_agents[0]  # Take the first trained agent
    test_agents = [
        best_agent,
        SimpleHeuristicAgent(),
        RandomAgent()
    ]
    
    # Run a shorter evaluation
    evaluation_scores = {i: [] for i in range(len(test_agents))}
    env = PokerEnv(num_players=3, initial_chips=1000, small_blind=10, big_blind=20)
    
    print("\nEvaluating trained agent against other strategies...")
    for episode in range(100):  # Evaluation games
        observation, info = env.reset()
        episode_rewards = {i: 0 for i in range(len(test_agents))}
        
        while True:
            current_player = env.current_player
            if current_player >= len(test_agents):
                break
                
            agent = test_agents[current_player]
            legal_actions = [Action.FOLD.value, Action.CALL.value, Action.RAISE.value, Action.ALL_IN.value]
            
            action = agent.get_action(observation, legal_actions)
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards[current_player] += reward
            
            if terminated or truncated:
                break
        
        for i, reward in episode_rewards.items():
            evaluation_scores[i].append(reward)
    
    print("\n=== Evaluation Results ===")
    agent_names = ["Trained RL Agent", "Heuristic Agent", "Random Agent"]
    for i, name in enumerate(agent_names):
        avg_score = np.mean(evaluation_scores[i])
        print(f"{name}: Average score = {avg_score:.3f}")
    
    print("\nPlaying demonstration game...")
    
    # Play a demonstration game
    play_single_game(test_agents, render=True)
