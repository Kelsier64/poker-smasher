# Poker Smasher 🃏

A comprehensive Texas Hold'em poker environment for reinforcement learning training.

## Features

- **Gymnasium-compatible environment** for easy integration with RL frameworks
- **Texas Hold'em poker** with 2-6 players
- **Configurable betting structures** (blinds, max raises, etc.)
- **Multiple agent types** included:
  - Random Agent (baseline)
  - Heuristic Agent (rule-based)
  - Q-Learning Agent (example RL)
- **Rich observation space** including:
  - Hole cards and community cards
  - Pot size and betting information
  - Player positions and stack sizes
  - Betting round information
- **Proper hand evaluation** with standard poker hand rankings
- **Training utilities** and demonstration scripts

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install gymnasium numpy typing_extensions
   ```

## Quick Start

Run the main demo:
```bash
python main.py
```

This will:
1. Create a poker environment
2. Demonstrate different agent types
3. Play a sample game with visualization
4. Run a short training session

## Usage

### Basic Environment Usage

```python
from poker_env import PokerEnv

# Create environment
env = PokerEnv(
    num_players=6,
    initial_chips=1000,
    small_blind=10,
    big_blind=20
)

# Reset environment
observation, info = env.reset()

# Take actions
action = 1  # 0=fold, 1=call, 2=raise
observation, reward, terminated, truncated, info = env.step(action)

# Render game state
env.render()
```

### Training Your Own Agents

```python
from poker_env import PokerEnv
from train import QLearningAgent

# Create environment and agent
env = PokerEnv(num_players=3)
agent = QLearningAgent(env.observation_space.shape[0], env.action_space.n)

# Training loop
for episode in range(1000):
    observation, info = env.reset()
    
    while True:
        action = agent.get_action(observation, [0, 1, 2])  # legal actions
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Update agent
        agent.update({
            'observation': observation,
            'action': action,
            'reward': reward,
            'next_observation': next_obs,
            'done': done
        })
        
        if done or truncated:
            break
        
        observation = next_obs
```

## Environment Details

### Observation Space

The observation is a 1D numpy array containing:
- **Hole cards**: One-hot encoding of player's 2 cards (52 dimensions)
- **Community cards**: One-hot encoding of up to 5 community cards (52 dimensions)
- **Pot size**: Normalized pot size (1 dimension)
- **Player stacks**: Normalized stack sizes for all players (num_players dimensions)
- **Current bets**: Normalized current bet amounts (num_players dimensions)
- **Position**: Current player position (1 dimension)
- **Betting round**: 0=preflop, 1=flop, 2=turn, 3=river (1 dimension)
- **Active players**: Number of active players (1 dimension)
- **Amount to call**: Normalized amount needed to call (1 dimension)

### Action Space

- **0**: Fold
- **1**: Call/Check
- **2**: Raise (amount determined by betting structure)

### Rewards

Rewards are calculated at the end of each hand:
- Positive reward for winning chips
- Negative reward for losing chips
- Normalized by initial stack size

## Included Agents

### RandomAgent
Chooses random actions for baseline comparison.

### SimpleHeuristicAgent
Uses basic poker heuristics:
- Hand strength evaluation
- Pot odds calculation
- Position-aware decisions

### QLearningAgent
Simple Q-learning implementation with:
- Tabular Q-function
- Epsilon-greedy exploration
- State abstraction for tractable learning

## Extending the Environment

You can easily extend the environment by:

1. **Creating custom agents**: Inherit from `PokerAgent` and implement `get_action()` and `update()` methods
2. **Modifying betting structures**: Adjust blinds, raise limits, and bet sizing
3. **Adding more sophisticated RL algorithms**: The environment is compatible with any RL framework
4. **Implementing different poker variants**: Modify the game rules as needed

## Example: Custom Agent

```python
from train import PokerAgent
import numpy as np

class MyCustomAgent(PokerAgent):
    def get_action(self, observation: np.ndarray, legal_actions: list) -> int:
        # Your decision logic here
        return legal_actions[0]  # Simple example
    
    def update(self, experience: dict) -> None:
        # Your learning logic here
        pass
```

## Files Structure

- `poker_env.py`: Main poker environment implementation
- `train.py`: Training utilities and example agents
- `main.py`: Demo script and entry point
- `pyproject.toml`: Project configuration

## Contributing

Feel free to contribute by:
- Adding new agent implementations
- Improving the environment
- Adding more poker variants
- Optimizing performance
- Adding more sophisticated hand evaluation

## License

This project is open source. Feel free to use and modify as needed.

---

Happy training! 🎰🤖
