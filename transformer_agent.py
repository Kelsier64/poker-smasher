"""
Transformer-based Poker Agent for Self-Play Training

This module implements a transformer neural network for poker decision making,
using attention mechanisms to process game states and card sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
from dataclasses import dataclass
import math

from train import PokerAgent
from poker_env import Action


@dataclass
class TransformerConfig:
    """Configuration for the transformer model"""
    state_dim: int = 106  # Poker observation space size
    action_dim: int = 4   # Number of actions (FOLD, CALL, RAISE, ALL_IN)
    sequence_length: int = 32  # Maximum sequence length for game history
    hidden_dim: int = 256  # Hidden dimension
    num_heads: int = 8     # Number of attention heads
    num_layers: int = 6    # Number of transformer layers
    dropout: float = 0.1   # Dropout rate
    learning_rate: float = 1e-4
    batch_size: int = 64
    gamma: float = 0.99    # Discount factor
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: int = 1000
    memory_size: int = 10000
    target_update: int = 100


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PokerTransformer(nn.Module):
    """Transformer neural network for poker decision making"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.input_embedding = nn.Linear(config.state_dim, config.hidden_dim)
        self.pos_encoder = PositionalEncoding(config.hidden_dim, config.sequence_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Output heads
        self.q_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.action_dim)
        )
        
        # Value head for advantage estimation
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1)
        )
        
    def forward(self, x, mask=None):
        """
        Forward pass through the transformer
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, state_dim)
            mask: Optional attention mask
        
        Returns:
            q_values: Q-values for each action
            state_value: State value estimate
        """
        # Input embedding
        embedded = self.input_embedding(x)  # (batch, seq_len, hidden_dim)
        embedded = embedded.transpose(0, 1)  # (seq_len, batch, hidden_dim)
        embedded = self.pos_encoder(embedded)
        embedded = embedded.transpose(0, 1)  # (batch, seq_len, hidden_dim)
        
        # Transformer encoding
        if mask is not None:
            # Create attention mask for padding
            attn_mask = mask.unsqueeze(1).repeat(1, mask.size(1), 1)
            encoded = self.transformer(embedded, src_key_padding_mask=~mask)
        else:
            encoded = self.transformer(embedded)
        
        # Use the last non-masked position for output
        if mask is not None:
            # Get the last valid position for each sequence
            seq_lengths = mask.sum(dim=1) - 1
            batch_idx = torch.arange(encoded.size(0))
            last_encoded = encoded[batch_idx, seq_lengths]
        else:
            last_encoded = encoded[:, -1, :]  # Use last position
        
        # Output heads
        q_values = self.q_head(last_encoded)
        state_value = self.value_head(last_encoded)
        
        return q_values, state_value


class ExperienceReplay:
    """Experience replay buffer for training"""
    
    def __init__(self, capacity: int, sequence_length: int):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=capacity)
        
    def push(self, sequence: List[np.ndarray], action: int, reward: float, 
             next_sequence: List[np.ndarray], done: bool):
        """Add experience to the buffer"""
        # Pad sequences to fixed length
        padded_seq = self._pad_sequence(sequence)
        padded_next_seq = self._pad_sequence(next_sequence)
        
        self.buffer.append((padded_seq, action, reward, padded_next_seq, done))
    
    def _pad_sequence(self, sequence: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Pad sequence to fixed length and create mask"""
        if len(sequence) >= self.sequence_length:
            # Truncate if too long
            padded = np.stack(sequence[-self.sequence_length:])
            mask = np.ones(self.sequence_length, dtype=bool)
        else:
            # Pad if too short
            padding_length = self.sequence_length - len(sequence)
            padding = np.zeros((padding_length, sequence[0].shape[0]))
            padded = np.vstack([padding, np.stack(sequence)])
            mask = np.concatenate([np.zeros(padding_length, dtype=bool), 
                                 np.ones(len(sequence), dtype=bool)])
        
        return padded, mask
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        states, masks, actions, rewards, next_states, next_masks, dones = [], [], [], [], [], [], []
        
        for seq, action, reward, next_seq, done in batch:
            state, mask = seq
            next_state, next_mask = next_seq
            
            states.append(state)
            masks.append(mask)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            next_masks.append(next_mask)
            dones.append(done)
        
        return (torch.FloatTensor(states),
                torch.BoolTensor(masks),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.BoolTensor(next_masks),
                torch.BoolTensor(dones))
    
    def __len__(self):
        return len(self.buffer)


class TransformerAgent(PokerAgent):
    """Transformer-based poker agent using DQN with experience replay"""
    
    def __init__(self, config: TransformerConfig = None, device: str = None):
        self.config = config or TransformerConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Neural networks
        self.q_network = PokerTransformer(self.config).to(self.device)
        self.target_network = PokerTransformer(self.config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        
        # Experience replay
        self.memory = ExperienceReplay(self.config.memory_size, self.config.sequence_length)
        
        # Training state
        self.steps_done = 0
        self.episode_count = 0
        self.state_sequence = []
        self.last_sequence = None
        self.last_action = None
        
        # Metrics
        self.training_losses = []
        self.q_values_history = []
    
    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        """Get action using epsilon-greedy policy with transformer network"""
        # Add current observation to sequence
        self.state_sequence.append(observation.copy())
        
        # Calculate epsilon for exploration
        epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * \
                 math.exp(-1. * self.steps_done / self.config.epsilon_decay)
        
        if random.random() > epsilon:
            # Exploit: use neural network
            with torch.no_grad():
                # Prepare sequence
                if len(self.state_sequence) >= self.config.sequence_length:
                    seq = self.state_sequence[-self.config.sequence_length:]
                    mask = torch.ones(self.config.sequence_length, dtype=torch.bool)
                else:
                    # Pad sequence
                    padding_length = self.config.sequence_length - len(self.state_sequence)
                    padding = [np.zeros_like(observation) for _ in range(padding_length)]
                    seq = padding + self.state_sequence
                    mask = torch.cat([torch.zeros(padding_length, dtype=torch.bool), 
                                    torch.ones(len(self.state_sequence), dtype=torch.bool)])
                
                # Convert to tensor
                state_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                mask_tensor = mask.unsqueeze(0).to(self.device)
                
                # Get Q-values
                q_values, _ = self.q_network(state_tensor, mask_tensor)
                q_values = q_values.cpu().numpy().flatten()
                
                # Mask illegal actions
                masked_q_values = np.full(len(q_values), -float('inf'))
                for action in legal_actions:
                    masked_q_values[action] = q_values[action]
                
                action = np.argmax(masked_q_values)
                
                # Store Q-values for analysis
                self.q_values_history.append(q_values.copy())
        else:
            # Explore: random action
            action = random.choice(legal_actions)
        
        # Store for learning
        self.last_sequence = self.state_sequence.copy()
        self.last_action = action
        self.steps_done += 1
        
        return action
    
    def update(self, experience: Dict[str, Any]) -> None:
        """Update the agent with new experience"""
        if self.last_sequence is None or self.last_action is None:
            return
        
        reward = experience.get('reward', 0)
        next_observation = experience.get('next_observation')
        done = experience.get('done', False)
        
        # Create next sequence
        if next_observation is not None and not done:
            next_sequence = self.state_sequence.copy()
        else:
            next_sequence = self.last_sequence.copy()
        
        # Add experience to replay buffer
        self.memory.push(self.last_sequence, self.last_action, reward, next_sequence, done)
        
        # Clear sequence at episode end
        if done:
            self.state_sequence = []
            self.episode_count += 1
        
        # Train if we have enough experiences
        if len(self.memory) >= self.config.batch_size:
            self._train_step()
        
        # Update target network periodically
        if self.steps_done % self.config.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _train_step(self):
        """Perform one training step"""
        # Sample batch
        states, masks, actions, rewards, next_states, next_masks, dones = \
            self.memory.sample(self.config.batch_size)
        
        states = states.to(self.device)
        masks = masks.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        next_masks = next_masks.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q_values, current_values = self.q_network(states, masks)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Next Q-values (Double DQN)
        with torch.no_grad():
            next_q_values, next_values = self.q_network(next_states, next_masks)
            next_actions = next_q_values.argmax(1)
            
            target_next_q_values, _ = self.target_network(next_states, next_masks)
            target_next_q_values = target_next_q_values.gather(1, next_actions.unsqueeze(1))
            
            target_q_values = rewards.unsqueeze(1) + \
                            (self.config.gamma * target_next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Store loss for monitoring
        self.training_losses.append(loss.item())
    
    def save(self, filepath: str) -> None:
        """Save the transformer agent"""
        save_dict = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'steps_done': self.steps_done,
            'episode_count': self.episode_count,
            'training_losses': self.training_losses,
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(save_dict, filepath)
        print(f"Transformer agent saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: str = None) -> 'TransformerAgent':
        """Load a transformer agent"""
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        save_dict = torch.load(filepath, map_location=device)
        
        # Create agent
        agent = cls(config=save_dict['config'], device=device)
        
        # Load state
        agent.q_network.load_state_dict(save_dict['q_network_state_dict'])
        agent.target_network.load_state_dict(save_dict['target_network_state_dict'])
        agent.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        agent.steps_done = save_dict.get('steps_done', 0)
        agent.episode_count = save_dict.get('episode_count', 0)
        agent.training_losses = save_dict.get('training_losses', [])
        
        print(f"Transformer agent loaded from {filepath}")
        print(f"Training steps: {agent.steps_done}, Episodes: {agent.episode_count}")
        
        return agent
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'steps_done': self.steps_done,
            'episode_count': self.episode_count,
            'memory_size': len(self.memory),
            'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0,
            'avg_q_values': np.mean([np.mean(q) for q in self.q_values_history[-100:]]) if self.q_values_history else 0,
            'epsilon': self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * \
                     math.exp(-1. * self.steps_done / self.config.epsilon_decay)
        }
