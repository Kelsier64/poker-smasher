"""
Texas Hold'em Poker Environment for Reinforcement Learning

This module provides a gymnasium-compatible poker environment for training RL agents.
Supports 2-6 players with configurable betting structures.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import random
from dataclasses import dataclass


class Action(Enum):
    FOLD = 0
    CALL = 1
    RAISE = 2
    ALL_IN = 3


class HandRank(Enum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10


@dataclass
class Card:
    suit: int  # 0-3 (spades, hearts, diamonds, clubs)
    rank: int  # 2-14 (2-10, J=11, Q=12, K=13, A=14)
    
    def __str__(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['♠', '♥', '♦', '♣']
        return f"{ranks[self.rank-2]}{suits[self.suit]}"
    
    def __eq__(self, other):
        return self.suit == other.suit and self.rank == other.rank
    
    def __hash__(self):
        return hash((self.suit, self.rank))


@dataclass
class Player:
    chips: int
    hole_cards: List[Card]
    current_bet: int
    is_folded: bool
    is_all_in: bool
    total_bet_this_hand: int
    
    def __post_init__(self):
        if self.hole_cards is None:
            self.hole_cards = []


class PokerEnv(gym.Env):
    """
    Texas Hold'em Poker Environment
    
    Observation Space:
    - Player's hole cards (2 cards)
    - Community cards (up to 5 cards)
    - Pot size
    - Player stack sizes
    - Current bet amounts
    - Position information
    - Betting round information
    
    Action Space:
    - 0: Fold
    - 1: Call/Check
    - 2: Raise (amount determined by betting structure)
    - 3: All-in (bet all remaining chips)
    """
    
    def __init__(self, 
                 num_players: int = 6,
                 initial_chips: int = 1000,
                 small_blind: int = 10,
                 big_blind: int = 20,
                 max_raises: int = 4):
        super().__init__()
        
        self.num_players = num_players
        self.initial_chips = initial_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.max_raises = max_raises
        
        # Action space: 0=fold, 1=call, 2=raise, 3=all-in
        self.action_space = spaces.Discrete(4)
        
        # Observation space
        # 52 cards (one-hot encoding) + pot + stacks + bets + position + round info
        obs_size = (
            52 +  # hole cards (2 cards one-hot)
            52 +  # community cards (5 cards one-hot)
            1 +   # pot size (normalized)
            num_players +  # player stack sizes (normalized)
            num_players +  # current bet amounts (normalized)
            1 +   # current position
            1 +   # betting round (preflop=0, flop=1, turn=2, river=3)
            1 +   # number of active players
            1     # amount to call (normalized)
        )
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize players
        self.players = [
            Player(
                chips=self.initial_chips,
                hole_cards=[],
                current_bet=0,
                is_folded=False,
                is_all_in=False,
                total_bet_this_hand=0
            ) for _ in range(self.num_players)
        ]
        
        # Game state
        self.deck = self._create_deck()
        self.community_cards = []
        self.pot = 0
        self.current_player = 0
        self.dealer_position = 0
        self.betting_round = 0  # 0=preflop, 1=flop, 2=turn, 3=river
        self.current_bet = self.big_blind
        self.min_raise = self.big_blind
        self.raises_this_round = 0
        self.last_aggressive_player = -1
        
        # Deal initial cards and post blinds
        self._deal_hole_cards()
        self._post_blinds()
        
        # Set first player to act (after big blind)
        self.current_player = (self.dealer_position + 3) % self.num_players
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int):
        if self._is_terminal():
            raise ValueError("Game is already terminal")
        
        player = self.players[self.current_player]
        reward = 0
        
        # Execute action
        if action == Action.FOLD.value:
            player.is_folded = True
        elif action == Action.CALL.value:
            call_amount = min(self.current_bet - player.current_bet, player.chips)
            player.current_bet += call_amount
            player.chips -= call_amount
            self.pot += call_amount
            if player.chips == 0:
                player.is_all_in = True
        elif action == Action.ALL_IN.value:
            # Player goes all-in with all remaining chips
            all_in_amount = player.chips
            player.current_bet += all_in_amount
            player.chips = 0
            self.pot += all_in_amount
            player.is_all_in = True
            
            # Update current bet if this all-in is higher
            if player.current_bet > self.current_bet:
                self.current_bet = player.current_bet
                self.last_aggressive_player = self.current_player
        elif action == Action.RAISE.value:
            if self.raises_this_round < self.max_raises and player.chips > 0:
                # Minimum raise is the current bet + minimum raise amount
                raise_amount = min(self.min_raise, player.chips - (self.current_bet - player.current_bet))
                if raise_amount > 0:
                    call_amount = self.current_bet - player.current_bet
                    total_bet = call_amount + raise_amount
                    player.current_bet += total_bet
                    player.chips -= total_bet
                    self.pot += total_bet
                    self.current_bet = player.current_bet
                    self.raises_this_round += 1
                    self.last_aggressive_player = self.current_player
                    if player.chips == 0:
                        player.is_all_in = True
                else:
                    # Can't raise, treat as call
                    call_amount = min(self.current_bet - player.current_bet, player.chips)
                    player.current_bet += call_amount
                    player.chips -= call_amount
                    self.pot += call_amount
                    if player.chips == 0:
                        player.is_all_in = True
            else:
                # Can't raise, treat as call
                call_amount = min(self.current_bet - player.current_bet, player.chips)
                player.current_bet += call_amount
                player.chips -= call_amount
                self.pot += call_amount
                if player.chips == 0:
                    player.is_all_in = True
        
        # Move to next player
        self.current_player = self._get_next_active_player()
        
        # Check if betting round is complete
        if self._is_betting_round_complete():
            self._advance_to_next_round()
        
        # Check if game is terminal
        terminated = self._is_terminal()
        
        if terminated:
            # Award pot to winner(s) before calculating reward
            self._award_pot()
            reward = self._calculate_final_reward()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _create_deck(self) -> List[Card]:
        """Create a standard 52-card deck"""
        deck = []
        for suit in range(4):
            for rank in range(2, 15):  # 2-14 (A=14)
                deck.append(Card(suit, rank))
        random.shuffle(deck)
        return deck
    
    def _deal_hole_cards(self):
        """Deal 2 hole cards to each player"""
        for _ in range(2):
            for player in self.players:
                player.hole_cards.append(self.deck.pop())
    
    def _post_blinds(self):
        """Post small and big blinds"""
        sb_pos = (self.dealer_position + 1) % self.num_players
        bb_pos = (self.dealer_position + 2) % self.num_players
        
        # Small blind
        sb_amount = min(self.small_blind, self.players[sb_pos].chips)
        self.players[sb_pos].current_bet = sb_amount
        self.players[sb_pos].chips -= sb_amount
        self.pot += sb_amount
        
        # Big blind
        bb_amount = min(self.big_blind, self.players[bb_pos].chips)
        self.players[bb_pos].current_bet = bb_amount
        self.players[bb_pos].chips -= bb_amount
        self.pot += bb_amount
        
        if self.players[bb_pos].chips == 0:
            self.players[bb_pos].is_all_in = True
    
    def _advance_to_next_round(self):
        """Advance to next betting round"""
        # Reset betting for next round
        for player in self.players:
            player.current_bet = 0
        
        self.current_bet = 0
        self.raises_this_round = 0
        self.last_aggressive_player = -1
        
        # Deal community cards
        if self.betting_round == 0:  # Flop
            self.community_cards.extend([self.deck.pop(), self.deck.pop(), self.deck.pop()])
        elif self.betting_round == 1:  # Turn
            self.community_cards.append(self.deck.pop())
        elif self.betting_round == 2:  # River
            self.community_cards.append(self.deck.pop())
        
        self.betting_round += 1
        
        # Set first player to act (first active player after dealer)
        self.current_player = self._get_next_active_player_from_position(self.dealer_position)
    
    def _get_next_active_player(self) -> int:
        """Get next active player in the current betting round"""
        start_pos = self.current_player
        pos = (start_pos + 1) % self.num_players
        
        while pos != start_pos:
            if not self.players[pos].is_folded and not self.players[pos].is_all_in:
                return pos
            pos = (pos + 1) % self.num_players
        
        return self.current_player
    
    def _get_next_active_player_from_position(self, start_pos: int) -> int:
        """Get next active player from a given position"""
        pos = (start_pos + 1) % self.num_players
        
        while pos != start_pos:
            if not self.players[pos].is_folded and not self.players[pos].is_all_in:
                return pos
            pos = (pos + 1) % self.num_players
        
        return start_pos
    
    def _is_betting_round_complete(self) -> bool:
        """Check if current betting round is complete"""
        active_players = [p for p in self.players if not p.is_folded and not p.is_all_in]
        
        if len(active_players) <= 1:
            return True
        
        # Check if all active players have equal bets
        current_bets = [p.current_bet for p in active_players]
        return len(set(current_bets)) <= 1
    
    def _is_terminal(self) -> bool:
        """Check if the game is terminal"""
        active_players = [p for p in self.players if not p.is_folded]
        
        # Only one player left (everyone else folded)
        if len(active_players) <= 1:
            return True
        
        # Reached showdown (river betting complete)
        if self.betting_round >= 4:
            return True
        
        return False
    
    def _calculate_final_reward(self) -> float:
        """Calculate reward for the current player at game end"""
        player = self.players[self.current_player]
        # Calculate net change from initial stack
        net_change = player.chips - self.initial_chips
        
        # Normalize by initial chips for consistent reward scale
        return net_change / self.initial_chips
    
    def _determine_winners(self) -> List[int]:
        """Determine winners at showdown"""
        active_players = [(i, p) for i, p in enumerate(self.players) if not p.is_folded]
        
        if len(active_players) == 1:
            return [active_players[0][0]]
        
        # Evaluate hands and find winners
        hand_strengths = []
        for player_idx, player in active_players:
            strength = self._evaluate_hand(player.hole_cards + self.community_cards)
            hand_strengths.append((player_idx, strength))
        
        # Find best hand strength
        best_strength = max(hand_strengths, key=lambda x: x[1])[1]
        winners = [player_idx for player_idx, strength in hand_strengths if strength == best_strength]
        
        return winners
    
    def _award_pot(self):
        """Award the pot to the winner(s)"""
        active_players = [p for p in self.players if not p.is_folded]
        
        if len(active_players) == 1:
            # Only one player left, they win the entire pot
            winner_idx = None
            for i, player in enumerate(self.players):
                if not player.is_folded:
                    winner_idx = i
                    break
            
            if winner_idx is not None:
                self.players[winner_idx].chips += self.pot
                self.pot = 0
        
        elif len(active_players) > 1:
            # Showdown - determine winners and split pot
            winners = self._determine_winners()
            pot_share = self.pot // len(winners)  # Integer division for clean split
            remainder = self.pot % len(winners)   # Any remainder chips
            
            for winner_idx in winners:
                self.players[winner_idx].chips += pot_share
                # Give remainder to first winner (dealer's choice rule)
                if remainder > 0 and winner_idx == winners[0]:
                    self.players[winner_idx].chips += remainder
            
            self.pot = 0
    
    def _evaluate_hand(self, cards: List[Card]) -> int:
        """Evaluate poker hand strength (simplified)"""
        if len(cards) < 5:
            return 0
        
        # Sort cards by rank
        sorted_cards = sorted(cards, key=lambda c: c.rank, reverse=True)
        ranks = [c.rank for c in sorted_cards]
        suits = [c.suit for c in sorted_cards]
        
        # Count ranks
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        counts = sorted(rank_counts.values(), reverse=True)
        unique_ranks = sorted(rank_counts.keys(), reverse=True)
        
        # Check for flush
        suit_counts = {}
        for suit in suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        is_flush = max(suit_counts.values()) >= 5
        
        # Check for straight
        is_straight = False
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i + 4] == 4:
                is_straight = True
                break
        
        # Special case: A-2-3-4-5 straight
        if set([14, 2, 3, 4, 5]).issubset(set(unique_ranks)):
            is_straight = True
        
        # Determine hand rank
        if is_straight and is_flush:
            return 8000000 + max(unique_ranks)
        elif counts[0] == 4:
            return 7000000 + unique_ranks[0] * 100 + unique_ranks[1]
        elif counts[0] == 3 and counts[1] == 2:
            return 6000000 + unique_ranks[0] * 100 + unique_ranks[1]
        elif is_flush:
            return 5000000 + sum(unique_ranks[:5])
        elif is_straight:
            return 4000000 + max(unique_ranks)
        elif counts[0] == 3:
            return 3000000 + unique_ranks[0] * 10000 + sum(unique_ranks[1:3])
        elif counts[0] == 2 and counts[1] == 2:
            return 2000000 + unique_ranks[0] * 10000 + unique_ranks[1] * 100 + unique_ranks[2]
        elif counts[0] == 2:
            return 1000000 + unique_ranks[0] * 1000000 + sum(unique_ranks[1:4])
        else:
            return sum(unique_ranks[:5])
    
    def _card_to_index(self, card: Card) -> int:
        """Convert card to index for one-hot encoding"""
        return card.suit * 13 + (card.rank - 2)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation for the active player"""
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        idx = 0
        
        # Player's hole cards (one-hot encoding)
        if self.current_player < len(self.players):
            player = self.players[self.current_player]
            for card in player.hole_cards:
                card_idx = self._card_to_index(card)
                obs[idx + card_idx] = 1.0
        idx += 52
        
        # Community cards (one-hot encoding)
        for card in self.community_cards:
            card_idx = self._card_to_index(card)
            obs[idx + card_idx] = 1.0
        idx += 52
        
        # Pot size (normalized)
        obs[idx] = self.pot / (self.initial_chips * self.num_players)
        idx += 1
        
        # Player stack sizes (normalized)
        for player in self.players:
            obs[idx] = player.chips / self.initial_chips
            idx += 1
        
        # Current bet amounts (normalized)
        for player in self.players:
            obs[idx] = player.current_bet / self.initial_chips
            idx += 1
        
        # Current position
        obs[idx] = self.current_player / self.num_players
        idx += 1
        
        # Betting round
        obs[idx] = self.betting_round / 3.0
        idx += 1
        
        # Number of active players
        active_count = len([p for p in self.players if not p.is_folded])
        obs[idx] = active_count / self.num_players
        idx += 1
        
        # Amount to call (normalized)
        if self.current_player < len(self.players):
            player = self.players[self.current_player]
            to_call = max(0, self.current_bet - player.current_bet)
            obs[idx] = to_call / self.initial_chips
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info for debugging/logging"""
        return {
            'pot': self.pot,
            'current_bet': self.current_bet,
            'betting_round': self.betting_round,
            'active_players': len([p for p in self.players if not p.is_folded]),
            'community_cards': [str(card) for card in self.community_cards],
            'player_stacks': [p.chips for p in self.players],
            'player_bets': [p.current_bet for p in self.players]
        }
    
    def render(self, mode='human'):
        """Render the current game state"""
        print(f"\n=== Poker Game - Round {self.betting_round} ===")
        print(f"Pot: ${self.pot}")
        print(f"Current bet: ${self.current_bet}")
        print(f"Community cards: {' '.join([str(card) for card in self.community_cards])}")
        
        print("\nPlayers:")
        for i, player in enumerate(self.players):
            status = ""
            if player.is_folded:
                status = " (FOLDED)"
            elif player.is_all_in:
                status = " (ALL-IN)"
            elif i == self.current_player:
                status = " (TO ACT)"
            
            hole_cards = " ".join([str(card) for card in player.hole_cards])
            print(f"  Player {i}: ${player.chips} (bet: ${player.current_bet}) [{hole_cards}]{status}")
        
        print()
