"""
Environments for RL experiments.

Provides simple tabular MDPs for testing numerical properties:
- Gridworld (4x4, 8x8)
- FrozenLake
- Multi-armed bandits
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class MDPSpec:
    """Specification of a Markov Decision Process."""
    n_states: int
    n_actions: int
    rewards: torch.Tensor  # (n_states, n_actions)
    transitions: torch.Tensor  # (n_states, n_actions, n_states)
    initial_state: int = 0
    terminal_states: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.terminal_states is None:
            self.terminal_states = []


class Gridworld:
    """
    Simple gridworld MDP.
    
    States: grid cells (row, col) flattened to integers
    Actions: 0=up, 1=right, 2=down, 3=left
    Rewards: -1 per step, +10 at goal, -10 at holes
    Transitions: deterministic movement (or stay if hit wall)
    """
    
    def __init__(self, size: int = 4, goal_reward: float = 10.0, hole_reward: float = -10.0):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.goal_reward = goal_reward
        self.hole_reward = hole_reward
        
        # Goal at bottom-right
        self.goal_state = self.n_states - 1
        
        # Holes at random positions (but not start or goal)
        np.random.seed(42)
        n_holes = max(1, size - 2)
        possible_holes = [i for i in range(1, self.n_states - 1)]
        self.holes = sorted(np.random.choice(possible_holes, size=n_holes, replace=False).tolist())
        
        self.terminal_states = [self.goal_state] + self.holes
        
        # Build MDP
        self.rewards, self.transitions = self._build_mdp()
    
    def _state_to_coords(self, state: int) -> Tuple[int, int]:
        """Convert state index to (row, col) coordinates."""
        return (state // self.size, state % self.size)
    
    def _coords_to_state(self, row: int, col: int) -> int:
        """Convert (row, col) to state index."""
        return row * self.size + col
    
    def _next_state(self, state: int, action: int) -> int:
        """Compute next state given action (deterministic)."""
        row, col = self._state_to_coords(state)
        
        # Action effects: up, right, down, left
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # right
            col = min(self.size - 1, col + 1)
        elif action == 2:  # down
            row = min(self.size - 1, row + 1)
        elif action == 3:  # left
            col = max(0, col - 1)
        
        return self._coords_to_state(row, col)
    
    def _build_mdp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build reward and transition tensors."""
        rewards = torch.zeros((self.n_states, self.n_actions))
        transitions = torch.zeros((self.n_states, self.n_actions, self.n_states))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                # Terminal states have no transitions
                if s in self.terminal_states:
                    transitions[s, a, s] = 1.0
                    rewards[s, a] = 0.0
                else:
                    next_s = self._next_state(s, a)
                    transitions[s, a, next_s] = 1.0
                    
                    # Reward structure
                    if next_s == self.goal_state:
                        rewards[s, a] = self.goal_reward
                    elif next_s in self.holes:
                        rewards[s, a] = self.hole_reward
                    else:
                        rewards[s, a] = -1.0  # Step cost
        
        return rewards, transitions
    
    def to_mdp_spec(self) -> MDPSpec:
        """Convert to MDPSpec."""
        return MDPSpec(
            n_states=self.n_states,
            n_actions=self.n_actions,
            rewards=self.rewards,
            transitions=self.transitions,
            initial_state=0,
            terminal_states=self.terminal_states
        )
    
    def render_policy(self, V: torch.Tensor, Q: Optional[torch.Tensor] = None) -> str:
        """
        Render value function as grid.
        
        Args:
            V: Value function (n_states,)
            Q: Optional Q-function (n_states, n_actions) for policy arrows
        
        Returns:
            String representation of grid
        """
        lines = []
        arrows = ['↑', '→', '↓', '←']
        
        for row in range(self.size):
            line_vals = []
            line_policy = []
            
            for col in range(self.size):
                state = self._coords_to_state(row, col)
                val = V[state].item()
                
                # Format value
                if state == self.goal_state:
                    cell = f"GOAL"
                elif state in self.holes:
                    cell = f"HOLE"
                else:
                    cell = f"{val:+.1f}"
                
                line_vals.append(f"{cell:>6}")
                
                # Policy arrow
                if Q is not None and state not in self.terminal_states:
                    best_action = torch.argmax(Q[state]).item()
                    line_policy.append(arrows[best_action])
                else:
                    line_policy.append(' ')
            
            lines.append(' '.join(line_vals))
            if Q is not None:
                lines.append('   ' + '     '.join(line_policy))
        
        return '\n'.join(lines)


class FrozenLake:
    """
    FrozenLake environment (4x4).
    
    States: 16 grid cells
    Actions: 0=left, 1=down, 2=right, 3=up
    Stochastic transitions: intended direction with prob 1/3,
                           perpendicular directions with prob 1/3 each
    Rewards: +1 at goal, 0 elsewhere
    """
    
    def __init__(self, is_slippery: bool = True):
        self.size = 4
        self.n_states = 16
        self.n_actions = 4
        self.is_slippery = is_slippery
        
        # Layout (standard FrozenLake):
        # SFFF
        # FHFH
        # FFFH
        # HFFG
        # S=start, F=frozen, H=hole, G=goal
        self.holes = [5, 7, 11, 12]
        self.goal_state = 15
        self.terminal_states = self.holes + [self.goal_state]
        
        self.rewards, self.transitions = self._build_mdp()
    
    def _state_to_coords(self, state: int) -> Tuple[int, int]:
        return (state // self.size, state % self.size)
    
    def _coords_to_state(self, row: int, col: int) -> int:
        # Clip to valid range
        row = max(0, min(self.size - 1, row))
        col = max(0, min(self.size - 1, col))
        return row * self.size + col
    
    def _next_states(self, state: int, action: int) -> List[Tuple[int, float]]:
        """
        Get possible next states with probabilities.
        
        Returns:
            List of (next_state, probability) pairs
        """
        row, col = self._state_to_coords(state)
        
        # Action to direction: left, down, right, up
        directions = [
            (0, -1),  # left
            (1, 0),   # down
            (0, 1),   # right
            (-1, 0)   # up
        ]
        
        if not self.is_slippery:
            # Deterministic
            dr, dc = directions[action]
            next_state = self._coords_to_state(row + dr, col + dc)
            return [(next_state, 1.0)]
        else:
            # Stochastic: intended direction 1/3, perpendiculars 1/3 each
            # Perpendicular to action a: (a-1)%4 and (a+1)%4
            actions = [
                action,
                (action - 1) % 4,
                (action + 1) % 4
            ]
            
            result = []
            for a in actions:
                dr, dc = directions[a]
                next_state = self._coords_to_state(row + dr, col + dc)
                result.append((next_state, 1.0 / 3.0))
            
            return result
    
    def _build_mdp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build reward and transition tensors."""
        rewards = torch.zeros((self.n_states, self.n_actions))
        transitions = torch.zeros((self.n_states, self.n_actions, self.n_states))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if s in self.terminal_states:
                    # Terminal states: self-loop
                    transitions[s, a, s] = 1.0
                    rewards[s, a] = 0.0
                else:
                    # Get next states with probabilities
                    next_states = self._next_states(s, a)
                    
                    for next_s, prob in next_states:
                        transitions[s, a, next_s] += prob
                        
                        # Reward for reaching goal
                        if next_s == self.goal_state:
                            rewards[s, a] += prob * 1.0
    
        return rewards, transitions
    
    def to_mdp_spec(self) -> MDPSpec:
        return MDPSpec(
            n_states=self.n_states,
            n_actions=self.n_actions,
            rewards=self.rewards,
            transitions=self.transitions,
            initial_state=0,
            terminal_states=self.terminal_states
        )


class MultiArmedBandit:
    """
    Multi-armed bandit as a single-state MDP.
    
    States: 1 (always in same state)
    Actions: n_arms
    Rewards: drawn from known distributions
    Transitions: deterministic (stay in same state)
    """
    
    def __init__(self, n_arms: int = 10, reward_std: float = 1.0):
        self.n_arms = n_arms
        self.n_states = 1
        self.n_actions = n_arms
        
        # True mean rewards (unknown to agent)
        np.random.seed(42)
        self.true_means = np.random.randn(n_arms)
        self.reward_std = reward_std
        
        # For MDP specification, use mean rewards
        rewards = torch.zeros((1, n_arms))
        rewards[0, :] = torch.from_numpy(self.true_means).float()
        
        # Deterministic transition (stay in state 0)
        transitions = torch.zeros((1, n_arms, 1))
        transitions[0, :, 0] = 1.0
        
        self.rewards = rewards
        self.transitions = transitions
    
    def sample_reward(self, action: int) -> float:
        """Sample reward for taking action (arm)."""
        return np.random.normal(self.true_means[action], self.reward_std)
    
    def to_mdp_spec(self) -> MDPSpec:
        return MDPSpec(
            n_states=self.n_states,
            n_actions=self.n_actions,
            rewards=self.rewards,
            transitions=self.transitions,
            initial_state=0,
            terminal_states=[]
        )


def solve_bellman_exact(mdp: MDPSpec, gamma: float, tol: float = 1e-10) -> torch.Tensor:
    """
    Solve Bellman equation exactly using double precision value iteration.
    
    This provides ground truth V* for error measurement.
    """
    V = torch.zeros(mdp.n_states, dtype=torch.float64)
    
    for _ in range(10000):
        # Bellman update
        expected_next = torch.matmul(mdp.transitions.double(), V)
        Q = mdp.rewards.double() + gamma * expected_next
        V_new = torch.max(Q, dim=1)[0]
        
        # Check convergence
        if torch.max(torch.abs(V_new - V)).item() < tol:
            break
        
        V = V_new
    
    return V.float()


class CartPoleEnvWrapper:
    """
    Wrapper for CartPole with tiny function approximator.
    
    This is not a tabular MDP, but we can still analyze numerical properties
    of the Bellman updates in the function approximation setting.
    """
    
    def __init__(self, gamma: float = 0.99):
        try:
            import gym
            self.env = gym.make('CartPole-v1')
        except:
            # Fallback: create a mock environment
            self.env = None
        
        self.gamma = gamma
        self.n_actions = 2
        self.state_dim = 4
    
    def reset(self):
        if self.env is not None:
            return self.env.reset()
        else:
            return np.random.randn(4) * 0.1
    
    def step(self, action):
        if self.env is not None:
            return self.env.step(action)
        else:
            # Mock step
            next_state = np.random.randn(4) * 0.1
            reward = 1.0
            done = np.random.random() < 0.01
            info = {}
            return next_state, reward, done, info
    
    def close(self):
        if self.env is not None:
            self.env.close()
