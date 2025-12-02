"""
Comprehensive experiments for Numerical Geometry of RL.

Experiments:
1. Precision threshold detection
2. Error accumulation tracking
3. Q-learning stability
4. Discount factor sensitivity
5. Function approximation with DQN
"""

import torch
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(__file__))

from numerical_rl import (
    BellmanOperator, LowPrecisionBellman, QLearningAgent,
    check_rl_precision, simulate_precision
)
from environments import (
    Gridworld, FrozenLake, MultiArmedBandit,
    solve_bellman_exact, MDPSpec
)


class Experiment1_PrecisionThreshold:
    """
    Experiment 1: Detect precision threshold where value iteration fails.
    
    For each (environment, γ) pair:
    - Run value iteration at different precision levels
    - Identify threshold p* where convergence fails
    - Compare to theoretical prediction
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """Run precision threshold experiments."""
        print("\n=== Experiment 1: Precision Threshold Detection ===\n")
        
        results = {
            'gridworld_4x4': {},
            'gridworld_8x8': {},
            'frozenlake': {}
        }
        
        # Test configurations
        configs = [
            ('gridworld_4x4', Gridworld(size=4), [0.7, 0.8, 0.9, 0.95, 0.99]),
            ('gridworld_8x8', Gridworld(size=8), [0.7, 0.8, 0.9, 0.95]),
            ('frozenlake', FrozenLake(is_slippery=True), [0.7, 0.8, 0.9, 0.95])
        ]
        
        for env_name, env_obj, gammas in configs:
            print(f"\n--- {env_name} ---")
            mdp = env_obj.to_mdp_spec()
            
            for gamma in tqdm(gammas, desc=f"{env_name}"):
                result = self._test_gamma(mdp, gamma, env_name)
                results[env_name][f'gamma_{gamma}'] = result
        
        # Save results
        output_file = self.data_dir / 'experiment1_precision_threshold.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        return results
    
    def _test_gamma(self, mdp: MDPSpec, gamma: float, env_name: str) -> Dict[str, Any]:
        """Test precision threshold for a specific gamma."""
        # Precision levels to test (in bits) - extended range
        precision_levels = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 48, 64]
        
        # Ground truth
        V_star = solve_bellman_exact(mdp, gamma)
        
        # Theoretical prediction - use SAME target error as observation check
        R_max = torch.max(torch.abs(mdp.rewards)).item()
        target_error = 0.01  # Relaxed but realistic target (1% of value scale)
        theoretical = check_rl_precision(gamma, R_max, mdp.n_states, target_error=target_error)
        
        # Test each precision level
        convergence_results = []
        
        for p_bits in precision_levels:
            # Run value iteration at this precision
            bellman = LowPrecisionBellman(
                mdp.rewards, mdp.transitions, gamma, 
                precision_bits=p_bits
            )
            
            result = bellman.value_iteration(
                max_iters=1000,
                tol=1e-6,
                V_star=V_star
            )
            
            final_error = result['error_history'][-1] if len(result['error_history']) > 0 else float('inf')
            
            convergence_results.append({
                'precision_bits': p_bits,
                'converged': result['converged'],
                'final_error': final_error,
                'iterations': result['iterations'],
                'theoretical_error': result['theoretical_error']
            })
        
        # Find observed threshold - use CONSISTENT target_error
        observed_threshold = None
        for cr in convergence_results:
            # Converges if: (1) value iteration converged AND (2) final error within target
            if cr['converged'] and cr['final_error'] < target_error:
                observed_threshold = cr['precision_bits']
                break
        
        return {
            'gamma': gamma,
            'target_error': target_error,
            'theoretical_min_bits': theoretical['min_bits'],
            'theoretical_safe_bits': theoretical['safe_bits'],
            'observed_threshold': observed_threshold,
            'convergence_results': convergence_results,
            'V_max': theoretical['V_max'],
            'R_max': theoretical['R_max']
        }


class Experiment2_ErrorAccumulation:
    """
    Experiment 2: Track error accumulation over iterations.
    
    Verify that error follows predicted trajectory from Stability Composition Theorem:
    ||Ṽ_k - V*|| ≤ γ^k ||V_0 - V*|| + Δ_T·(1-γ^k)/(1-γ)
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """Run error accumulation experiments."""
        print("\n=== Experiment 2: Error Accumulation Tracking ===\n")
        
        # Use 4x4 gridworld for clarity
        env = Gridworld(size=4)
        mdp = env.to_mdp_spec()
        
        gamma = 0.9
        precision_levels = [8, 16, 24, 32, 64]
        
        # Ground truth
        V_star = solve_bellman_exact(mdp, gamma)
        
        results = {}
        
        for p_bits in tqdm(precision_levels, desc="Precision levels"):
            bellman = LowPrecisionBellman(
                mdp.rewards, mdp.transitions, gamma,
                precision_bits=p_bits
            )
            
            # Run value iteration with tracking
            result = bellman.value_iteration(
                max_iters=200,
                tol=1e-10,  # Don't stop early
                track_error=True,
                V_star=V_star
            )
            
            # Compute theoretical error bound at each iteration
            V_init = torch.zeros(mdp.n_states)
            init_error = torch.max(torch.abs(V_init - V_star)).item()
            
            theoretical_errors = []
            for k in range(len(result['error_history'])):
                morphism_k = bellman.morphism.iterate(k)
                theoretical_error = morphism_k.error_functional(init_error)
                theoretical_errors.append(theoretical_error)
            
            results[f'{p_bits}bit'] = {
                'precision_bits': p_bits,
                'observed_errors': result['error_history'].tolist(),
                'theoretical_errors': theoretical_errors,
                'Delta_T': bellman.Delta_T,
                'gamma': gamma,
                'iterations': len(result['error_history'])
            }
        
        # Save results
        output_file = self.data_dir / 'experiment2_error_accumulation.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        return results


class Experiment3_QLearningStability:
    """
    Experiment 3: Q-learning stability at different precisions.
    
    Measure:
    - Convergence rate
    - Final policy quality
    - Training instability (variance)
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, n_episodes: int = 1000, n_trials: int = 5) -> Dict[str, Any]:
        """Run Q-learning stability experiments."""
        print("\n=== Experiment 3: Q-learning Stability ===\n")
        
        # Use FrozenLake (stochastic)
        env = FrozenLake(is_slippery=True)
        
        precision_levels = [8, 16, 32]
        gammas = [0.8, 0.9, 0.95, 0.99]
        
        results = {}
        
        for gamma in tqdm(gammas, desc="Discount factors"):
            results[f'gamma_{gamma}'] = {}
            
            for p_bits in precision_levels:
                # Map bits to dtype
                if p_bits >= 32:
                    dtype = torch.float32
                elif p_bits >= 16:
                    dtype = torch.float16
                else:
                    dtype = torch.float32  # Will simulate
                
                trial_results = []
                
                for trial in range(n_trials):
                    result = self._run_qlearning_trial(
                        env, gamma, dtype, p_bits, n_episodes
                    )
                    trial_results.append(result)
                
                # Aggregate trials
                returns = [tr['episode_returns'] for tr in trial_results]
                avg_returns = np.mean(returns, axis=0).tolist()
                std_returns = np.std(returns, axis=0).tolist()
                
                final_policy_values = [tr['final_avg_return'] for tr in trial_results]
                
                results[f'gamma_{gamma}'][f'{p_bits}bit'] = {
                    'precision_bits': p_bits,
                    'avg_episode_returns': avg_returns,
                    'std_episode_returns': std_returns,
                    'final_policy_mean': float(np.mean(final_policy_values)),
                    'final_policy_std': float(np.std(final_policy_values)),
                    'converged': float(np.mean(final_policy_values)) > 0.5
                }
        
        # Save results
        output_file = self.data_dir / 'experiment3_qlearning_stability.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        return results
    
    def _run_qlearning_trial(
        self, 
        env: FrozenLake, 
        gamma: float, 
        dtype: torch.dtype,
        p_bits: int,
        n_episodes: int
    ) -> Dict[str, Any]:
        """Run a single Q-learning trial."""
        mdp = env.to_mdp_spec()
        
        # Q-learning agent
        agent = QLearningAgent(
            n_states=mdp.n_states,
            n_actions=mdp.n_actions,
            gamma=gamma,
            alpha=0.1,
            epsilon=0.1,
            dtype=dtype
        )
        
        episode_returns = []
        
        for episode in range(n_episodes):
            state = mdp.initial_state
            episode_return = 0.0
            steps = 0
            max_steps = 100
            
            while state not in mdp.terminal_states and steps < max_steps:
                # Select action
                action = agent.select_action(state)
                
                # Sample next state from transition probabilities
                probs = mdp.transitions[state, action].numpy()
                next_state = np.random.choice(mdp.n_states, p=probs)
                
                # Get reward
                reward = mdp.rewards[state, action].item()
                done = next_state in mdp.terminal_states
                
                # Update Q-table
                agent.update(state, action, reward, next_state, done)
                
                episode_return += reward * (gamma ** steps)
                state = next_state
                steps += 1
            
            episode_returns.append(episode_return)
        
        # Evaluate final policy
        final_returns = []
        for _ in range(100):
            state = mdp.initial_state
            ret = 0.0
            steps = 0
            
            while state not in mdp.terminal_states and steps < 100:
                action = torch.argmax(agent.Q[state]).item()
                probs = mdp.transitions[state, action].numpy()
                state = np.random.choice(mdp.n_states, p=probs)
                reward = mdp.rewards[state, action].item()
                ret += reward * (gamma ** steps)
                steps += 1
            
            final_returns.append(ret)
        
        return {
            'episode_returns': episode_returns,
            'final_avg_return': float(np.mean(final_returns))
        }


class Experiment4_DiscountSensitivity:
    """
    Experiment 4: Verify precision scales as log(1/(1-γ)).
    
    Vary γ from 0.5 to 0.99 and measure precision requirements.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """Run discount sensitivity experiments."""
        print("\n=== Experiment 4: Discount Factor Sensitivity ===\n")
        
        env = Gridworld(size=4)
        mdp = env.to_mdp_spec()
        
        # Vary gamma
        gammas = np.linspace(0.5, 0.99, 20)
        
        results = {
            'gammas': gammas.tolist(),
            'theoretical_min_bits': [],
            'observed_min_bits': [],
            'V_max_values': [],
            'Delta_T_values': []
        }
        
        R_max = torch.max(torch.abs(mdp.rewards)).item()
        target_error = 1e-3
        
        for gamma in tqdm(gammas, desc="Gamma values"):
            # Theoretical prediction
            theoretical = check_rl_precision(gamma, R_max, mdp.n_states, target_error)
            results['theoretical_min_bits'].append(theoretical['min_bits'])
            results['V_max_values'].append(theoretical['V_max'])
            
            # Find observed minimum by binary search
            V_star = solve_bellman_exact(mdp, gamma)
            
            # Test precision levels
            test_precisions = [4, 6, 8, 10, 12, 16, 20, 24, 32]
            observed_min = None
            
            for p_bits in test_precisions:
                bellman = LowPrecisionBellman(
                    mdp.rewards, mdp.transitions, gamma,
                    precision_bits=p_bits
                )
                
                result = bellman.value_iteration(
                    max_iters=500,
                    tol=1e-6,
                    V_star=V_star
                )
                
                final_error = result['error_history'][-1] if len(result['error_history']) > 0 else float('inf')
                
                if result['converged'] and final_error < 0.01:
                    observed_min = p_bits
                    break
            
            results['observed_min_bits'].append(observed_min if observed_min else 64)
            results['Delta_T_values'].append(bellman.Delta_T)
        
        # Save results
        output_file = self.data_dir / 'experiment4_discount_sensitivity.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        return results


class Experiment5_FunctionApproximation:
    """
    Experiment 5: DQN with tiny network on CartPole.
    
    Compare float32 vs float16 training at different discount factors.
    Show that float16 fails for γ > 0.95 but works for γ ≤ 0.9.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, n_episodes: int = 500) -> Dict[str, Any]:
        """Run function approximation experiments."""
        print("\n=== Experiment 5: Function Approximation (Tiny DQN) ===\n")
        
        # Test configurations
        configs = [
            (0.9, torch.float32, 'float32'),
            (0.9, torch.float16, 'float16'),
            (0.95, torch.float32, 'float32'),
            (0.95, torch.float16, 'float16'),
            (0.99, torch.float32, 'float32'),
            (0.99, torch.float16, 'float16'),
        ]
        
        results = {}
        
        for gamma, dtype, dtype_name in tqdm(configs, desc="Configurations"):
            key = f"gamma_{gamma}_{dtype_name}"
            
            # Run multiple trials
            trial_results = []
            for trial in range(3):  # 3 trials each
                result = self._run_tiny_dqn(gamma, dtype, n_episodes)
                trial_results.append(result)
            
            # Aggregate
            returns = [tr['episode_returns'] for tr in trial_results]
            avg_returns = np.mean(returns, axis=0).tolist()
            std_returns = np.std(returns, axis=0).tolist()
            
            losses = [tr['losses'] for tr in trial_results]
            avg_losses = np.mean(losses, axis=0).tolist()
            
            results[key] = {
                'gamma': gamma,
                'dtype': dtype_name,
                'avg_episode_returns': avg_returns,
                'std_episode_returns': std_returns,
                'avg_losses': avg_losses,
                'final_return_mean': float(np.mean([tr['final_avg_return'] for tr in trial_results])),
                'final_return_std': float(np.std([tr['final_avg_return'] for tr in trial_results]))
            }
        
        # Save results
        output_file = self.data_dir / 'experiment5_function_approximation.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        return results
    
    def _run_tiny_dqn(self, gamma: float, dtype: torch.dtype, n_episodes: int) -> Dict[str, Any]:
        """Run tiny DQN on CartPole."""
        import torch.nn as nn
        import torch.optim as optim
        
        # Tiny network: 4 → 16 → 2
        class TinyQNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 16)
                self.fc2 = nn.Linear(16, 2)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)
        
        # Create environment (simplified CartPole)
        class SimpleCartPole:
            def __init__(self):
                self.state = None
                self.steps = 0
            
            def reset(self):
                self.state = torch.randn(4) * 0.1
                self.steps = 0
                return self.state
            
            def step(self, action):
                # Simplified dynamics
                self.state = self.state + torch.randn(4) * 0.1
                self.steps += 1
                
                # Reward: survive
                reward = 1.0
                
                # Done if pole falls or too many steps
                done = (abs(self.state[2]) > 0.5) or (self.steps >= 200)
                
                return self.state, reward, done, {}
        
        env = SimpleCartPole()
        
        # Q-network
        q_net = TinyQNetwork().to(dtype=dtype)
        optimizer = optim.Adam(q_net.parameters(), lr=0.001)
        
        # Experience replay buffer (small)
        buffer = []
        buffer_size = 1000
        
        episode_returns = []
        losses = []
        epsilon = 0.1
        
        for episode in range(n_episodes):
            state = env.reset()
            episode_return = 0.0
            episode_loss = []
            
            for step in range(200):
                # Epsilon-greedy
                if np.random.random() < epsilon:
                    action = np.random.randint(2)
                else:
                    with torch.no_grad():
                        q_values = q_net(state.to(dtype=dtype))
                        action = torch.argmax(q_values).item()
                
                next_state, reward, done, _ = env.step(action)
                
                # Store transition
                buffer.append((state, action, reward, next_state, done))
                if len(buffer) > buffer_size:
                    buffer.pop(0)
                
                episode_return += reward
                
                # Training step
                if len(buffer) >= 32:
                    # Sample mini-batch
                    batch_indices = np.random.choice(len(buffer), 32, replace=False)
                    batch = [buffer[i] for i in batch_indices]
                    
                    states = torch.stack([b[0] for b in batch]).to(dtype=dtype)
                    actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
                    rewards_b = torch.tensor([b[2] for b in batch], dtype=dtype)
                    next_states = torch.stack([b[3] for b in batch]).to(dtype=dtype)
                    dones = torch.tensor([b[4] for b in batch], dtype=dtype)
                    
                    # Q-learning update
                    q_values = q_net(states)
                    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
                    
                    with torch.no_grad():
                        next_q_values = q_net(next_states)
                        max_next_q = torch.max(next_q_values, dim=1)[0]
                        targets = rewards_b + gamma * max_next_q * (1 - dones)
                    
                    loss = nn.MSELoss()(q_values, targets)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    episode_loss.append(loss.item())
                
                state = next_state
                
                if done:
                    break
            
            episode_returns.append(episode_return)
            losses.append(np.mean(episode_loss) if episode_loss else 0.0)
        
        # Evaluate final policy
        final_returns = []
        for _ in range(20):
            state = env.reset()
            ret = 0.0
            for _ in range(200):
                with torch.no_grad():
                    q_values = q_net(state.to(dtype=dtype))
                    action = torch.argmax(q_values).item()
                state, reward, done, _ = env.step(action)
                ret += reward
                if done:
                    break
            final_returns.append(ret)
        
        return {
            'episode_returns': episode_returns,
            'losses': losses,
            'final_avg_return': float(np.mean(final_returns))
        }


def run_all_experiments(data_dir: Path):
    """Run all experiments."""
    print("=" * 80)
    print("RUNNING ALL EXPERIMENTS: Numerical Geometry of RL")
    print("=" * 80)
    
    start_time = time.time()
    
    # Experiment 1: Precision Threshold
    exp1 = Experiment1_PrecisionThreshold(data_dir)
    exp1.run()
    
    # Experiment 2: Error Accumulation
    exp2 = Experiment2_ErrorAccumulation(data_dir)
    exp2.run()
    
    # Experiment 3: Q-Learning Stability
    exp3 = Experiment3_QLearningStability(data_dir)
    exp3.run(n_episodes=500, n_trials=3)  # Reduced for speed
    
    # Experiment 4: Discount Sensitivity
    exp4 = Experiment4_DiscountSensitivity(data_dir)
    exp4.run()
    
    # Experiment 5: Function Approximation
    exp5 = Experiment5_FunctionApproximation(data_dir)
    exp5.run(n_episodes=300)  # Reduced for speed
    
    elapsed = time.time() - start_time
    print(f"\n" + "=" * 80)
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Data saved to: {data_dir}")
    print("=" * 80)


if __name__ == '__main__':
    # Get data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    
    # Run all experiments
    run_all_experiments(data_dir)
