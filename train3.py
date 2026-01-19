import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
from torch.distributions.multivariate_normal import MultivariateNormal

from environment import MazeEnv
from model import Model
from algorithm import NEXT_plan, RRTS_plan
from utils import set_random_seed, load_model, plot_tree


class MSILTrainer:
    """Meta Self-Improving Learning Trainer"""

    def __init__(self,
                 dim: int = 2,
                 cuda: bool = True,
                 learning_rate: float = 1e-3,
                 lambda_reg: float = 0.01,
                 n_problems: int = 2000,
                 dataset_size: int = 250,
                 update_frequency: int = 10,
                 planning_steps: int = 500,
                 g_explore_eps: float = 0.1,
                 UCB_type: str = 'kde',
                 env_width: int = 15,
                 model_cap: int = 8,
                 random_seed: Optional[int] = None):
        """
        Args:
            dim: Dimension of the problem (2 or 3)
            cuda: Whether to use GPU
            learning_rate: Learning rate for gradient descent
            lambda_reg: Regularization coefficient (λ in loss function)
            n_problems: Total number of problems to solve (2000 in paper)
            dataset_size: Maximum size of dataset D_n
            planning_steps: T parameter for planning algorithms
            g_explore_eps: Exploration epsilon for goal sampling
            UCB_type: Type of UCB sampling ('kde' or other)
            env_width: Width of environment grid
            model_cap: Capacity parameter for model
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            set_random_seed(random_seed)

        self.dim = dim
        self.cuda = cuda
        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.lr = learning_rate
        self.lambda_reg = lambda_reg
        self.n_problems = n_problems
        self.dataset_size = dataset_size
        self.planning_steps = planning_steps
        self.g_explore_eps = g_explore_eps
        self.UCB_type = UCB_type
        self.update_frequency = update_frequency

        # Initialize environment and model
        self.environment = MazeEnv(dim=dim)
        self.model = Model(
            cuda=cuda,
            env_width=env_width,
            model_cap=model_cap,
            dim=dim,
            UCB_type=UCB_type
        )

        # Setup optimizer
        self.optimizer = optim.Adam(self.model.net.parameters(), lr=learning_rate)

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[80, 120, 160],  # 800/10, 1200/10, 1600/10
            gamma=0.5
        )

        # Initialize empty dataset: stores (search_tree, problem) tuples
        self.dataset = []

        print("=" * 60)
        print("MSIL Trainer initialized")
        print(f"  Dimension: {dim}D")
        print(f"  Environment width: {env_width}")
        print(f"  Model capacity: {model_cap}")
        print(f"  Device: {'GPU' if cuda else 'CPU'}")
        if cuda and torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_id)
            print(
                f"  Device: GPU ({props.name}) | "
                f"VRAM: {props.total_memory / 1024 ** 3:.1f} GB | "
                f"SMs: {props.multi_processor_count}"
            )
        print("=" * 60)

    def load_pretrained_model(self, model_file: str):
        """Load a pretrained model to continue training"""
        load_model(self.model.net, model_file, self.cuda)
        print(f"Loaded pretrained model from {model_file}")

    def compute_epsilon(self, problem_num: int) -> float:
        """
        Compute epsilon based on problem number following the paper's schedule:
        ε = 1.0                              if i < 1000
        ε = 0.5 - 0.1 * ⌊(i - 1000)/200⌋     if 1000 ≤ i < 2000
        ε = 0.1                              otherwise

        Args:
            problem_num: Current problem number (1-indexed)

        Returns:
            epsilon: Exploration probability
        """
        if problem_num < 1000:
            return 1.0
        elif problem_num < 2000:
            return 0.5 - 0.1 * np.floor((problem_num - 1000) / 200)
        else:
            return 0.1

    def sample_planning_problem(self, problem_num: int) -> dict:
        """
        Sample a planning problem and update environment
        Uses TSA(U) with epsilon ∈ Uni[0,1]

        Args:
            problem_num: Current problem number

        Returns:
            problem: Dictionary with start, goal, map
        """
        problem = self.environment.init_new_problem(problem_num)

        # Set problem in model (computes problem representation)
        self.model.set_problem(problem)

        return problem

    def plan_with_exploration(self, epsilon: float) -> Tuple[object, bool]:
        # With probability epsilon, use pure RRT exploration
        if np.random.random() < epsilon:
            search_tree, success = RRTS_plan(
                env=self.environment,
                T=self.planning_steps,
                stop_when_success=True
            )
        else:
            # Use NEXT with learned model
            search_tree, success = NEXT_plan(
                env=self.environment,
                model=self.model,
                T=self.planning_steps,
                g_explore_eps=self.g_explore_eps,
                stop_when_success=True,
                UCB_type=self.UCB_type
            )

        return search_tree, success

    def reconstruct_optimal_path(self, search_tree: object) -> List[int]:
        """
        Reconstruct optimal path {s^i}_{i=1}^m from search tree
        Uses rewired_parents to get the optimized path from RRT*

        Args:
            search_tree: SearchTree object with states and parents

        Returns:
            path_indices: List of state indices from start to goal
        """
        if not search_tree.in_goal_region[-1]:
            return []

        path = []

        # Start from the last state (should be in goal region)
        current_idx = len(search_tree.states) - 1

        # Work backwards using rewired_parents (optimized by RRT*)
        while current_idx is not None:
            path.append(current_idx)
            current_idx = search_tree.rewired_parents[current_idx]

        # Reverse to get start-to-goal order
        path.reverse()

        return path

    def compute_path_costs(self, search_tree: object, path_indices: List[int]) -> torch.Tensor:
        """
        Compute cost-to-go for each state in path
        y^i = sum of costs from s^i to goal (cost-to-go)
        """
        m = len(path_indices)
        costs = torch.zeros(m)

        # Goal state has cost 0
        costs[m - 1] = 0.0

        # Work backwards from goal
        for i in range(m - 2, -1, -1):
            state_curr = search_tree.states[path_indices[i]]
            state_next = search_tree.states[path_indices[i + 1]]
            step_cost = self.environment.distance(state_curr, state_next)
            costs[i] = costs[i + 1] + float(step_cost)

        return costs

    def compute_log_prob_policy(self, state: np.ndarray, next_state: np.ndarray) -> torch.Tensor:
        """
        Compute log π*(s_{i+1}|s_i) using the model's policy

        Args:
            state: Current state s_i
            next_state: Next state s_{i+1}

        Returns:
            log_prob: Log probability of the action
        """
        # Get predicted action mean from the model
        pred_action, _ = self.model.net_forward(state)

        # The action taken was next_state - state
        actual_action = next_state - state

        # Model uses Gaussian policy with fixed covariance
        var = self.model.var
        if self.cuda:
            var = var.cuda()

        # Create distribution
        mean = torch.FloatTensor(pred_action)
        if self.cuda:
            mean = mean.cuda()

        m = MultivariateNormal(mean, var)

        # Compute log probability
        action_tensor = torch.FloatTensor(actual_action)
        if self.cuda:
            action_tensor = action_tensor.cuda()

        log_prob = m.log_prob(action_tensor)

        return log_prob

    def compute_loss(self) -> torch.Tensor:
        if len(self.dataset) == 0:
            return torch.zeros((), device=self.device)

        total_policy_loss = torch.zeros((), device=self.device)
        total_value_loss = torch.zeros((), device=self.device)
        total_transitions = 0  # Count transitions, not problems

        for search_tree, problem in self.dataset:
            self.model.set_problem(problem)
            path_indices = self.reconstruct_optimal_path(search_tree)
            if len(path_indices) < 2:
                continue

            states = search_tree.states[path_indices]
            costs = self.compute_path_costs(search_tree, path_indices).to(self.device)

            # Policy loss - accumulate per transition
            for i in range(len(states) - 1):
                total_policy_loss -= self.compute_log_prob_policy(states[i], states[i + 1])
                total_transitions += 1

            # Value loss - accumulate per state
            for i, state in enumerate(states):
                v_pred = self.model.pred_value(state)
                total_value_loss += (v_pred - costs[i]).pow(2)

        if total_transitions == 0:
            return torch.zeros((), device=self.device)

        # Average over ALL transitions, not problems
        avg_policy_loss = total_policy_loss / total_transitions
        avg_value_loss = total_value_loss / total_transitions

        # Regularization
        reg_loss = torch.zeros((), device=self.device)
        for p in self.model.net.parameters():
            reg_loss += p.pow(2).sum()

        return avg_policy_loss + avg_value_loss + self.lambda_reg * reg_loss

    def update_parameters(self):
        """Update model parameters using gradient descent on dataset"""
        if len(self.dataset) == 0:
            return None

        self.optimizer.zero_grad()
        loss = self.compute_loss()

        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.net.parameters(), max_norm=1.0)
            self.optimizer.step()
            return loss.item()
        else:
            return loss.item()

    def train(self):
        """Main training loop implementing Algorithm 3 with paper's hyperparameters"""

        print("\n" + "=" * 60)
        print("Starting Meta Self-Improving Learning (MSIL)")
        print("=" * 60)
        print(f"Dimension: {self.dim}D")
        print(f"Total problems: {self.n_problems}")
        print(f"Dataset size: {self.dataset_size}")
        print(f"Planning steps: {self.planning_steps}")
        print(f"Learning rate: {self.lr}")
        print(f"Regularization (λ): {self.lambda_reg}")
        print("=" * 60)
        print("Epsilon schedule:")
        print("  Problems 1-999:     ε = 1.0   (pure exploration)")
        print("  Problems 1000-1999: ε = 0.5 - 0.1⌊(i-1000)/200⌋")
        print("  Problems 2000+:     ε = 0.1   (mostly exploitation)")
        print("=" * 60 + "\n")

        # Initialize dataset D_0 (empty)
        self.dataset = []

        success_count = 0
        rrt_count = 0
        next_count = 0
        problems_since_update = 0

        for problem_num in range(1, self.n_problems + 1):
            # Compute epsilon for this problem
            epsilon = self.compute_epsilon(problem_num)

            # Print progress every 100 problems
            if problem_num % 100 == 1 or problem_num == self.n_problems:
                print(f"\n{'=' * 60}")
                print(f"Problem {problem_num}/{self.n_problems}")
                print(f"{'=' * 60}")
                print(f"Current epsilon: {epsilon:.4f}")
                print(f"Dataset size: {len(self.dataset)}/{self.dataset_size}")
                print(
                    f"Success rate: {success_count}/{problem_num - 1} ({100 * success_count / max(1, problem_num - 1):.1f}%)")
                print(f"Problems since last update: {problems_since_update}")

            # Step 1: Sample planning problem U from TSA(U)
            problem = self.sample_planning_problem(problem_num)

            # Step 2: Plan with epsilon-greedy exploration
            search_tree, success = self.plan_with_exploration(epsilon)

            if success:
                success_count += 1

                # Step 3: Add successful experience to dataset
                self.dataset.append((search_tree, problem))

                # Maintain fixed dataset size (like experience replay)
                if len(self.dataset) > self.dataset_size:
                    self.dataset.pop(0)

                if problem_num % 100 == 1 or problem_num == self.n_problems:
                    path = self.reconstruct_optimal_path(search_tree)
                    print(f"✓ Planning successful!")
                    print(f"  States explored: {len(search_tree.states)}")
                    print(f"  Path length (L): {len(path)}")
            else:
                if problem_num % 100 == 1 or problem_num == self.n_problems:
                    print("✗ Planning failed")

            problems_since_update += 1

            # Step 4: Update parameters every 'update_frequency' problems
            if problems_since_update >= self.update_frequency and len(self.dataset) > 0:
                loss = self.update_parameters()
                if loss is not None:
                    self.scheduler.step()
                    print(f"\n>>> Parameter update at problem {problem_num}")
                    print(f"    Loss: {loss:.6f}")
                    print(f"    Dataset size: {len(self.dataset)}")

                problems_since_update = 0

            # Periodic model saving
            if problem_num % 500 == 0:
                self.save_model(f'msil_problem_{problem_num}.pt')

        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        print(f"Final success rate: {success_count}/{self.n_problems} ({100 * success_count / self.n_problems:.1f}%)")
        print(f"Final dataset size: {len(self.dataset)}")
        print("=" * 60)

    def save_model(self, filepath: str):
        """Save the current model"""
        torch.save(self.model.net.state_dict(), filepath)
        print(f"  Model saved to {filepath}")

# Usage
if __name__ == "__main__":
    # Initialize trainer for 2D problems with paper's hyperparameters
    trainer = MSILTrainer(
        dim=2,
        cuda=True,
        n_problems=2000,
        dataset_size=250,
        planning_steps=500,
        learning_rate=1e-3,
        lambda_reg=1e-5,
        random_seed=42
    )

    # Optional: Load pretrained model
    # trainer.load_pretrained_model('trained_models/NEXT_2d.pt')

    # Run training
    trainer.train()

    # Save final model
    trainer.save_model('trained_models/MSIL_2d_final.pt')