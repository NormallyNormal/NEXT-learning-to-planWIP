"""
Test script for evaluating a trained NEXT model.

Loads weights from a trained model and computes loss on planning problems.
"""

import argparse
import numpy as np
import torch

from environment import MazeEnv
from model import Model
from algorithm import NEXT_plan, RRTS_plan
from utils import set_random_seed, load_model


def extract_path_from_tree(search_tree, env):
    """
    Extract the successful path from the search tree.
    """
    goal_idx = None
    for i, in_goal in enumerate(search_tree.in_goal_region):
        if in_goal and search_tree.freesp[i]:
            goal_idx = i
            break

    if goal_idx is None:
        return None, None

    path = []
    costs = []
    idx = goal_idx

    while idx is not None:
        path.append(search_tree.states[idx])
        parent_idx = search_tree.rewired_parents[idx]
        if parent_idx is not None:
            cost = env.distance(search_tree.states[parent_idx], search_tree.states[idx])
            costs.append(cost)
        idx = parent_idx

    path = path[::-1]
    costs = costs[::-1]

    return path, costs


def compute_costs_to_go(costs):
    """
    Compute cost-to-go for each state in the path.
    """
    costs_arr = np.array(costs, dtype=np.float32)
    total = float(np.sum(costs_arr))

    costs_to_go = []
    cumsum = 0.0
    for c in costs_arr:
        costs_to_go.append(total - cumsum)
        cumsum += float(c)
    costs_to_go.append(0.0)

    return costs_to_go


gaussianNLLLoss = torch.nn.GaussianNLLLoss(reduction='sum')


def compute_loss(model, path, costs_to_go, device):
    """
    Compute the loss for a given path and costs-to-go.
    """
    dim = model.dim
    std = model.std

    states = torch.FloatTensor(np.array(path))
    targets_value = torch.FloatTensor(np.array(costs_to_go))

    if model.cuda:
        states = states.cuda()
        targets_value = targets_value.cuda()

    y = model.net.state_forward(states, model.pb_rep)
    pred_actions = y[:, :dim]
    pred_values = y[:, -1]

    policy_loss = torch.tensor(0.0, device=device)

    if len(path) > 1:
        actions = states[1:] - states[:-1]
        action_means = pred_actions[:-1]
        var = std ** 2
        policy_loss = gaussianNLLLoss(action_means, actions, torch.full_like(action_means, var))

    value_loss = ((pred_values - targets_value) ** 2).sum()

    total_loss = policy_loss + value_loss

    return total_loss, policy_loss.item(), value_loss.item()


def run_planning(env, model, T, epsilon):
    """
    Run planning with epsilon-mixture of RRT and NEXT expansion.
    """
    if epsilon >= 1.0:
        return RRTS_plan(env=env, T=T, stop_when_success=True)
    else:
        return NEXT_plan(
            env=env,
            model=model,
            T=T,
            g_explore_eps=epsilon,
            stop_when_success=True
        )


def test(args):
    """
    Test the trained model on planning problems and compute loss.
    """
    set_random_seed(args.seed)

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"\nInitializing {args.dim}D environment...")
    env = MazeEnv(dim=args.dim)
    num_problems = min(args.num_problems, env.size)
    print(f"Total problems available: {env.size}")
    print(f"Problems to evaluate: {num_problems}")

    model = Model(cuda=args.cuda, dim=args.dim)

    print(f"\nLoading model weights from {args.model_path}")
    load_model(model.net, args.model_path, args.cuda)
    model.net.eval()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_successes = 0
    num_evaluated = 0
    path_costs = []

    print("\n" + "=" * 70)
    print("Starting evaluation")
    print("=" * 70)

    for problem_idx in range(num_problems):
        problem = env.init_new_problem(problem_idx)

        with torch.no_grad():
            model.set_problem(problem)

            search_tree, success = run_planning(
                env=env,
                model=model,
                T=args.max_samples,
                epsilon=args.epsilon
            )

        if success:
            num_successes += 1
            path, costs = extract_path_from_tree(search_tree, env)

            if path is not None and len(path) > 1:
                costs_to_go = compute_costs_to_go(costs)
                path_cost = float(np.sum(costs))
                path_costs.append(path_cost)

                with torch.no_grad():
                    loss, policy_loss, value_loss = compute_loss(
                        model=model,
                        path=path,
                        costs_to_go=costs_to_go,
                        device=device
                    )

                total_loss += loss.item()
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                num_evaluated += 1

        if (problem_idx + 1) % args.report_frequency == 0:
            success_rate = num_successes / (problem_idx + 1)
            avg_loss = total_loss / num_evaluated if num_evaluated > 0 else 0
            avg_policy = total_policy_loss / num_evaluated if num_evaluated > 0 else 0
            avg_value = total_value_loss / num_evaluated if num_evaluated > 0 else 0
            avg_path_cost = np.mean(path_costs) if path_costs else 0

            print(f"\n[Problem {problem_idx + 1}/{num_problems}]")
            print(f"  Success rate: {success_rate:.3f}")
            print(f"  Avg path cost: {avg_path_cost:.4f}")
            print(f"  Avg total loss: {avg_loss:.4f}")
            print(f"  Avg policy loss: {avg_policy:.4f}")
            print(f"  Avg value loss: {avg_value:.4f}")

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)

    success_rate = num_successes / num_problems
    avg_loss = total_loss / num_evaluated if num_evaluated > 0 else 0
    avg_policy = total_policy_loss / num_evaluated if num_evaluated > 0 else 0
    avg_value = total_value_loss / num_evaluated if num_evaluated > 0 else 0
    avg_path_cost = np.mean(path_costs) if path_costs else 0

    print(f"\nFinal Results:")
    print(f"  Problems evaluated: {num_problems}")
    print(f"  Successful paths: {num_successes}")
    print(f"  Success rate: {success_rate:.3f}")
    print(f"  Average path cost: {avg_path_cost:.4f}")
    print(f"  Average total loss: {avg_loss:.4f}")
    print(f"  Average policy loss: {avg_policy:.4f}")
    print(f"  Average value loss: {avg_value:.4f}")

    return {
        'success_rate': success_rate,
        'avg_loss': avg_loss,
        'avg_policy_loss': avg_policy,
        'avg_value_loss': avg_value,
        'avg_path_cost': avg_path_cost,
        'num_successes': num_successes,
        'num_evaluated': num_evaluated
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test a trained NEXT model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model-path', type=str, default='trained_models/NEXT_2d.pt',
                        help='Path to trained model weights')
    parser.add_argument('--dim', type=int, default=2, choices=[2, 3],
                        help='Dimension of the planning problem')
    parser.add_argument('--num-problems', type=int, default=100,
                        help='Number of problems to evaluate')
    parser.add_argument('--max-samples', type=int, default=500,
                        help='Maximum samples per planning problem')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Epsilon for RRT/NEXT mixture (0.1 = mostly NEXT)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use CUDA if available')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda',
                        help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--report-frequency', type=int, default=10,
                        help='Print progress every N problems')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("NEXT Model Evaluation")
    print("=" * 70)
    print("\nConfiguration:")
    print("-" * 40)
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print("-" * 40)

    test(args)


if __name__ == '__main__':
    main()
