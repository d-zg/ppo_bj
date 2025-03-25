#!/usr/bin/env python3
import argparse
import time

import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from sb3_utils import get_flattened_env, test_policy_table, create_env, CustomEvalCallback
from optimal import optimal_policy  # if needed elsewhere


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on Blackjack-v1 with evaluation, plotting, and saving."
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1_000_000,
        help="Total number of timesteps for training.",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=100_000,
        help="Number of timesteps between evaluations.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="ppo_blackjack_model",
        help="File path to save the trained model.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Plot the training performance after training.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run with human rendering after training.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use for training (cpu or cuda)."
    )
    return parser.parse_args()



def train_and_evaluate(model, eval_env, total_timesteps, eval_freq):
    timesteps_list = [0]
    mean_rewards = []

    # Initial evaluation before training begins
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000, deterministic=True)
    print(f"Initial evaluation: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    mean_rewards.append(mean_reward)

    current_steps = 0
    while current_steps < total_timesteps:
        model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
        current_steps += eval_freq

        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000, deterministic=True)
        timesteps_list.append(current_steps)
        mean_rewards.append(mean_reward)
        print(f"After {current_steps} timesteps: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    return timesteps_list, mean_rewards


def plot_performance(timesteps, rewards):
    plt.figure(figsize=(8, 4))
    plt.plot(timesteps, rewards, marker="o")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("PPO Performance over Time")
    plt.grid(True)
    plt.show()


def run_human_rendering(model, episodes):
    env = create_env(render_mode="human", evaluation_mode=True)
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        print(f"Starting episode {ep+1}")
        env.render()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            print("Observation:", obs)
            env.render()
            time.sleep(0.2)  # slow down rendering for visibility
        input(f"Episode {ep+1} finished. Press Enter to continue to the next episode...")
    env.close()


def save_model(model, save_path):
    model.save(save_path)
    print(f"Model saved to {save_path}")


def main():
    args = parse_args()

    # Create environments for training and evaluation
    train_env = create_env(render_mode="rgb_array", evaluation_mode=True)
    eval_env = create_env(render_mode=None, evaluation_mode=True)

    callback_env = Monitor(eval_env)

    # Set up the evaluation callback to run every 5000 steps over 5 episodes
    eval_callback = CustomEvalCallback(
        callback_env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=args.eval_freq,
        n_eval_episodes=1000,
        verbose=1
    )

    print("Training Environment created")
    print("Action Space:", train_env.action_space)
    print("Observation Space:", train_env.observation_space)

    # Initialize the PPO model
    model = PPO("MlpPolicy", train_env, device=args.device, learning_rate = lambda progress_remaining: progress_remaining * 3e-4)
 

    # Train and periodically evaluate the model
    # timesteps, rewards = train_and_evaluate(model, eval_env, args.total_timesteps, args.eval_freq)
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
    timesteps = eval_callback.eval_timesteps
    rewards = eval_callback.mean_rewards

    # Optionally plot the performance
    if args.plot:
        plot_performance(timesteps, rewards)

    # Save the trained model
    save_model(model, args.save_path)

    # Run a few episodes with human rendering for inspection
    run_human_rendering(model, args.episodes)

    # Clean up evaluation environment
    eval_env.close()

    # Optionally, test the policy table
    test_policy_table(model)


if __name__ == "__main__":
    main()
