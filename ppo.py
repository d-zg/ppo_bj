import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_utils import get_flattened_env, test_policy_table
from optimal import optimal_policy
import time

# Create training environment with rendering mode turned off during training
env = gym.make('Blackjack-v1', render_mode="rgb_array", natural=True, sab=False, evaluation_mode=True)
print("Action Space:", env.action_space)
print("Observation Space:", env.observation_space)
env = get_flattened_env(env)

# Create a separate evaluation environment (without rendering)
eval_env = gym.make('Blackjack-v1', render_mode=None, natural=True, sab=False, evaluation_mode=True)
eval_env = get_flattened_env(eval_env)

# Instantiate the PPO model
model = PPO("MlpPolicy", env, device="cpu")

# Set training parameters
total_timesteps = 10000000
eval_freq = 1000000  # evaluate every 100 timesteps

# Lists to store timesteps and performance
timesteps_list = []
mean_rewards = []

current_steps = 0
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000, deterministic=True)
print(mean_reward)
timesteps_list.append(0)
mean_rewards.append(mean_reward)
while current_steps < total_timesteps:
    # Train for a short interval; reset_num_timesteps=False to keep timesteps cumulative
    model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
    current_steps += eval_freq

    # Evaluate the current model on the evaluation environment
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000, deterministic=True)
    timesteps_list.append(current_steps)
    mean_rewards.append(mean_reward)
    print(f"After {current_steps} timesteps: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")

# Plot the performance over timesteps
plt.figure(figsize=(8, 4))
plt.plot(timesteps_list, mean_rewards, marker='o')
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title("PPO Performance over Time")
plt.grid(True)
plt.show()

# Now, run a few episodes with rendering so you can visually inspect the agent.
# Switch the environment to human rendering mode.
env = gym.make('Blackjack-v1', render_mode="human", natural=True, sab=False, evaluation_mode=True)
env = get_flattened_env(env)

num_episodes = 5
for ep in range(num_episodes):
    obs, info = env.reset()
    done = False
    print(f"Starting episode {ep+1}")
    # Render the initial frame
    env.render()
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        print(obs)
        env.render()
        time.sleep(0.2)  # slow down rendering so you can see what's happening
    input(f"Episode {ep+1} finished. Press Enter to continue to the next episode...")
env.close()
eval_env.close()

test_policy_table(model)
