import gymnasium as gym
import numpy as np
from sb3_utils import get_flattened_env
# implements the with replacement optimal strategy (no hitting or doubling)


def optimal_policy(obs):
    """
    Given a flattened observation (an array of integers):
      [player_total, dealer_card, usable_ace, true_count]
    returns the optimal action (0 for stick, 1 for hit) based on a simplified basic strategy.
    Assumes no splitting or doubling.
    """
    total, dealer_card, usable, _, _ = obs  # ignore true count
    is_soft = False
    # If there is a usable ace and counting it as 11 doesn't bust, treat as soft.
    if usable == 1 and total + 10 <= 21:
        total += 10
        is_soft = True

    # Hard totals:
    if not is_soft:
        if total <= 11:
            return 1  # hit
        if total >= 17:
            return 0  # stick
        if total in [12, 13, 14, 15, 16]:
            # Stand if dealer shows a weak card (2-6); otherwise hit.
            return 0 if 2 <= dealer_card <= 6 else 1

    # Soft totals:
    else:
        if total >= 19:
            return 0  # stick on soft 19 or higher
        if total == 18:
            # Stand if dealer shows 2, 7, or 8; otherwise hit.
            return 0 if dealer_card in [2, 7, 8] else 1
        # For soft totals below 18, hit.
        return 1

def play_episode(env):
    """
    Plays one episode using the optimal_policy and returns the cumulative reward.
    """
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = optimal_policy(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    return total_reward

def main():
    # Create the environment without rendering.
    env = gym.make('Blackjack-v1', render_mode=None, natural=True, sab=False)
    env = get_flattened_env(env)

    num_episodes = 100000 # Change this to any number of episodes you want to run.
    episode_rewards = []

    for ep in range(num_episodes):
        ep_reward = play_episode(env)
        episode_rewards.append(ep_reward)
        # print(f"Episode {ep+1} reward: {ep_reward}")

    env.close()

    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\nPlayed {num_episodes} episodes with an average reward of {avg_reward:.4f} ± {std_reward:.2f}")

if __name__ == "__main__":
    main()
