import numpy as np
import gymnasium as gym
from gymnasium.spaces import Tuple, MultiDiscrete, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy 
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt


def load_model(file_path: str, env=None):
    """
    Loads a saved PPO model from the given file path.
    
    Args:
        file_path (str): The path to the saved model file.
        env (gym.Env, optional): The environment to associate with the model.
            This is useful if you want to continue training or evaluate the model.
    
    Returns:
        PPO: The loaded PPO model.
    """
    try:
        model = PPO.load(file_path, env=env)
        print(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {file_path}: {e}")
        raise

class FlattenTupleObservation(gym.ObservationWrapper):
    """
    An observation wrapper that converts a Tuple observation space (composed of Discrete and/or MultiDiscrete spaces)
    into a single MultiDiscrete space by concatenating the information.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Ensure that the observation space is a Tuple.
        if not isinstance(env.observation_space, Tuple):
            raise ValueError("FlattenTupleObservation only works with Tuple observation spaces.")
        
        # Build the new nvec list from each subspace.
        new_nvec = []
        for space in env.observation_space.spaces:
            if isinstance(space, Discrete):
                new_nvec.append(space.n)
            elif isinstance(space, MultiDiscrete):
                new_nvec.extend(space.nvec)
            else:
                raise NotImplementedError("Only Discrete and MultiDiscrete subspaces are supported for flattening.")
        self.observation_space = MultiDiscrete(np.array(new_nvec, dtype=np.int64))
    
    def observation(self, observation):
        """
        Flattens the tuple observation into a single 1D array.
        """
        flat_obs = []
        for obs in observation:
            if isinstance(obs, (int, np.integer)):
                flat_obs.append(obs)
            else:
                flat_obs.extend(np.array(obs).flatten().tolist())
        return np.array(flat_obs, dtype=np.int64)

def get_flattened_env(env: gym.Env) -> gym.Env:
    """
    Takes an environment whose observation space is a Tuple and returns an identical environment
    wrapped with FlattenTupleObservation so that its observation space is flattened into a single MultiDiscrete.
    
    :param env: The original gym environment.
    :return: The wrapped environment with a flattened observation space.
    """
    if isinstance(env.observation_space, Tuple):
        env = FlattenTupleObservation(env)
    return env

def test_policy_table(model):
    """
    Revised function:
    
    - The model now returns one of four actions:
         0: Stick, 1: Hit, 2: Double Down, 3: Split
    - We iterate over all true counts from 0 to 10.
    - For each true count we compute:
         * A hard-hand policy table (usable_ace=0, can_split=0) for player totals 4-21.
         * A soft-hand policy table (usable_ace=1, can_split=0) for player totals 4-21.
         * A splitting decision table. Here, we only consider even player totals.
           For player_total==22 (i.e. pair of aces) we set usable_ace=1; for other even totals, usable_ace=0.
           In this case, the can_split flag is set to 1.
    
    - The full set of tables (for each true count) is stored in dictionaries.
    - We plot the hard, soft, and split tables for true count = 5.
    
    Returns:
      - policy_tables: a dict mapping each true count to a dict with keys 'hard' and 'soft'
      - split_tables: a dict mapping each true count to the corresponding splitting table.
    """
    # Define ranges.
    true_counts = range(0, 11)             # True counts 0 to 10 inclusive.
    player_totals = range(4, 22)           # Policy tables for player totals 4 to 21.
    dealer_cards = range(1, 11)            # Dealer cards 1 to 10.
    
    num_totals = len(player_totals)        # 18 rows.
    num_dealer = len(dealer_cards)         # 10 columns.
    
    # Dictionaries to store the results for each true count.
    policy_tables = {}  # Each key: true_count, value: {'hard': table, 'soft': table}
    split_tables = {}   # Each key: true_count, value: splitting table.
    
    # For splitting decisions, we only consider even player totals.
    # We take even totals in 4-21 plus add 22 (to represent pair of aces).
    splitting_totals = [pt for pt in range(4, 22) if pt % 2 == 0]
    splitting_totals.append(22)  # 22 will represent the pair of aces.
    splitting_totals = sorted(splitting_totals)
    num_split_totals = len(splitting_totals)
    
    for tc in true_counts:
        # Initialize tables for policy decisions.
        hard_table = np.zeros((num_totals, num_dealer), dtype=int)
        soft_table = np.zeros((num_totals, num_dealer), dtype=int)
        # Initialize table for splitting decisions.
        # We fill with -1 where splitting is not applicable.
        split_table = np.full((num_split_totals, num_dealer), -1, dtype=int)
        
        # Compute policy tables (can_split flag = 0).
        for i, pt in enumerate(player_totals):
            for j, dc in enumerate(dealer_cards):
                # Hard hand: usable_ace = 0.
                obs_hard = (pt, dc, 0, tc, 0)
                action_hard, _ = model.predict(obs_hard, deterministic=True)
                hard_table[i, j] = action_hard
                
                # Soft hand: usable_ace = 1.
                obs_soft = (pt, dc, 1, tc, 0)
                action_soft, _ = model.predict(obs_soft, deterministic=True)
                soft_table[i, j] = action_soft
        
        # Compute splitting table (can_split flag = 1) for even totals.
        for i, pt in enumerate(splitting_totals):
            for j, dc in enumerate(dealer_cards):
                # For a pair of aces (represented as 22) use usable_ace = 1.
                usable_ace = 1 if pt == 22 else 0
                obs_split = (pt, dc, usable_ace, tc, 1)
                action_split, _ = model.predict(obs_split, deterministic=True)
                split_table[i, j] = action_split
        
        # Save results for this true count.
        policy_tables[tc] = {'hard': hard_table, 'soft': soft_table}
        split_tables[tc] = split_table

    # Define a mapping from action numbers to labels.
    action_labels = {
        0: "Stick",
        1: "Hit",
        2: "Double",
        3: "Split"
    }
    
    # Plotting for true count = 5.
    tc_plot = 5
    fig, axs = plt.subplots(1, 3, figsize=(21, 8))
    
    # Plot Hard Hand Policy Table.
    hard_table_plot = policy_tables[tc_plot]['hard']
    im0 = axs[0].imshow(hard_table_plot, cmap="coolwarm", origin="lower", aspect="auto")
    axs[0].set_title(f"Policy Table (Hard) - True Count {tc_plot}")
    axs[0].set_xlabel("Dealer Card")
    axs[0].set_ylabel("Player Total")
    axs[0].set_xticks(np.arange(num_dealer))
    axs[0].set_xticklabels(list(dealer_cards))
    axs[0].set_yticks(np.arange(num_totals))
    axs[0].set_yticklabels(list(player_totals))
    # Annotate each cell.
    for i in range(num_totals):
        for j in range(num_dealer):
            action = hard_table_plot[i, j]
            label = action_labels.get(action, str(action))
            axs[0].text(j, i, label, ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im0, ax=axs[0], ticks=[0, 1, 2, 3], label="Action")
    
    # Plot Soft Hand Policy Table.
    soft_table_plot = policy_tables[tc_plot]['soft']
    im1 = axs[1].imshow(soft_table_plot, cmap="coolwarm", origin="lower", aspect="auto")
    axs[1].set_title(f"Policy Table (Soft) - True Count {tc_plot}")
    axs[1].set_xlabel("Dealer Card")
    axs[1].set_ylabel("Player Total")
    axs[1].set_xticks(np.arange(num_dealer))
    axs[1].set_xticklabels(list(dealer_cards))
    axs[1].set_yticks(np.arange(num_totals))
    axs[1].set_yticklabels(list(player_totals))
    for i in range(num_totals):
        for j in range(num_dealer):
            action = soft_table_plot[i, j]
            label = action_labels.get(action, str(action))
            axs[1].text(j, i, label, ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im1, ax=axs[1], ticks=[0, 1, 2, 3], label="Action")
    
    # Plot Splitting Table.
    im2 = axs[2].imshow(split_tables[tc_plot], cmap="coolwarm", origin="lower", aspect="auto")
    axs[2].set_title(f"Split Decision Table - True Count {tc_plot}")
    axs[2].set_xlabel("Dealer Card")
    axs[2].set_ylabel("Player Total (Pairs)")
    axs[2].set_xticks(np.arange(num_dealer))
    axs[2].set_xticklabels(list(dealer_cards))
    axs[2].set_yticks(np.arange(num_split_totals))
    axs[2].set_yticklabels(splitting_totals)
    for i in range(num_split_totals):
        for j in range(num_dealer):
            action = split_tables[tc_plot][i, j]
            label = action_labels.get(action, str(action))
            axs[2].text(j, i, label, ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im2, ax=axs[2], ticks=[0, 1, 2, 3], label="Action")
    
    plt.tight_layout()
    plt.show()
    
    return policy_tables, split_tables


def compute_policy_and_split_tables(model):
    """
    Computes policy tables (hard and soft) and split tables for true counts from 0 to 10.
    
    For each true count:
      - Policy tables are computed over player totals 4 to 21 and dealer cards 1 to 10.
        * Hard table uses usable_ace = 0 and can_split = 0.
        * Soft table uses usable_ace = 1 and can_split = 0.
      - The splitting table is computed only for even player totals (plus 22 for a pair of aces).
        * For a player total of 22 (pair of aces) usable_ace is set to 1; otherwise 0.
        * Here, can_split flag is set to 1.
    
    Returns:
      - policy_tables: dict mapping each true count to a dict with keys 'hard' and 'soft'.
      - split_tables: dict mapping each true count to the corresponding splitting table.
    """
    true_counts = range(0, 11)             # True counts 0 to 10 inclusive.
    player_totals = range(4, 22)           # For policy tables: player totals 4 to 21.
    dealer_cards = range(1, 11)            # Dealer cards 1 to 10.
    
    num_totals = len(player_totals)        # 18 rows.
    num_dealer = len(dealer_cards)         # 10 columns.
    
    # Define the even totals (plus 22 to represent a pair of aces) for splitting decisions.
    splitting_totals = [pt for pt in range(4, 22) if pt % 2 == 0]
    splitting_totals.append(22)  # 22 will represent the pair of aces.
    splitting_totals = sorted(splitting_totals)
    num_split_totals = len(splitting_totals)
    
    policy_tables = {}  # Dictionary to store policy tables for each true count.
    split_tables = {}   # Dictionary to store splitting tables for each true count.
    
    for tc in true_counts:
        # Initialize policy tables for this true count.
        hard_table = np.zeros((num_totals, num_dealer), dtype=int)
        soft_table = np.zeros((num_totals, num_dealer), dtype=int)
        # Initialize split table with -1 where splitting is not applicable.
        split_table = np.full((num_split_totals, num_dealer), -1, dtype=int)
        
        # Compute hard and soft policy tables.
        for i, pt in enumerate(player_totals):
            for j, dc in enumerate(dealer_cards):
                # Hard hand: usable_ace = 0, can_split = 0.
                obs_hard = (pt, dc, 0, tc, 0)
                action_hard, _ = model.predict(obs_hard, deterministic=True)
                hard_table[i, j] = action_hard
                
                # Soft hand: usable_ace = 1, can_split = 0.
                obs_soft = (pt, dc, 1, tc, 0)
                action_soft, _ = model.predict(obs_soft, deterministic=True)
                soft_table[i, j] = action_soft
        
        # Compute splitting table for even player totals (and 22 for aces).
        for i, pt in enumerate(splitting_totals):
            for j, dc in enumerate(dealer_cards):
                # For a pair of aces (pt == 22), use usable_ace = 1; otherwise 0.
                usable_ace = 1 if pt == 22 else 0
                obs_split = (pt, dc, usable_ace, tc, 1)
                action_split, _ = model.predict(obs_split, deterministic=True)
                split_table[i, j] = action_split
        
        policy_tables[tc] = {'hard': hard_table, 'soft': soft_table}
        split_tables[tc] = split_table

    return policy_tables, split_tables

def plot_policy_and_split_tables(policy_tables, split_tables, true_count=5):
    """
    Plots the hard hand, soft hand, and split decision tables for a given true count.
    
    Parameters:
      - policy_tables: dict (from compute_policy_and_split_tables) containing keys 'hard' and 'soft'.
      - split_tables: dict (from compute_policy_and_split_tables) with split decision tables.
      - true_count: the true count value to plot (default is 5).
    """
    # Define labels for actions.
    action_labels = {
        0: "Stick",
        1: "Hit",
        2: "Double",
        3: "Split"
    }
    
    # Setup ranges for plotting.
    player_totals = list(range(4, 22))
    dealer_cards = list(range(1, 11))
    num_totals = len(player_totals)
    num_dealer = len(dealer_cards)
    
    # For splitting decisions.
    splitting_totals = [pt for pt in range(4, 22) if pt % 2 == 0]
    splitting_totals.append(22)
    splitting_totals = sorted(splitting_totals)
    num_split_totals = len(splitting_totals)
    
    # Extract tables for the specified true count.
    hard_table = policy_tables[true_count]['hard']
    soft_table = policy_tables[true_count]['soft']
    split_table = split_tables[true_count]
    
    # Create a figure with three subplots.
    fig, axs = plt.subplots(1, 3, figsize=(21, 8))
    
    # Plot Hard Hand Policy Table.
    im0 = axs[0].imshow(hard_table, cmap="coolwarm", origin="lower", aspect="auto")
    axs[0].set_title(f"Policy Table (Hard) - True Count {true_count}")
    axs[0].set_xlabel("Dealer Card")
    axs[0].set_ylabel("Player Total")
    axs[0].set_xticks(np.arange(num_dealer))
    axs[0].set_xticklabels(dealer_cards)
    axs[0].set_yticks(np.arange(num_totals))
    axs[0].set_yticklabels(player_totals)
    for i in range(num_totals):
        for j in range(num_dealer):
            action = hard_table[i, j]
            label = action_labels.get(action, str(action))
            axs[0].text(j, i, label, ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im0, ax=axs[0], ticks=[0, 1, 2, 3], label="Action")
    
    # Plot Soft Hand Policy Table.
    im1 = axs[1].imshow(soft_table, cmap="coolwarm", origin="lower", aspect="auto")
    axs[1].set_title(f"Policy Table (Soft) - True Count {true_count}")
    axs[1].set_xlabel("Dealer Card")
    axs[1].set_ylabel("Player Total")
    axs[1].set_xticks(np.arange(num_dealer))
    axs[1].set_xticklabels(dealer_cards)
    axs[1].set_yticks(np.arange(num_totals))
    axs[1].set_yticklabels(player_totals)
    for i in range(num_totals):
        for j in range(num_dealer):
            action = soft_table[i, j]
            label = action_labels.get(action, str(action))
            axs[1].text(j, i, label, ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im1, ax=axs[1], ticks=[0, 1, 2, 3], label="Action")
    
    # Plot Splitting Table.
    im2 = axs[2].imshow(split_table, cmap="coolwarm", origin="lower", aspect="auto")
    axs[2].set_title(f"Split Decision Table - True Count {true_count}")
    axs[2].set_xlabel("Dealer Card")
    axs[2].set_ylabel("Player Total (Pairs)")
    axs[2].set_xticks(np.arange(num_dealer))
    axs[2].set_xticklabels(dealer_cards)
    axs[2].set_yticks(np.arange(num_split_totals))
    axs[2].set_yticklabels(splitting_totals)
    for i in range(num_split_totals):
        for j in range(num_dealer):
            action = split_table[i, j]
            label = action_labels.get(action, str(action))
            axs[2].text(j, i, label, ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im2, ax=axs[2], ticks=[0, 1, 2, 3], label="Action")
    
    plt.tight_layout()
    plt.show()

def create_env(render_mode, evaluation_mode=True):
    env = gym.make(
        "Blackjack-v1",
        render_mode=render_mode,
        natural=True,
        sab=False,
        evaluation_mode=evaluation_mode,
    )
    return get_flattened_env(env)

def evaluate_reward(model, eval_env, num_episodes):
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=num_episodes, deterministic=True)
    print(f"Initial evaluation: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes=5, **kwargs):
        super().__init__(eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, **kwargs)
        self.eval_timesteps = []
        self.mean_rewards = []

    def _on_step(self) -> bool:
        # Call parent class evaluation step
        continue_training = super()._on_step()
        # When evaluation happens, the callback records the timesteps and mean reward.
        if self.n_calls % self.eval_freq == 0:
            self.eval_timesteps.append(self.num_timesteps)
            self.mean_rewards.append(self.last_mean_reward)
        return continue_training