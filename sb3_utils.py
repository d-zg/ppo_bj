import numpy as np
import gymnasium as gym
from gymnasium.spaces import Tuple, MultiDiscrete, Discrete
import matplotlib.pyplot as plt

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
    Iterates over possible player totals (from 4 to 21),
    dealer face-up cards (1 to 10), and usable ace flags (0 or 1),
    and graphs a table showing the action chosen by the SB3 model.
    
    We fix true_count to 5 (neutral).
    The observation is (player_total, dealer_card, usable_ace, true_count).
    
    Two tables are produced:
      - One for hard hands (usable_ace = 0)
      - One for soft hands (usable_ace = 1)
    """
    fixed_true_count = 5
    # Player totals from 4 to 21: 18 possible values.
    num_totals = 21 - 4 + 1
    # Dealer's card from 1 to 10.
    num_dealer = 10

    # Prepare matrices for hard and soft hands.
    hard_table = np.zeros((num_totals, num_dealer), dtype=int)
    soft_table = np.zeros((num_totals, num_dealer), dtype=int)

    # Iterate over player total, dealer card, and usable ace flag.
    for player_total in range(4, 22):
        row = player_total - 4  # 0-indexed row for totals 4..21.
        for dealer_card in range(1, 11):
            col = dealer_card - 1  # 0-indexed column for dealer cards 1..10.
            # Hard hand (no usable ace)
            obs_hard = (player_total, dealer_card, 0, fixed_true_count, 0)
            # model.predict returns (action, state); we only care about action.
            action_hard, _ = model.predict(obs_hard, deterministic=True)
            hard_table[row, col] = action_hard
            
            # Soft hand (usable ace)
            obs_soft = (player_total, dealer_card, 1, fixed_true_count, 0)
            action_soft, _ = model.predict(obs_soft, deterministic=True)
            soft_table[row, col] = action_soft

    # Now plot the tables.
    fig, axs = plt.subplots(1, 2, figsize=(14, 8))
    
    # Plot hard hand table.
    im0 = axs[0].imshow(hard_table, cmap="coolwarm", origin="lower", aspect="auto")
    axs[0].set_title("Policy Table - Hard Hands")
    axs[0].set_xlabel("Dealer Card")
    axs[0].set_ylabel("Player Total")
    axs[0].set_xticks(np.arange(num_dealer))
    axs[0].set_xticklabels(np.arange(1, 11))
    axs[0].set_yticks(np.arange(num_totals))
    axs[0].set_yticklabels(np.arange(4, 22))
    # Annotate each cell.
    for i in range(num_totals):
        for j in range(num_dealer):
            action = hard_table[i, j]
            label = "S" if action == 0 else "H"
            axs[0].text(j, i, label, ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im0, ax=axs[0], ticks=[0, 1], label="Action (0=Stick, 1=Hit)")

    # Plot soft hand table.
    im1 = axs[1].imshow(soft_table, cmap="coolwarm", origin="lower", aspect="auto")
    axs[1].set_title("Policy Table - Soft Hands")
    axs[1].set_xlabel("Dealer Card")
    axs[1].set_ylabel("Player Total")
    axs[1].set_xticks(np.arange(num_dealer))
    axs[1].set_xticklabels(np.arange(1, 11))
    axs[1].set_yticks(np.arange(num_totals))
    axs[1].set_yticklabels(np.arange(4, 22))
    # Annotate each cell.
    for i in range(num_totals):
        for j in range(num_dealer):
            action = soft_table[i, j]
            label = "S" if action == 0 else "H"
            axs[1].text(j, i, label, ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im1, ax=axs[1], ticks=[0, 1], label="Action (0=Stick, 1=Hit)")

    plt.tight_layout()
    plt.show()
