#!/usr/bin/env python3
import time
import sys
import gymnasium as gym 

def main():
    # Create the environment with render_mode "human"
    # env = BlackjackEnv(render_mode="human", natural=True, sab=False, num_decks=5, evaluation_mode=False)
    env = gym.make('Blackjack-v1', render_mode="human", natural=True, sab=False, evaluation_mode=True)
    
    try:
        while True:
            # Reset the environment at the start of each hand (episode)
            obs, info = env.reset()
            done = False
            print("\n=== New Hand Started! ===")
            env.render()
            while not done:
                # Prompt the user for an action
                user_input = input("Enter action (0: Stick, 1: Hit, 2: Double Down, 3: Split): ").strip()
                try:
                    action = int(user_input)
                except ValueError:
                    print("Invalid input. Exiting the game.")
                    sys.exit(0)
                if action not in [0, 1, 2, 3]:
                    print("Action not in allowed set {0,1,2,3}. Exiting the game.")
                    sys.exit(0)
                # Take the action in the environment
                obs, reward, done, truncated, info = env.step(action)
                print(obs)
                print(f"Step reward: {reward}")
                env.render()
                time.sleep(0.2)  # Optional: slow down the loop for viewing
            print("Hand finished.")
    except KeyboardInterrupt:
        print("\nCtrl-C received, exiting game.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
