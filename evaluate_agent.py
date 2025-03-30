import time
import gymnasium as gym
from stable_baselines3 import PPO

# Make sure main.py is accessible
try:
    from main import BouncyBreakoutEnv
except ImportError as e:
    print(f"Error: Could not import BouncyBreakoutEnv from main.py: {e}")
    exit()

# --- Configuration ---
# IMPORTANT: Use the exact path to the saved model .zip file
MODEL_PATH = "logs/models/ppo_bouncy_breakout_v1_10000000_steps.zip"
N_EPISODES_TO_RUN = 3  # How many games to watch the agent play

# --- Load the Trained Agent ---
print(f"Loading model from: {MODEL_PATH}")
try:
    # Set device to 'auto' to match training (important for MPS)
    model = PPO.load(MODEL_PATH, device="auto")
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Did the training script save the model correctly?")
    exit()
except Exception as e:
    print(f"An error occurred loading the model: {e}")
    exit()


# --- Create Environment for Evaluation (with rendering) ---
print("Creating evaluation environment with rendering...")
eval_env = BouncyBreakoutEnv(render_mode="human")

# --- Run Evaluation Loop ---
print(f"Running {N_EPISODES_TO_RUN} evaluation episodes...")
all_rewards = []

for episode in range(N_EPISODES_TO_RUN):
    obs, info = eval_env.reset()
    terminated = False
    truncated = False
    episode_reward = 0
    step = 0
    print(f"--- Starting Episode {episode + 1} ---")

    while not terminated and not truncated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)

        # --- Explicitly call render ---
        eval_env.render()

        episode_reward += reward
        step += 1

        # Optional: Add a small delay to make watching easier
        # time.sleep(0.01)

        if terminated or truncated:
            print(f"Episode {episode + 1} finished after {step} steps.")
            print(f"Episode Reward: {episode_reward:.2f}")
            print(f"Bricks Left: {info.get('bricks_left', 'N/A')}")
            all_rewards.append(episode_reward)

print("--- Evaluation Complete ---")
if all_rewards:
    mean_reward = sum(all_rewards) / len(all_rewards)
    print(f"Average Reward over {len(all_rewards)} episodes: {mean_reward:.2f}")
else:
    print("No episodes were completed.")

# --- Close Environment ---
eval_env.close()
print("Evaluation environment closed.")
