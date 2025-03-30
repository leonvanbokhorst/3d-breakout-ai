import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gymnasium.utils.env_checker import check_env

# --- Configuration ---
LOG_DIR = "logs"
TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, "tensorboard_logs")
MODEL_SAVE_DIR = os.path.join(LOG_DIR, "models")
MODEL_NAME = "ppo_bouncy_breakout_v1"
TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 5

os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Import Environment HERE ---
    print("Attempting to import BouncyBreakoutEnv...")
    try:
        from main import BouncyBreakoutEnv

        print("Import successful.")
    except ImportError as e:
        print(f"Error: Could not import BouncyBreakoutEnv from main.py: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during BouncyBreakoutEnv import: {e}")
        exit()

    # --- Optional Environment Check ---
    # print("Checking single environment instance...")
    # try:
    #     single_env = BouncyBreakoutEnv()
    #     check_env(single_env)
    #     print("Environment check passed!")
    #     single_env.close()
    # except Exception as e:
    #      print(f"Environment check failed: {e}")
    #      exit()

    # --- Create VecEnv ---
    print(f"Creating VecEnv with N_ENVS={N_ENVS}...")
    # Use DummyVecEnv for N_ENVS=1, otherwise consider SubprocVecEnv
    if N_ENVS == 1:
        env = DummyVecEnv([lambda: BouncyBreakoutEnv()])
    else:
        # Use SubprocVecEnv for parallel execution
        # Ensure BouncyBreakoutEnv is pickleable if using SubprocVecEnv
        env = make_vec_env(
            lambda: BouncyBreakoutEnv(), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv
        )

    # --- Create Agent ---
    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        device="auto",
        ent_coef=0.01,
        # Add other hyperparameters here if needed (e.g., learning_rate)
    )

    # --- Train Agent (using model.learn) ---
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    print(f"Logging TensorBoard data to: {TENSORBOARD_LOG_DIR}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=MODEL_NAME,
        progress_bar=True,  # Show a progress bar
    )
    print("Training finished!")

    # --- Save Agent ---
    model_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_{TOTAL_TIMESTEPS}_steps")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # --- Close Environment ---
    env.close()
    print("Environment closed.")

    # --- Tensorboard Instructions ---
    print("To view training logs:")
    print(f"1. Activate your virtual environment")
    print(f"2. Run TensorBoard: tensorboard --logdir {TENSORBOARD_LOG_DIR}")
    print("3. Open the URL provided by TensorBoard in your browser.")
