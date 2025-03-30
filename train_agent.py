import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.callbacks import (
    EvalCallback,
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

# --- Configuration ---
LOG_DIR = "logs"
TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, "tensorboard_logs")
MODEL_SAVE_DIR = os.path.join(LOG_DIR, "models")
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "best_model")
EVAL_LOG_PATH = os.path.join(LOG_DIR, "eval_logs")
MODEL_NAME = "ppo_bouncy_breakout_v1"
TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 5
EVAL_FREQ = 50_000
N_EVAL_EPISODES = 20

os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(EVAL_LOG_PATH, exist_ok=True)

# Define the expected final model path (adjust naming if needed)
# Note: SB3 often saves intermediate checkpoints too, but we'll target the final one for simplicity
# You might need more sophisticated logic to find the *latest* checkpoint if training is interrupted often.
# Let's assume we save with cumulative steps in the name for resuming.
# We'll construct the path based on the *current progress* later.
# For now, define a base path pattern.
model_path_pattern = os.path.join(
    MODEL_SAVE_DIR, f"{MODEL_NAME}_{{timesteps}}_steps.zip"
)  # Placeholder for timesteps


# --- Custom Callback for Printing Eval Results ---
class PrintEvalCallback(EvalCallback):
    """
    Callback that extends EvalCallback to print evaluation results to the console.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        # Call the parent's _on_step method to perform evaluation
        continue_training = super()._on_step()

        # Check if an evaluation was performed in this step
        # self.last_mean_reward is updated by the parent EvalCallback
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.last_mean_reward is not None and self.last_mean_reward != -np.inf:
                print(
                    f"Eval@{self.num_timesteps}: Mean reward={self.last_mean_reward:.2f} (Best: {self.best_mean_reward:.2f})"
                )
            else:
                # Should not happen if eval runs, but handle gracefully
                print(
                    f"Eval@{self.num_timesteps}: Evaluation ran, but no reward recorded?"
                )

        return continue_training


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

    # --- Create VecEnv for Training ---
    print(f"Creating Training VecEnv with N_ENVS={N_ENVS}...")
    if N_ENVS == 1:
        env = DummyVecEnv([lambda: BouncyBreakoutEnv()])
    else:
        env = make_vec_env(
            lambda: BouncyBreakoutEnv(), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv
        )

    # --- Create Separate VecEnv for Evaluation ---
    print("Creating Evaluation VecEnv...")
    eval_env = BouncyBreakoutEnv()
    eval_env = Monitor(eval_env)

    # --- Setup Custom EvalCallback ---
    eval_callback = PrintEvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_SAVE_PATH,
        log_path=EVAL_LOG_PATH,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    # --- Load Existing Model or Create New One ---
    # Prioritize loading the best model saved by EvalCallback
    best_model_path_full = os.path.join(BEST_MODEL_SAVE_PATH, "best_model.zip")
    model = None  # Initialize model to None

    if os.path.exists(best_model_path_full):
        print(f"Loading best model from: {best_model_path_full}")
        try:
            model = PPO.load(
                best_model_path_full,
                env=env,  # Pass the training env
                device="auto",
                tensorboard_log=TENSORBOARD_LOG_DIR,  # Re-specify log dir
            )
            print(
                f"Successfully loaded model. Current timesteps: {model.num_timesteps}"
            )
        except Exception as e:
            print(f"Error loading model: {e}. Creating a new one.")
            # Ensure model is None if loading failed
            model = None

    # If the best model wasn't found or failed to load, create a new one
    if model is None:
        print("Best model not found or failed to load. Creating new PPO agent...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,  # Set back to 1 initially, can adjust later if needed
            tensorboard_log=TENSORBOARD_LOG_DIR,
            device="auto",
            learning_rate=0.0003,  # Example: Use default or a specific LR
            ent_coef=0.01,
            # Add other hyperparameters here if needed
        )
        print("New model created.")

    # --- Train Agent ---
    # The model.learn call will use model.num_timesteps internally
    # if reset_num_timesteps=False (which it is)
    print(f"Starting training loop. Target total timesteps: {TOTAL_TIMESTEPS}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=MODEL_NAME,
        reset_num_timesteps=False,  # IMPORTANT: Ensures resuming from loaded model's steps
        progress_bar=True,
        callback=eval_callback,  # Use our evaluation callback
    )
    print("Training finished or target timesteps reached!")

    # --- Save Final Agent (Optional, EvalCallback saves the best) ---
    final_model_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_final")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}.zip")

    # --- Close Environments ---
    env.close()
    eval_env.close()
    print("\nEnvironments closed.")

    # --- Tensorboard Instructions ---
    print("\nTo view training logs:")
    print(f"1. Activate your virtual environment")
    print(f"2. Run TensorBoard: tensorboard --logdir {TENSORBOARD_LOG_DIR}")
    print("3. Open the URL provided by TensorBoard in your browser.")
