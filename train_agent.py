import os
import gymnasium as gym
import numpy as np
import time
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
TOTAL_TIMESTEPS = 100_000_000
N_ENVS = 64
EVAL_FREQ = 100_000
N_EVAL_EPISODES = 50
# Stable Baselines recommends CPU for MlpPolicy (not CNN)
# See warning: https://github.com/DLR-RM/stable-baselines3/issues/1245
DEVICE = "cpu"  # Using CPU as recommended for PPO with MlpPolicy

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


# --- Custom Callback for Printing Eval Results & Early Stopping ---
class PrintEvalCallback(EvalCallback):
    """
    Callback that extends EvalCallback to print evaluation results to the console
    and implement custom early stopping based on patience.
    """

    def __init__(self, *args, patience=10, **kwargs):
        # Pass verbose=1 to parent to ensure it logs best model saving
        super().__init__(*args, **kwargs, verbose=1)
        self.patience = patience
        self.no_improvement_evals = 0
        print(
            f"[PrintEvalCallback] Early stopping enabled with patience={self.patience}"
        )

    def _on_step(self) -> bool:
        # Store best mean reward *before* evaluation
        best_reward_before = self.best_mean_reward

        # Call the parent's _on_step method to perform evaluation
        # This also updates self.best_mean_reward
        continue_training = super()._on_step()

        # If the parent callback decided to stop (e.g., due to NaN), respect that
        if not continue_training:
            return False

        # Check if an evaluation was performed in this step
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.last_mean_reward is not None and self.last_mean_reward != -np.inf:
                print(
                    f"Eval@{self.num_timesteps}: Mean reward={self.last_mean_reward:.2f} (Best: {self.best_mean_reward:.2f})"
                )
                # --- Early Stopping Logic ---
                # Check if the best mean reward improved during the parent call
                if self.best_mean_reward > best_reward_before:
                    # Reset counter if new best model found
                    self.no_improvement_evals = 0
                    print(
                        f"[PrintEvalCallback] New best mean reward {self.best_mean_reward:.2f} > {best_reward_before:.2f}. Resetting patience counter."
                    )
                else:
                    # Increment counter if no improvement
                    self.no_improvement_evals += 1
                    print(
                        f"[PrintEvalCallback] No improvement from best {self.best_mean_reward:.2f}. Patience: {self.no_improvement_evals}/{self.patience}."
                    )

                # Check if patience is exceeded
                if self.no_improvement_evals >= self.patience:
                    print(
                        f"[PrintEvalCallback] Stopping training early: No improvement in mean reward for {self.patience} evaluations."
                    )
                    return False  # Signal to stop training
            else:
                print(
                    f"Eval@{self.num_timesteps}: Evaluation ran, but no reward recorded?"
                )

        return True  # Continue training by default


# --- Speed Monitor Callback ---
class SpeedMonitorCallback(BaseCallback):
    """
    Callback for monitoring and displaying training speed metrics.
    """
    def __init__(self, verbose=0, display_interval=10000):
        super().__init__(verbose)
        self.display_interval = display_interval
        self.start_time = None
        self.last_display_time = None
        self.last_display_timesteps = 0
        self.total_steps = 0
        self.episode_count = 0
        # For recording speed data over time
        self.speed_log = []
        # For CPU monitoring
        try:
            import psutil
            self.psutil_available = True
            print("CPU monitoring enabled")
        except ImportError:
            self.psutil_available = False
            print("Note: Install psutil for CPU monitoring")
        
    def _on_training_start(self):
        self.start_time = time.time()
        self.last_display_time = self.start_time
        # Reset these in case callback is reused
        self.speed_log = []
        self.total_steps = 0
        self.episode_count = 0
        
    def _on_step(self):
        # Update counters
        self.total_steps += 1
        
        # Check if enough time has elapsed to display metrics
        current_time = time.time()
        timesteps = self.model.num_timesteps
        elapsed = current_time - self.last_display_time
        
        if timesteps % self.display_interval == 0 and elapsed > 0:
            # Calculate speeds
            steps_since_last = timesteps - self.last_display_timesteps
            iterations_per_second = steps_since_last / elapsed
            
            # Calculate ETA
            total_elapsed = current_time - self.start_time
            progress_percentage = timesteps / TOTAL_TIMESTEPS * 100
            eta_seconds = (TOTAL_TIMESTEPS - timesteps) / iterations_per_second if iterations_per_second > 0 else float('inf')
            
            # Format ETA nicely
            if eta_seconds < float('inf'):
                eta_hours = int(eta_seconds // 3600)
                eta_minutes = int((eta_seconds % 3600) // 60)
                eta_seconds = int(eta_seconds % 60)
                eta_str = f"{eta_hours:02d}h:{eta_minutes:02d}m:{eta_seconds:02d}s"
            else:
                eta_str = "unknown"
            
            # Get CPU usage if available
            cpu_info = ""
            if self.psutil_available:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_used = psutil.virtual_memory().percent
                cpu_info = f" | CPU: {cpu_percent:.1f}% | RAM: {memory_used:.1f}%"
            
            # Log to console
            print(f"⚡ Speed: {iterations_per_second:.1f} it/s | "
                  f"Progress: {progress_percentage:.1f}%{cpu_info} | "
                  f"Total Steps: {timesteps:,} | "
                  f"ETA: {eta_str}")
            
            # Store for later analysis
            self.speed_log.append({
                'timesteps': timesteps,
                'time': current_time - self.start_time,
                'speed': iterations_per_second,
            })
            
            # Update for next interval
            self.last_display_time = current_time
            self.last_display_timesteps = timesteps
        
        return True  # Continue training
    
    def get_speed_stats(self):
        """Return speed statistics at the end of training"""
        if not self.speed_log:
            return None
        
        speeds = [entry['speed'] for entry in self.speed_log]
        return {
            'mean_speed': sum(speeds) / len(speeds) if speeds else 0,
            'max_speed': max(speeds) if speeds else 0,
            'min_speed': min(speeds) if speeds else 0,
            'speeds': speeds,
            'total_time': time.time() - self.start_time if self.start_time else 0,
        }
    
    def _on_training_end(self):
        """Print final speed statistics"""
        stats = self.get_speed_stats()
        if stats:
            print("\n----- Training Speed Summary -----")
            print(f"Mean Speed: {stats['mean_speed']:.1f} it/s")
            print(f"Max Speed: {stats['max_speed']:.1f} it/s")
            print(f"Min Speed: {stats['min_speed']:.1f} it/s")
            print(f"Total Training Time: {stats['total_time']:.1f} seconds")
            print("----------------------------------")


# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Boost Process Priority (macOS only) ---
    try:
        import platform
        system = platform.system()
        if system == "Darwin":  # macOS
            import os
            os.nice(-10)  # Boost priority on macOS (-20 to 20, lower is higher priority)
            print("Boosted process priority on macOS")
        elif system == "Windows":  # Windows
            import psutil
            import os
            # Set process priority to high on Windows
            p = psutil.Process(os.getpid())
            p.nice(psutil.HIGH_PRIORITY_CLASS)
            print("Boosted process priority on Windows to HIGH_PRIORITY_CLASS")
    except Exception as e:
        print(f"Note: Couldn't boost process priority: {e}")

    # --- CUDA Diagnostics ---
    print("\n----- PyTorch CUDA Diagnostics -----")
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        # Test a small tensor operation on GPU
        test_tensor = torch.tensor([1.0, 2.0, 3.0]).to('cuda')
        print(f"Test tensor on GPU: {test_tensor.device}")
        print("\nNOTE: Even though CUDA is available, we're using CPU for training")
        print("because Stable Baselines3 recommends CPU for PPO with MlpPolicy.")
        print("GPU would actually be slower for this specific algorithm+policy combination.")
        print("See: https://github.com/DLR-RM/stable-baselines3/issues/1245")
    else:
        print("WARNING: CUDA is not available! Training will be on CPU.")
    print("-------------------------------\n")

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

    # --- Setup Callbacks ---
    print("Setting up callbacks...")
    # Performance monitor
    speed_callback = SpeedMonitorCallback(display_interval=20000)  # Increased for high-speed training
    
    # Evaluation callback
    eval_callback = PrintEvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_SAVE_PATH,
        log_path=EVAL_LOG_PATH,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        patience=25,  # Pass patience to our custom __init__
    )
    
    # Checkpoint callback to save intermediate models
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000, EVAL_FREQ),  # Save at least as often as eval
        save_path=MODEL_SAVE_DIR,
        name_prefix=MODEL_NAME,
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # Combine all callbacks
    callbacks = [speed_callback, eval_callback, checkpoint_callback]

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
                device=DEVICE,  # Use GPU when loading the model
                tensorboard_log=TENSORBOARD_LOG_DIR,  # Re-specify log dir
            )
            print(
                f"Successfully loaded model on {DEVICE}. Current timesteps: {model.num_timesteps}"
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
            verbose=1,
            tensorboard_log=TENSORBOARD_LOG_DIR,
            device=DEVICE,
            # Hyperparameter tuning for maximum speed on Win11
            n_steps=1024,     # Reduced from 2048 to process more frequently with more envs
            batch_size=1024,  # Doubled from 512 for better Win11 throughput
            n_epochs=5,       # Reduced from 10 but still effective
            ent_coef=0.01,    # Maintain entropy coefficient 
            learning_rate=3e-4, # Explicit learning rate
            # Optimize network size for speed while maintaining capability
            policy_kwargs={"net_arch": [dict(pi=[64, 64], vf=[64, 64])]}
        )
        print(f"New model created on device: {DEVICE} with hyper-optimized batch size for Win11")

    # --- Train Agent ---
    print(
        f"Starting training loop on {DEVICE}. Target total timesteps: {TOTAL_TIMESTEPS}. Early stopping enabled."
    )
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            tb_log_name=MODEL_NAME,
            reset_num_timesteps=False,  # IMPORTANT: Ensures resuming from loaded model's steps
            progress_bar=True,
            callback=callbacks,  # Use our evaluation callback
        )
        print("Training finished or target timesteps reached!")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # --- Save Final Agent (Optional, EvalCallback saves the best) ---
        final_model_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_final")
        if model:  # Ensure model exists before saving
            model.save(final_model_path)
            print(f"Final model saved to {final_model_path}.zip")
        else:
            print("No model object available to save.")

        # --- Close Environments ---
        try:
            env.close()
            if eval_env:
                eval_env.close()
            print("Environments closed.")
        except Exception as e:
            print(f"Error closing environments: {e}")

    # --- Tensorboard Instructions ---
    print("\nTo view training logs:")
    print(f"1. Activate your virtual environment")
    print(f"2. Run TensorBoard: tensorboard --logdir {TENSORBOARD_LOG_DIR}")
    print("3. Open the URL provided by TensorBoard in your browser.")

    print(f"Training script finished. Used device: {DEVICE}")
