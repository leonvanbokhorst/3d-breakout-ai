import gymnasium as gym
from gymnasium import spaces
import numpy as np

# --- Move import pygame inside class ---
# import pygame # MOVED
import sys
import random
import math

# --- Constants / Rewards / States ---
# (... Constants defined without Pygame...)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BG_COLOR = (20, 160, 200)
BALL_COLOR = (255, 100, 100)
BALL_RADIUS = 20
INITIAL_BALL_SPEED_X = 3
INITIAL_BALL_SPEED_Y = 3
TARGET_BALL_SPEED_MAGNITUDE = math.sqrt(
    INITIAL_BALL_SPEED_X**2 + INITIAL_BALL_SPEED_Y**2
)
BALL_SPEED_DECAY_FACTOR = 0.999
BOUNCE_FACTOR_MIN = 0.95
BOUNCE_FACTOR_MAX = 1.05
ANGLE_NUDGE = 1.0
BRICK_WIDTH = 75
BRICK_HEIGHT = 20
BRICK_COLOR = (133, 133, 133)
BRICK_HEALTH = 1.0
BRICK_WOBBLE_DURATION = 30
BRICK_WOBBLE_AMOUNT = 3
BRICK_WOBBLE_SPEED = 0.5
BRICK_ROWS = 5
BRICK_COLS = SCREEN_WIDTH // (BRICK_WIDTH + 5)
BRICK_TOP_PADDING = 50
BRICK_HORIZONTAL_PADDING = (SCREEN_WIDTH - (BRICK_COLS * (BRICK_WIDTH + 5)) + 5) // 2
BRICK_VERTICAL_SPACING = 5
BRICK_HORIZONTAL_SPACING = 5
POGO_WIDTH = 100
POGO_HEIGHT = 20
POGO_COLOR = (50, 200, 50)
POGO_SPEED = 7
GRAVITY = 0.2
POGO_GROUND_Y = SCREEN_HEIGHT - POGO_HEIGHT - 10
POGO_DEFLECTION_FACTOR = 0.3
POGO_BOOST_FACTOR = 0.5
POGO_FLOOR_BOUNCE_DAMPING = 0.6
POGO_MIN_FLOOR_BOUNCE = 8.0
POGO_RECOIL_STRENGTH = 3.0
POGO_VERTICAL_FORCE = 0.5
REWARD_BRICK_HIT = 0.2
REWARD_BRICK_BROKEN = 5.0
REWARD_WIN = 50.0
REWARD_LOSE = -50.0
REWARD_POGO_HIT = 0.05
PENALTY_BALL_LOW = -0.1
REWARD_PER_STEP = 0.0
STATE_PLAYING = 0
STATE_GAME_OVER = 1
STATE_YOU_WIN = 2
FONT_SIZE = 48

# --- Global variable to hold Pygame reference after import ---
# This is a workaround for delayed import
pygame_module = None


class BouncyBreakoutEnv(gym.Env):
    """Custom Environment for Bouncy Breakout Bonanza with reward shaping."""

    metadata = {"render_modes": ["human", None], "render_fps": 60}

    # --- Define as Class Attribute ---
    FORCE_COOLDOWN_DURATION = 180  # Frames (3 seconds at 60 FPS)

    def __init__(self, render_mode=None):
        global pygame_module
        super().__init__()

        # --- Import and initialize Pygame HERE ---
        if pygame_module is None:
            try:
                import pygame

                pygame_module = pygame  # Assign to global variable
                # print("DEBUG: Pygame imported successfully in __init__") # Optional debug
            except ImportError as e:
                print(f"FATAL: Failed to import Pygame: {e}")
                raise e  # Re-raise error

        # Ensure font module is init (safe to call multiple times if pygame is init)
        try:
            pygame_module.font.init()
            self.font = pygame_module.font.Font(None, FONT_SIZE)
        except Exception as e:
            print(f"Warning: Failed to initialize font: {e}")
            self.font = None

        # --- Gym Setup ---
        # (... Action/Observation space definition ...)
        self.action_space = spaces.Discrete(5)
        num_core_features = 7
        num_brick_features = BRICK_ROWS * BRICK_COLS
        num_cooldown_features = 2
        obs_dims = num_core_features + num_brick_features + num_cooldown_features
        core_low = np.array([0, 0, -15, -15, 0, 0, -20], dtype=np.float32)
        core_high = np.array([1, 1, 15, 15, 1, 1, 20], dtype=np.float32)
        brick_low = np.zeros(num_brick_features, dtype=np.float32)
        brick_high = np.ones(num_brick_features, dtype=np.float32)
        cooldown_low = np.zeros(num_cooldown_features, dtype=np.float32)
        cooldown_high = np.ones(num_cooldown_features, dtype=np.float32)
        low_bounds = np.concatenate((core_low, brick_low, cooldown_low))
        high_bounds = np.concatenate((core_high, brick_high, cooldown_high))
        self.observation_space = spaces.Box(low_bounds, high_bounds, dtype=np.float32)

        # --- Rendering Setup (Conditional) ---
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        # Font is already attempted above
        if self.render_mode == "human":
            # If rendering, we need the full pygame init & display setup
            try:
                pygame_module.init()  # Full init needed for display/clock
                # Re-initialize font just in case full init reset something
                pygame_module.font.init()
                self.font = pygame_module.font.Font(None, FONT_SIZE)
            except Exception as e:
                print(f"Warning: Pygame full initialization failed: {e}")
            self._setup_display()

        # --- Internal State ---
        self._reset_internal_state()

    def _pygame(self):
        """Helper to access the imported Pygame module."""
        # Access the global variable directly.
        # __init__ should have imported and assigned it.
        if pygame_module is None:
            # This indicates a serious problem if it occurs after __init__
            # Let's raise an error for clarity during debugging.
            raise RuntimeError(
                "Pygame module accessed before initialization in __init__."
            )
        return pygame_module

    def _setup_display(self):
        """Initializes Pygame display, clock, and font ONLY for human rendering."""
        pg = self._pygame()  # Use helper to get pygame reference
        try:
            # Ensure full Pygame is init if rendering
            pg.init()

            # Initialize font module and load font (safe to re-init)
            pg.font.init()
            self.font = pg.font.Font(None, FONT_SIZE)

            # Initialize display and clock
            if self.screen is None:
                pg.display.init()
                self.screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pg.display.set_caption("Bouncy Breakout Bonanza - RL Env")
            if self.clock is None:
                self.clock = pg.time.Clock()
        except Exception as e:  # Catch potential pygame errors
            print(f"Error setting up display/font: {e}")
            self.render_mode = None

    # ... (Methods now need to use self._pygame() instead of pygame directly) ...

    def _reset_internal_state(self):  # Reset logic
        # ... (Reset vars) ...
        self.game_state = STATE_PLAYING
        self.frame_count = 0
        self.ball_x = 0
        self.ball_y = 0
        self.ball_speed_x = 0
        self.ball_speed_y = 0
        self.pogo_x = 0
        self.pogo_y = 0
        self.pogo_y_velocity = 0
        self.bricks = []
        self.push_up_cooldown = 0
        self.push_down_cooldown = 0

    def _get_obs(self):  # Observation calculation (no pygame needed)
        # ... (Core Features calculation) ...
        core_obs = np.array(
            [
                np.clip(self.ball_x / SCREEN_WIDTH, 0.0, 1.0),
                np.clip(self.ball_y / SCREEN_HEIGHT, 0.0, 1.0),
                self.ball_speed_x,
                self.ball_speed_y,
                np.clip(self.pogo_x / SCREEN_WIDTH, 0.0, 1.0),
                np.clip(self.pogo_y / SCREEN_HEIGHT, 0.0, 1.0),
                self.pogo_y_velocity,
            ],
            dtype=np.float32,
        )

        # ... (Brick Grid Features calculation) ...
        brick_grid = np.zeros((BRICK_ROWS, BRICK_COLS), dtype=np.float32)
        for b in self.bricks:
            r, c = b["row"], b["col"]
            if 0 <= r < BRICK_ROWS and 0 <= c < BRICK_COLS:
                brick_grid[r, c] = b["health"] / BRICK_HEALTH
        flattened_bricks = brick_grid.flatten()

        # --- Cooldown Features ---
        up_cooldown_norm = 0.0
        if self.FORCE_COOLDOWN_DURATION > 0:
            up_cooldown_norm = np.clip(
                self.push_up_cooldown / self.FORCE_COOLDOWN_DURATION, 0.0, 1.0
            )

        down_cooldown_norm = 0.0
        if self.FORCE_COOLDOWN_DURATION > 0:
            down_cooldown_norm = np.clip(
                self.push_down_cooldown / self.FORCE_COOLDOWN_DURATION, 0.0, 1.0
            )

        cooldown_obs = np.array(
            [up_cooldown_norm, down_cooldown_norm], dtype=np.float32
        )

        # --- Concatenate ---
        full_obs = np.concatenate((core_obs, flattened_bricks, cooldown_obs))

        # Clip observation to defined bounds as a safeguard
        full_obs = np.clip(
            full_obs, self.observation_space.low, self.observation_space.high
        )

        return full_obs

    def _get_info(self):
        return {"bricks_left": len(self.bricks)}  # Info dict

    def reset(self, seed=None, options=None):  # Reset method
        pg = self._pygame()  # Get pygame ref
        super().reset(seed=seed)
        self._reset_internal_state()
        # ... (Ball, Pogo setup) ...
        self.ball_x = SCREEN_WIDTH // 2
        self.ball_y = SCREEN_HEIGHT // 2
        self.ball_speed_x = INITIAL_BALL_SPEED_X * self.np_random.choice([-1, 1])
        self.ball_speed_y = INITIAL_BALL_SPEED_Y
        self.pogo_x = (SCREEN_WIDTH - POGO_WIDTH) // 2
        self.pogo_y = POGO_GROUND_Y
        self.pogo_y_velocity = 0
        self.bricks = []

        # Brick setup (health is now 1.0)
        for r in range(BRICK_ROWS):
            for c in range(BRICK_COLS):
                bx = BRICK_HORIZONTAL_PADDING + c * (
                    BRICK_WIDTH + BRICK_HORIZONTAL_SPACING
                )
                by = BRICK_TOP_PADDING + r * (BRICK_HEIGHT + BRICK_VERTICAL_SPACING)
                br = pg.Rect(bx, by, BRICK_WIDTH, BRICK_HEIGHT)
                bid = f"brick_{r}_{c}"
                self.bricks.append(
                    {
                        "id": bid,
                        "rect": br,
                        "health": BRICK_HEALTH,
                        "color": BRICK_COLOR,
                        "original_y": by,
                        "is_wobbling": False,
                        "wobble_timer": 0,
                        "original_x": bx,
                        "row": r,
                        "col": c,
                    }
                )
        return self._get_obs(), self._get_info()

    def step(self, action):
        pg = self._pygame()
        if self.game_state != STATE_PLAYING:
            return self._get_obs(), 0.0, True, False, self._get_info()

        # Initialize reward to 0.0 (no per-step penalty)
        reward = 0.0
        terminated = False
        truncated = False

        # --- Record initial brick count ---
        bricks_before_step = len(self.bricks)

        # --- Decrement Cooldowns ---
        if self.push_up_cooldown > 0:
            self.push_up_cooldown -= 1
        if self.push_down_cooldown > 0:
            self.push_down_cooldown -= 1

        attempted_up_force = False
        attempted_down_force = False

        # --- 1. Apply Action (with cooldown checks) ---
        if action == 0:
            self.pogo_x -= POGO_SPEED
        elif action == 1:
            self.pogo_x += POGO_SPEED
        elif action == 2:  # Apply Up Force
            attempted_up_force = True
            if self.push_up_cooldown <= 0:
                self.pogo_y_velocity -= POGO_VERTICAL_FORCE
        elif action == 3:  # Apply Down Force
            attempted_down_force = True
            if self.push_down_cooldown <= 0:
                self.pogo_y_velocity += POGO_VERTICAL_FORCE

        self.pogo_x = max(0, min(self.pogo_x, SCREEN_WIDTH - POGO_WIDTH))
        self.frame_count += 1

        # --- 2. Update Physics ---
        # Wobble Timer
        for b in self.bricks:
            if b["is_wobbling"]:
                b["wobble_timer"] -= 1
            if b["wobble_timer"] <= 0:
                b["is_wobbling"] = False

        # Pogo Physics
        original_pogo_y = self.pogo_y
        self.pogo_y_velocity += GRAVITY
        self.pogo_y += self.pogo_y_velocity

        # Enforce Vertical Limits & Bounce / Cooldown
        clamped_top = False
        clamped_bottom = False
        if self.pogo_y < POGO_MIN_FLOOR_BOUNCE:
            self.pogo_y = POGO_MIN_FLOOR_BOUNCE
            self.pogo_y_velocity = 0
            clamped_top = True
        elif self.pogo_y >= POGO_GROUND_Y:
            self.pogo_y = POGO_GROUND_Y
            clamped_bottom = True
            if self.pogo_y_velocity > 0:
                bv = self.pogo_y_velocity * -POGO_FLOOR_BOUNCE_DAMPING
                self.pogo_y_velocity = (
                    -POGO_MIN_FLOOR_BOUNCE if abs(bv) < POGO_MIN_FLOOR_BOUNCE else bv
                )
            else:
                self.pogo_y_velocity = 0
        if clamped_top and attempted_up_force:
            self.push_up_cooldown = self.FORCE_COOLDOWN_DURATION
        if (
            clamped_bottom
            and attempted_down_force
            and original_pogo_y < POGO_GROUND_Y + 1
        ):
            self.push_down_cooldown = self.FORCE_COOLDOWN_DURATION

        # Ball Movement & Decay
        self.ball_x += self.ball_speed_x
        self.ball_y += self.ball_speed_y
        cs_sq = self.ball_speed_x**2 + self.ball_speed_y**2
        ts_sq = TARGET_BALL_SPEED_MAGNITUDE**2
        if cs_sq > ts_sq * 1.05:
            decay = math.sqrt(ts_sq / cs_sq) * BALL_SPEED_DECAY_FACTOR
            self.ball_speed_x *= decay
            self.ball_speed_y *= decay

        # Ball Low Penalty
        if self.ball_y > self.pogo_y + POGO_HEIGHT:
            reward += PENALTY_BALL_LOW

        # --- 3. Collisions & Associated Rewards ---
        ball_rect = pg.Rect(
            self.ball_x - BALL_RADIUS,
            self.ball_y - BALL_RADIUS,
            BALL_RADIUS * 2,
            BALL_RADIUS * 2,
        )

        # Wall Bounces
        if self.ball_y - BALL_RADIUS < 0:
            self.ball_y = BALL_RADIUS
            self.ball_speed_y = abs(self.ball_speed_y) * self.np_random.uniform(
                BOUNCE_FACTOR_MIN, BOUNCE_FACTOR_MAX
            )
            self.ball_speed_x += self.np_random.uniform(-ANGLE_NUDGE, ANGLE_NUDGE)
        if self.ball_x - BALL_RADIUS < 0 or self.ball_x + BALL_RADIUS > SCREEN_WIDTH:
            self.ball_speed_x *= -1 * self.np_random.uniform(
                BOUNCE_FACTOR_MIN, BOUNCE_FACTOR_MAX
            )
            self.ball_speed_y += self.np_random.uniform(-ANGLE_NUDGE, ANGLE_NUDGE)
            if self.ball_x - BALL_RADIUS < 0:
                self.ball_x = BALL_RADIUS
            if self.ball_x + BALL_RADIUS > SCREEN_WIDTH:
                self.ball_x = SCREEN_WIDTH - BALL_RADIUS

        # Brick Collision (Simplified for 1-hit break)
        bricks_to_remove_indices = []
        for i, b in enumerate(self.bricks):
            br = b["rect"]
            cx = b["original_x"]
            if b["is_wobbling"]:
                cx += (
                    math.sin(self.frame_count * BRICK_WOBBLE_SPEED)
                    * BRICK_WOBBLE_AMOUNT
                )
            cbr = pg.Rect(cx, br.y, br.width, br.height)
            if ball_rect.colliderect(cbr):
                # Apply HIT reward immediately on collision
                reward += REWARD_BRICK_HIT

                # Mark for removal (health logic removed)
                bricks_to_remove_indices.append(i)
                b["is_wobbling"] = False  # Stop wobble

                # Bounce logic
                self.ball_speed_y *= -1 * self.np_random.uniform(
                    BOUNCE_FACTOR_MIN, BOUNCE_FACTOR_MAX
                )
                self.ball_speed_x += self.np_random.uniform(-ANGLE_NUDGE, ANGLE_NUDGE)
                if self.ball_speed_y > 0:
                    self.ball_y = cbr.bottom + BALL_RADIUS
                else:
                    self.ball_y = cbr.top - BALL_RADIUS

                # Removed health decrement and damaged state logic
                # if b['health'] >= 1.0 and oh >= 2.0:
                #     ...
                # elif b['health'] <= 0:
                #     ...
                break

        # Calculate Bricks Broken and Apply Reward
        bricks_broken_this_step = 0
        if bricks_to_remove_indices:
            for index in sorted(bricks_to_remove_indices, reverse=True):
                del self.bricks[index]
            bricks_broken_this_step = len(bricks_to_remove_indices)
            # Apply the large break reward here
            reward += bricks_broken_this_step * REWARD_BRICK_BROKEN

        # Pogo Collision
        # --- Define pogo_rect AFTER clamping pogo_y ---
        pogo_rect = pg.Rect(self.pogo_x, self.pogo_y, POGO_WIDTH, POGO_HEIGHT)
        if (
            self.ball_speed_y > 0
            and ball_rect.colliderect(pogo_rect)
            and self.ball_y + BALL_RADIUS > self.pogo_y - 5
        ):
            reward += REWARD_POGO_HIT
            hpn = np.clip(
                (self.ball_x - (self.pogo_x + POGO_WIDTH / 2)) / (POGO_WIDTH / 2),
                -1.0,
                1.0,
            )
            self.ball_speed_y = -abs(self.ball_speed_y) * self.np_random.uniform(
                BOUNCE_FACTOR_MIN, BOUNCE_FACTOR_MAX
            )
            if self.pogo_y_velocity < 0:
                self.ball_speed_y += self.pogo_y_velocity * POGO_BOOST_FACTOR
            self.ball_speed_x += hpn * abs(self.ball_speed_y) * POGO_DEFLECTION_FACTOR
            self.ball_y = self.pogo_y - BALL_RADIUS - 1
            self.pogo_y_velocity += POGO_RECOIL_STRENGTH

        # --- 4. Check Termination ---
        if not self.bricks:
            self.game_state = STATE_YOU_WIN
            reward += REWARD_WIN
            terminated = True
        if not terminated and self.ball_y - BALL_RADIUS > SCREEN_HEIGHT:
            self.game_state = STATE_GAME_OVER
            reward += REWARD_LOSE
            terminated = True

        # --- 5. Get Obs/Info, Render ---
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        # if terminated: print(f"DEBUG: step returning terminated=True (state={self.game_state}, reward={float(reward):.2f})")
        return observation, float(reward), terminated, truncated, info

    def _render_frame(self):  # Rendering method
        pg = self._pygame()  # Use helper
        if self.screen is None or self.clock is None:
            return

        # --- Process Pygame Events ---
        # Essential for keeping the window responsive, even if not handling input here
        pg.event.pump()

        self.screen.fill(BG_COLOR)
        current_game_state = self.game_state
        if current_game_state == STATE_PLAYING:
            if self.font is None:
                print("WARN: Font not loaded for rendering state")
                return
            pg.draw.circle(
                self.screen,
                BALL_COLOR,
                (int(self.ball_x), int(self.ball_y)),
                BALL_RADIUS,
            )  # Use pg.draw
            pogo_rect = pg.Rect(
                self.pogo_x, self.pogo_y, POGO_WIDTH, POGO_HEIGHT
            )  # Use pg.Rect
            pg.draw.rect(self.screen, POGO_COLOR, pogo_rect)  # Use pg.draw
            for b in self.bricks:
                dx = b["original_x"]
                if b["is_wobbling"]:
                    dx += (
                        math.sin(self.frame_count * BRICK_WOBBLE_SPEED)
                        * BRICK_WOBBLE_AMOUNT
                    )
                dr = pg.Rect(
                    dx, b["rect"].y, b["rect"].width, b["rect"].height
                )  # Use pg.Rect
                pg.draw.rect(self.screen, b["color"], dr)  # Use pg.draw
        elif current_game_state == STATE_GAME_OVER:
            if self.font is None:
                print("WARN: Font not loaded for rendering game over")
                return
            txt = self.font.render("GAME OVER!", True, (255, 0, 0))
            tr = txt.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(txt, tr)
        elif current_game_state == STATE_YOU_WIN:
            if self.font is None:
                print("WARN: Font not loaded for rendering win")
                return
            txt = self.font.render("YOU WIN! BOUNCY!", True, (0, 255, 0))
            tr = txt.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(txt, tr)
        pg.display.flip()
        self.clock.tick(self.metadata["render_fps"])  # Use pg.display

    def render(self):  # Public render method
        if self.render_mode == "human":
            # Ensure pygame is imported before setup
            pg = self._pygame()
            if self.font is None or self.screen is None or self.clock is None:
                self._setup_display()
            return self._render_frame()

    def close(self):  # Cleanup
        # Use the helper to ensure pygame module exists before quitting
        pg = pygame_module  # Access global directly here is okay for cleanup
        if pg is not None:
            if self.screen is not None:
                pg.display.quit()
                self.screen = None
                self.clock = None
            pg.quit()


# --- Example Usage ---
# (... No changes needed here ...)
if __name__ == "__main__":
    env = BouncyBreakoutEnv()
    obs, info = env.reset()
    print(f"Observation Space Shape (inc cooldowns): {env.observation_space.shape}")
    print(
        f"Example Obs (first 7 + last 12): {np.round(np.concatenate((obs[:7], obs[-12:])), 2)}"
    )  # Show core + end of bricks + cooldowns

    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0
    max_steps = 5000
    print(
        "\nRunning random headless episode with debug prints (Delayed Pygame Import)..."
    )
    while not terminated and not truncated and step_count < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
    print(f"\nFinished random headless episode after {step_count} steps.")
    print(
        f"Final State: {'Win' if env.game_state == STATE_YOU_WIN else ('Game Over' if env.game_state == STATE_GAME_OVER else 'Truncated')}"
    )
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Bricks Left: {info['bricks_left']}")
    env.close()
    print("\nEnvironment closed.")
