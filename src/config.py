# src/config.py

# --- Hardware Bounds ---
GRID_X = 8                 # Sensor width
GRID_Y = 8                 # Sensor height
MAX_INTENSITY = 255         # 8-bit register maximum value

# --- Clock & Timing (Synchronous Translation) ---
CLOCK_FREQ_MHZ = 50.0                           # Target FPGA clock (50 MHz)
CLOCK_PERIOD_US = 1.0 / CLOCK_FREQ_MHZ          # 1 clock cycle = 0.02 microseconds

# --- Biological Decay Physics ---
TAU_US = 50_000.0             # Time constant for exponential decay (in microseconds)

# --- Clock-Gating (Activity Monitor) ---
IDLE_TIMEOUT_CYCLES = 100   # How many clock cycles to wait before going to sleep