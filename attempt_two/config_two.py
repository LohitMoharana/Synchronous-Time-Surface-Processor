# src/config_two.py

# --- Hardware Bounds ---
GRID_X = 8
GRID_Y = 8
MAX_INTENSITY = 255

# --- Clock & Timing ---
CLOCK_FREQ_MHZ = 50.0
CLOCK_PERIOD_US = 1.0 / CLOCK_FREQ_MHZ   # 0.02 microseconds per cycle

# --- Biological Decay Physics ---
# Effective tau = 76,650us (~76.6ms) matching RTL DECAY_FACTOR=255, DECAY_PERIOD=15000
TAU_US = 75_000.0

# --- RTL Hardware Parameters - must match Verilog exactly ---
RTL_DECAY_FACTOR = 255        # 255/256 per tick
RTL_DECAY_PERIOD_CYCLES = 15000  # one tick every 15000 cycles = 300us at 50MHz

# --- Activity Monitor ---
IDLE_TIMEOUT_CYCLES = 5000    # 100us at 50MHz

# --- Ninapro DB5 ---
EMG_SAMPLING_RATE_HZ = 200
EMG_SAMPLE_PERIOD_US = 5000.0   # 1/200Hz in microseconds
EMG_CHANNELS = 16
EMG_DELTA_THRESHOLD = 22.0