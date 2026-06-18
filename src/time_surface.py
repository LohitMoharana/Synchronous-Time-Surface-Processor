# src/time_surface.py
import math
import numpy as np
from .config import GRID_X, GRID_Y, MAX_INTENSITY, TAU_US, CLOCK_PERIOD_US


class TimeSurfaceModel:
    def __init__(self):
        # We maintain two separate memory grids, exactly like our hardware plan
        self.on_surface = np.zeros((GRID_X, GRID_Y), dtype=float)
        self.off_surface = np.zeros((GRID_X, GRID_Y), dtype=float)

        # Track the timestamp of the last processed clock cycle
        self.last_update_time_us = 0.0

    def apply_decay(self, current_time_us):
        """Applies the exact HOTS exponential decay to the entire matrix."""
        delta_t = current_time_us - self.last_update_time_us

        # If no time has passed, no decay happens
        if delta_t <= 0:
            return

        # The biological exponential decay equation: e^(-delta_t / tau)
        decay_factor = math.exp(-delta_t / TAU_US)

        self.on_surface *= decay_factor
        self.off_surface *= decay_factor

        self.last_update_time_us = current_time_us

    def process_event(self, x, y, timestamp_us, polarity):
        """Updates the matrix based on a new biological event."""
        # 1. Snap the biological time to the rigid Hardware Clock Grid
        hardware_clock_cycle = int(timestamp_us / CLOCK_PERIOD_US)
        snapped_time_us = hardware_clock_cycle * CLOCK_PERIOD_US

        # 2. Decay the existing memory up to this new snapped time
        self.apply_decay(snapped_time_us)

        # 3. Inject the new event at Maximum Intensity (255)
        if polarity == 1:
            self.on_surface[x, y] = MAX_INTENSITY
        else:
            self.off_surface[x, y] = MAX_INTENSITY

    def get_hardware_matrices(self):
        """Converts the floating-point math into 8-bit integers for hardware comparison."""
        # Hardware can only store integers, so we floor the floating-point values
        on_hw = np.floor(self.on_surface).astype(int)
        off_hw = np.floor(self.off_surface).astype(int)

        # Ensure values don't drop below 0 due to float rounding
        on_hw = np.clip(on_hw, 0, MAX_INTENSITY)
        off_hw = np.clip(off_hw, 0, MAX_INTENSITY)

        return on_hw, off_hw