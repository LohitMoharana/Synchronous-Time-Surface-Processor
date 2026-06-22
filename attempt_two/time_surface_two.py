# src/time_surface_two.py
import math
import numpy as np
from .config_two import (GRID_X, GRID_Y, MAX_INTENSITY, TAU_US,
                         CLOCK_PERIOD_US, RTL_DECAY_FACTOR,
                         RTL_DECAY_PERIOD_CYCLES)


class TimeSurfaceModel:
    """
    Software golden reference model.
    Uses continuous exponential decay matching HOTS formulation.
    This is the ideal baseline - what perfect infinite-precision
    hardware would compute.
    """
    def __init__(self):
        self.on_surface = np.zeros((GRID_X, GRID_Y), dtype=float)
        self.off_surface = np.zeros((GRID_X, GRID_Y), dtype=float)
        self.last_update_time_us = 0.0

    def apply_decay(self, current_time_us):
        delta_t = current_time_us - self.last_update_time_us
        if delta_t <= 0:
            return
        decay_factor = math.exp(-delta_t / TAU_US)
        self.on_surface *= decay_factor
        self.off_surface *= decay_factor
        self.last_update_time_us = current_time_us

    def process_event(self, x, y, timestamp_us, polarity):
        hardware_clock_cycle = int(timestamp_us / CLOCK_PERIOD_US)
        snapped_time_us = hardware_clock_cycle * CLOCK_PERIOD_US
        self.apply_decay(snapped_time_us)
        if polarity == 1:
            self.on_surface[x, y] = MAX_INTENSITY
        else:
            self.off_surface[x, y] = MAX_INTENSITY

    def get_hardware_matrices(self):
        on_hw = np.clip(
            np.floor(self.on_surface).astype(int), 0, MAX_INTENSITY)
        off_hw = np.clip(
            np.floor(self.off_surface).astype(int), 0, MAX_INTENSITY)
        return on_hw, off_hw


class TimeSurfaceModelRTL:
    """
    RTL-equivalent model.
    Uses fixed-point multiply-shift decay matching the Verilog exactly.
    DECAY_FACTOR=255, DECAY_PERIOD=15000 cycles (300us at 50MHz).
    Internal storage is 16-bit fixed point (value << 8).
    Output is upper 8 bits, matching flat_surface[15:8] in RTL.
    Compare accuracy of this vs TimeSurfaceModel to get quantisation
    loss for the paper.
    """
    def __init__(self):
        # 16-bit internal storage matching RTL surface_fp registers
        self.on_surface_fp = np.zeros(
            (GRID_X, GRID_Y), dtype=np.int32)
        self.off_surface_fp = np.zeros(
            (GRID_X, GRID_Y), dtype=np.int32)
        self.last_decay_cycle = 0

    def _apply_decay_ticks(self, n_ticks):
        for _ in range(n_ticks):
            self.on_surface_fp = (
                self.on_surface_fp * RTL_DECAY_FACTOR) >> 8
            self.off_surface_fp = (
                self.off_surface_fp * RTL_DECAY_FACTOR) >> 8

    def process_event(self, x, y, timestamp_us, polarity):
        event_cycle = int(timestamp_us / CLOCK_PERIOD_US)
        cycles_elapsed = event_cycle - self.last_decay_cycle
        ticks = cycles_elapsed // RTL_DECAY_PERIOD_CYCLES

        if ticks > 0:
            self._apply_decay_ticks(ticks)
            self.last_decay_cycle += ticks * RTL_DECAY_PERIOD_CYCLES

        # Injection: 255 in fixed-point = 255 << 8 = 65280
        if polarity == 1:
            self.on_surface_fp[x, y] = 255 << 8
        else:
            self.off_surface_fp[x, y] = 255 << 8

    def get_hardware_matrices(self):
        on_hw = np.clip(
            self.on_surface_fp >> 8, 0, MAX_INTENSITY).astype(int)
        off_hw = np.clip(
            self.off_surface_fp >> 8, 0, MAX_INTENSITY).astype(int)
        return on_hw, off_hw