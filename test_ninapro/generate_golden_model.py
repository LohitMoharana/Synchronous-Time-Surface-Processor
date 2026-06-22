import os
import sys
import scipy.io as sio

# --- REPO PATH INJECTION ---
REPO_DIR = r"C:\Users\MAHESH MISHRA\Documents\Synchronous-Time-Surface-Processor"
if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)

import src.config as config

# --- SIMULATION TIMELINE COMPRESSION ---
SIM_SPEEDUP = 1000.0
# Python Exponential Tau: 75us
config.TAU_US = 75_000.0 / SIM_SPEEDUP

from src.time_surface import TimeSurfaceModel

# --- EXACT PC PATHS ---
MAT_FILE = r"C:\Users\MAHESH MISHRA\Documents\MyProjects\SynchEMGFilter\datasets\datasets\Ninapro-DB5\Ninapro-DB5\s1\S1_E1_A1.mat"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VERILOG_STIMULUS_OUT = os.path.join(CURRENT_DIR, "synthetic_data.txt")
PYTHON_GOLDEN_OUT = os.path.join(CURRENT_DIR, "python_golden_matrix.txt")


def delta_modulator(emg_matrix, threshold=22.0):
    events, last_v = [], emg_matrix[0, :]
    for t_idx in range(1, emg_matrix.shape[0]):
        current_v = emg_matrix[t_idx, :]
        diffs = current_v - last_v
        for ch in range(16):
            if abs(diffs[ch]) >= threshold:
                x, y = (ch % 8), (2 if ch < 8 else 5)
                # Compress time by 1000x for simulator feasibility
                events.append({'t': float((t_idx * 5000.0) / SIM_SPEEDUP), 'x': x, 'y': y, 'p': 1})
                last_v[ch] = current_v[ch]
    return events


def custom_verilog_writer(events, filename):
    with open(filename, 'w') as f:
        last_cycle = 0
        for event in events:
            # 50MHz clock = 0.02us period
            current_cycle = int(event['t'] / 0.02)
            wait_cycles = max(0, current_cycle - last_cycle)
            last_cycle = current_cycle
            f.write(f"{wait_cycles} {event['p']} {event['x']} {event['y']}\n")


def generate_golden_equivalence():
    print("--- GENERATING EXPONENTIAL PYTHON vs DISCRETE VERILOG ---")
    mat_data = sio.loadmat(MAT_FILE)

    # Trim the dataset to the first 20,000 samples for a quick comparison
    emg_subset = mat_data['emg'][:20000, :]

    events = delta_modulator(emg_subset)
    custom_verilog_writer(events, VERILOG_STIMULUS_OUT)

    # Uses the ORIGINAL exponential decay math from the repo
    ts = TimeSurfaceModel()

    with open(PYTHON_GOLDEN_OUT, 'w') as f:
        for i, ev in enumerate(events):
            ts.process_event(ev['x'], ev['y'], ev['t'], ev['p'])
            on_matrix, _ = ts.get_hardware_matrices()

            f.write(f"EVENT_INDEX: {i} | TIME_US: {ev['t']}\n")

            # Correctly map Y to rows (printing top row 7 down to bottom row 0)
            # Correctly map X to columns (left to right 0 to 7)
            for y_row in range(7, -1, -1):
                row_vals = [str(int(on_matrix[x_col, y_row])) for x_col in range(8)]
                f.write(" ".join(row_vals) + "\n")
            f.write("---\n")

    print("[Done] Python Golden Model & Verilog Stimulus Generated.")


if __name__ == "__main__":
    generate_golden_equivalence()