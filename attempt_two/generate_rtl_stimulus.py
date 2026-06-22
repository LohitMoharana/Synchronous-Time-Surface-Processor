# generate_rtl_stimulus.py
# Place this in your project root
# Run this FIRST before anything else

import scipy.io as sio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import src.config as config

MAT_FILE = r"C:\Users\MAHESH MISHRA\Documents\MyProjects\SynchEMGFilter\datasets\datasets\Ninapro-DB5\Ninapro-DB5\s1\S1_E1_A1.mat"
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_data.txt")

print("Loading Ninapro S1 E1 A1...")
mat = sio.loadmat(MAT_FILE)
emg = mat['emg']
stimulus = mat['stimulus'].flatten()

print(f"EMG shape: {emg.shape}")
print(f"Generating events...")

events = []
last_v = emg[0, :].copy()

for t_idx in range(1, emg.shape[0]):
    current_v = emg[t_idx, :]
    diffs = current_v - last_v
    for ch in range(16):
        if abs(diffs[ch]) >= config.EMG_DELTA_THRESHOLD:
            events.append({
                't': float(t_idx * config.EMG_SAMPLE_PERIOD_US),
                'x': ch % 8,
                'y': 2 if ch < 8 else 5,
                'p': 1 if diffs[ch] > 0 else 0
            })
            last_v[ch] = emg[t_idx, ch]

print(f"Total events: {len(events)}")

# Also record trial boundaries for window_start signal
# A new trial starts when gap between events exceeds 50ms
TRIAL_GAP_US = 50_000.0  # 50ms gap = new trial
trial_boundaries = set()
for i in range(1, len(events)):
    if events[i]['t'] - events[i-1]['t'] > TRIAL_GAP_US:
        trial_boundaries.add(i)

print(f"Trial boundaries detected: {len(trial_boundaries)}")

# Write stimulus file
# Format: wait_cycles polarity x y window_start
with open(OUT_FILE, 'w') as f:
    last_cycle = 0
    for idx, ev in enumerate(events):
        current_cycle = int(ev['t'] / config.CLOCK_PERIOD_US)
        wait_cycles = max(0, current_cycle - last_cycle)
        last_cycle = current_cycle
        is_boundary = 1 if idx in trial_boundaries else 0
        f.write(f"{wait_cycles} {ev['p']} {ev['x']} {ev['y']} {is_boundary}\n")

print(f"Written to: {OUT_FILE}")
print(f"First 5 lines:")
with open(OUT_FILE) as f:
    for _ in range(5):
        print(f"  {f.readline().strip()}")

# Also print sleep fraction stats
total_cycles = 0
sleep_cycles = 0
awake_cycles = 0
last_cycle = 0
TIMEOUT = config.IDLE_TIMEOUT_CYCLES

for ev in events:
    current_cycle = int(ev['t'] / config.CLOCK_PERIOD_US)
    gap = current_cycle - last_cycle
    if gap > TIMEOUT:
        awake_cycles += TIMEOUT
        sleep_cycles += (gap - TIMEOUT)
    else:
        awake_cycles += gap
    total_cycles += gap
    last_cycle = current_cycle

print(f"\nSleep fraction analysis:")
print(f"Total cycles: {total_cycles:,}")
print(f"Awake: {awake_cycles:,} ({awake_cycles/total_cycles*100:.1f}%)")
print(f"Sleep: {sleep_cycles:,} ({sleep_cycles/total_cycles*100:.1f}%)")
print(f"Estimated power reduction from gating: "
      f"{sleep_cycles/total_cycles*100:.1f}%")

# Generate trial boundaries from stimulus signal directly
# A boundary occurs when stimulus transitions from 0 to nonzero
trial_boundaries_by_sample = set()
for i in range(1, len(stimulus)):
    if stimulus[i-1] == 0 and stimulus[i] != 0:
        # Convert sample index to event index
        boundary_time_us = i * config.EMG_SAMPLE_PERIOD_US
        trial_boundaries_by_sample.add(boundary_time_us)

print(f"True trial starts from stimulus: "
      f"{len(trial_boundaries_by_sample)}")

# Now when writing events, mark boundary if event time
# falls within one sample period of a trial start
trial_boundaries = set()
for idx, ev in enumerate(events):
    for boundary_us in trial_boundaries_by_sample:
        if abs(ev['t'] - boundary_us) < config.EMG_SAMPLE_PERIOD_US:
            trial_boundaries.add(idx)
            break