# Run this on one subject to see how many features are nonzero
import scipy.io as sio
import numpy as np
import sys, os
sys.path.insert(0, '.')
import src.config as config
from src.time_surface import TimeSurfaceModel

mat = sio.loadmat(r"C:\Users\MAHESH MISHRA\Documents\MyProjects\SynchEMGFilter\datasets\datasets\Ninapro-DB5\Ninapro-DB5\s1\S1_E1_A1.mat")
emg = mat['emg']
stimulus = mat['stimulus'].flatten()

# Take first active window
for i in range(len(stimulus)):
    if stimulus[i] != 0:
        break

window = emg[i:i+300, :]
last_v = window[0, :].copy()
events = []
for t_idx in range(1, 300):
    diffs = window[t_idx, :] - last_v
    for ch in range(16):
        if abs(diffs[ch]) >= 22.0:
            events.append({'t': float(t_idx*5000), 'x': ch%8,
                          'y': 2 if ch<8 else 5, 'p': 1})
            last_v[ch] = window[t_idx, ch]

ts = TimeSurfaceModel()
for ev in events:
    ts.process_event(ev['x'], ev['y'], ev['t'], ev['p'])

on_matrix, _ = ts.get_hardware_matrices()
print("Full 8x8 matrix:")
print(on_matrix)
print(f"\nNonzero pixels: {np.count_nonzero(on_matrix)} out of 64")
print(f"Active rows: {np.where(on_matrix.any(axis=1))[0]}")
print(f"Number of events generated: {len(events)}")