# src/dataset_loader.py
import csv
import os
from .config import GRID_X, GRID_Y


def load_aer_dataset(filepath, original_width=128, original_height=128):
    """
    Loads a real-world Address-Event Representation (AER) dataset from a CSV.
    Downsamples the spatial coordinates to fit the target hardware grid.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    events = []

    # Calculate the compression ratio (e.g., 128 / 8 = 16 pixels per hardware bin)
    scale_x = original_width / GRID_X
    scale_y = original_height / GRID_Y

    print(f"[Dataset Loader] Reading {filepath}...")
    print(
        f"[Dataset Loader] Downsampling spatial resolution from {original_width}x{original_height} to {GRID_X}x{GRID_Y}...")

    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                raw_x = int(row['x'])
                raw_y = int(row['y'])

                # Spatial Downsampling (Binning)
                # Forces high-res coordinates into our 0-7 hardware range
                hw_x = min(int(raw_x / scale_x), GRID_X - 1)
                hw_y = min(int(raw_y / scale_y), GRID_Y - 1)

                events.append({
                    't': float(row['timestamp_us']),
                    'x': hw_x,
                    'y': hw_y,
                    'p': int(row['polarity'])
                })
            except ValueError:
                # Skip rows with malformed data/headers
                continue

    print(f"[Dataset Loader] Successfully loaded and compressed {len(events)} events.")
    return events