# src/synthetic_emg.py
import random
from .config import GRID_X, GRID_Y

# Add this function to your existing src/synthetic_emg.py
import random


def inject_sensor_noise(events, noise_ratio=0.10):
    """
    Injects random biological noise artifacts into the event stream.
    noise_ratio: The percentage of total events that will be random noise.
    """
    from .config import GRID_X, GRID_Y

    num_noise_events = int(len(events) * noise_ratio)
    max_time = events[-1]['t'] if events else 10000.0

    for _ in range(num_noise_events):
        events.append({
            'x': random.randint(0, GRID_X - 1),
            'y': random.randint(0, GRID_Y - 1),
            't': random.uniform(0, max_time),  # Happens at a completely random time
            'p': random.choice([0, 1])  # Random polarity
        })

    # Re-sort the events chronologically because hardware must process time linearly
    events = sorted(events, key=lambda k: k['t'])
    return events

def generate_left_to_right_swipe():
    """
    Simulates a muscle contraction wave moving left-to-right across the 8x8 grid.
    Returns a list of events: [{'x': int, 'y': int, 't': float, 'p': int}, ...]
    """
    events = []
    current_time_us = 0.0

    # The wave moves across the X-axis (columns 0 to 7)
    for x in range(GRID_X):
        # A burst of events happens in the current column
        for y in range(GRID_Y):
            # Add a little random noise to the timing so it looks biological, not robotic
            time_jitter = random.uniform(5.0, 15.0)
            current_time_us += time_jitter

            # Polarity 1 (ON) represents the muscle actively flexing
            events.append({
                'x': x,
                'y': y,
                't': current_time_us,
                'p': 1
            })

        # Add a delay as the physical wave travels to the next column of sensors
        current_time_us += 200.0

    return events


if __name__ == "__main__":
    # Quick test to see the data format
    sample_data = generate_left_to_right_swipe()
    print(f"Generated {len(sample_data)} biological events.")
    print(f"First event: {sample_data[0]}")
    print(f"Last event:  {sample_data[-1]}")