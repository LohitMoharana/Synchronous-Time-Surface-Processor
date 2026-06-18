# src/hardware_writer.py
import os
from .config import CLOCK_PERIOD_US


def export_to_verilog_stimulus(events, filename="output/stimulus.txt"):
    """
    Converts the Python event list into a Verilog-readable text file.
    Format per line: [Wait_Cycles] [Polarity] [X] [Y]
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        last_cycle = 0

        for event in events:
            # 1. Calculate which clock cycle this event falls on
            current_cycle = int(event['t'] / CLOCK_PERIOD_US)

            # 2. Calculate how many clock ticks to wait since the *last* event
            cycles_to_wait = current_cycle - last_cycle

            # If multiple events happen in the exact same clock cycle, wait time is 0
            cycles_to_wait = max(0, cycles_to_wait)
            last_cycle = current_cycle

            # 3. Write to file (Format: Wait_Cycles Polarity X Y)
            f.write(f"{cycles_to_wait} {event['p']} {event['x']} {event['y']}\n")

    print(f"[Success] Exported {len(events)} events to {filename}")