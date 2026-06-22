# truncate_stimulus.py
# Caps large gaps AND limits total events
MAX_WAIT = 15001   # allow up to 1ms gaps
                   # decay fires every 15000 cycles
                   # so at least 3 decay ticks between events
MAX_EVENTS = 200    # only first 5000 events

with open(r"C:/Users/MAHESH MISHRA/Documents/Synchronous-Time-Surface-Processor/attempt_two/synthetic_data.txt") as f_in:
    with open(r"C:/Users/MAHESH MISHRA/Documents/Synchronous-Time-Surface-Processor/attempt_two/synthetic_data_short.txt", "w") as f_out:
        for i, line in enumerate(f_in):
            if i >= MAX_EVENTS:
                break
            parts = line.strip().split()
            if len(parts) == 5:
                wait = min(int(parts[0]), MAX_WAIT)
                f_out.write(f"{wait} {parts[1]} "
                           f"{parts[2]} {parts[3]} "
                           f"{parts[4]}\n")

print(f"Done. Written {MAX_EVENTS} events with "
      f"gaps capped at {MAX_WAIT} cycles.")