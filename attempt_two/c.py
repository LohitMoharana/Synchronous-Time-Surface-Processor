import os
path = "D:/RTL_OUT/power_sim.vcd"
if os.path.exists(path):
    size = os.path.getsize(path)
    print(f"VCD exists: {size} bytes")
else:
    print("VCD missing")

with open("D:/RTL_OUT/matrix_history.txt") as f:
    lines = f.readlines()
events = [l for l in lines if l.startswith("EVENT_INDEX")]
print(f"Events logged: {len(events)}")