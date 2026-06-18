# # # main.py
# # from src.synthetic_emg import generate_left_to_right_swipe
# # from src.time_surface import TimeSurfaceModel
# # from src.hardware_writer import export_to_verilog_stimulus
# #
# #
# # def main():
# #     print("--- Starting MIoT Time-Surface MVP ---")
# #
# #     # 1. Generate the synthetic biological data (Left-to-Right Swipe)
# #     print("Generating synthetic EMG wave...")
# #     events = generate_left_to_right_swipe()
# #
# #     # 2. Export the raw events to the Verilog stimulus file
# #     export_to_verilog_stimulus(events, "output/stimulus.txt")
# #
# #     # 3. Run the Python Golden Model
# #     print("Running data through the Mathematical Golden Model...")
# #     ts_model = TimeSurfaceModel()
# #
# #     for event in events:
# #         ts_model.process_event(event['x'], event['y'], event['t'], event['p'])
# #
# #     # 4. Extract the final heatmaps formatted for hardware comparison
# #     on_matrix, off_matrix = ts_model.get_hardware_matrices()
# #
# #     # 5. Print the target output
# #     print("\n==================================================")
# #     print("  FINAL ON-SURFACE MATRIX (EXPECTED VERILOG OUTPUT)")
# #     print("==================================================")
# #     # Print it nicely formatted so you can actually see the "Comet Tail" gradient
# #     for row in on_matrix:
# #         formatted_row = " ".join(f"{val:3}" for val in row)
# #         print(f"[ {formatted_row} ]")
# #     print("==================================================")
# #     print("If your Verilog ts_matrix_out matches these numbers exactly, your RTL is flawless.")
# #
# #
# # if __name__ == "__main__":
# #     main()
#
#
# # main.py
# from src.synthetic_emg import generate_left_to_right_swipe, inject_sensor_noise
# from src.time_surface import TimeSurfaceModel
# from src.visualizer import plot_time_surface
# import numpy as np
#
#
# def main():
#     print("--- Running Advanced MIoT Simulator ---")
#
#     # 1. Generate noisy biological data
#     clean_events = generate_left_to_right_swipe()
#     noisy_events = inject_sensor_noise(clean_events, noise_ratio=0.15)  # 15% noise
#
#     # 2. Run the Golden Model
#     ts_model = TimeSurfaceModel()
#     for event in noisy_events:
#         ts_model.process_event(event['x'], event['y'], event['t'], event['p'])
#
#     on_matrix, off_matrix = ts_model.get_hardware_matrices()
#
#     # 3. Generate Publication Graphs
#     plot_time_surface(on_matrix, "Hardware Time-Surface (ON Polarity)", "output/on_heatmap.png")
#
#     # 4. Calculate Sparsity Metrics
#     total_pixels = on_matrix.size
#     active_pixels = np.count_nonzero(on_matrix > 50)  # Threshold for "meaningful" data
#     sparsity_ratio = 100 - ((active_pixels / total_pixels) * 100)
#
#     print("\n==================================================")
#     print("  SIMULATION RESULTS & ANALYTICS")
#     print("==================================================")
#     print(f"Total Events Processed : {len(noisy_events)} (15% Artificial Noise)")
#     print(f"Matrix Sparsity Ratio  : {sparsity_ratio:.2f}% (Memory compression metric)")
#     print("==================================================")
#     print("Check the '/output' folder for your publication-ready graphs!")
#
#
# if __name__ == "__main__":
#     main()

# main.py
from src.dataset_loader import load_aer_dataset
from src.time_surface import TimeSurfaceModel
from src.hardware_writer import export_to_verilog_stimulus
from src.visualizer import plot_time_surface
from src.config import CLOCK_PERIOD_US, IDLE_TIMEOUT_CYCLES
import numpy as np


def main():
    print("--- Running MIoT Hardware Pipeline on REAL DATA ---")

    # 1. Load the real dataset (Assume the biological sensor was 128x128)
    dataset_path = "data/sample_real_emg.csv"
    try:
        events = load_aer_dataset(dataset_path, original_width=128, original_height=128)
    except FileNotFoundError:
        print("Please create the mock dataset in data/sample_real_emg.csv first!")
        return

    # 2. Export to Verilog so Cadence/Synopsys can run it later
    export_to_verilog_stimulus(events, "output/real_stimulus.txt")

    # 3. Run the Hardware Simulator
    ts_model = TimeSurfaceModel()

    total_awake_cycles = 0
    total_asleep_cycles = 0
    last_event_cycle = 0

    print("Feeding real biological data into the hardware core...")
    for event in events:
        ts_model.process_event(event['x'], event['y'], event['t'], event['p'])

        # Power Estimation
        current_cycle = int(event['t'] / CLOCK_PERIOD_US)
        cycle_gap = current_cycle - last_event_cycle

        if cycle_gap > IDLE_TIMEOUT_CYCLES:
            total_awake_cycles += IDLE_TIMEOUT_CYCLES
            total_asleep_cycles += (cycle_gap - IDLE_TIMEOUT_CYCLES)
        else:
            total_awake_cycles += cycle_gap
        last_event_cycle = current_cycle

    # 4. Extract and Plot
    on_matrix, off_matrix = ts_model.get_hardware_matrices()
    plot_time_surface(on_matrix, "Real Data Time-Surface (ON)", "output/real_on_heatmap.png")

    # 5. Output Analytics
    total_cycles = total_awake_cycles + total_asleep_cycles
    power_savings = (total_asleep_cycles / total_cycles) * 100 if total_cycles > 0 else 0

    active_pixels = np.count_nonzero(on_matrix > 50)
    sparsity_ratio = 100 - ((active_pixels / on_matrix.size) * 100)

    print("\n==================================================")
    print("  REAL DATA ANALYTICS & POWER REPORT")
    print("==================================================")
    print(f"Dataset             : {dataset_path}")
    print(f"Events Processed    : {len(events)}")
    print(f"Matrix Sparsity     : {sparsity_ratio:.2f}%")
    print(f"Dynamic Power Saved : {power_savings:.2f}% (via Clock-Gating)")
    print("==================================================")


if __name__ == "__main__":
    main()