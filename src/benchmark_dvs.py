# benchmark_dvs.py
import os
import glob
import csv
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Import our hardware pipeline
from src.aedat_parser import load_aedat_file
from src.time_surface import TimeSurfaceModel
from src.config import GRID_X, GRID_Y, CLOCK_PERIOD_US, IDLE_TIMEOUT_CYCLES

# --- SET THIS TO YOUR EXACT WINDOWS PATH ---
DATASET_PATH = r"D:\Projects\Personal\MIOT_time_surface\datasets\DVSGesture128\DvsGesture\DvsGesture"


def process_dvs_directory():
    X_features = []
    y_labels = []

    total_awake = 0
    total_asleep = 0

    label_files = glob.glob(os.path.join(DATASET_PATH, "*_labels.csv"))

    if not label_files:
        print(f"ERROR: Could not find any _labels.csv files in {DATASET_PATH}")
        return None, None, None

    print(f"--- Processing {len(label_files)} Trial Recordings ---")

    for label_file in label_files:
        aedat_file = label_file.replace('_labels.csv', '.aedat')
        if not os.path.exists(aedat_file):
            continue

        all_events = load_aedat_file(aedat_file)

        with open(label_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                try:
                    class_id = int(row[0])
                    start_time = float(row[1])
                    end_time = float(row[2])
                except (ValueError, IndexError):
                    continue

                gesture_events = [e for e in all_events if start_time <= e['t'] <= end_time]
                gesture_events.sort(key=lambda x: x['t'])

                if len(gesture_events) < 100:
                    continue

                ts_model = TimeSurfaceModel()
                last_cycle = int(start_time / CLOCK_PERIOD_US)

                scale_x = 128 / GRID_X
                scale_y = 128 / GRID_Y

                # --- NEW LOGIC: Calculate temporal breakpoints for our 3 frames ---
                duration = end_time - start_time
                t_33 = start_time + (duration * 0.33)
                t_66 = start_time + (duration * 0.66)

                frame1_captured, frame2_captured = False, False
                gesture_feature_vector = []

                for ev in gesture_events:
                    hw_x = min(int(ev['x'] / scale_x), GRID_X - 1)
                    hw_y = min(int(ev['y'] / scale_y), GRID_Y - 1)

                    ts_model.process_event(hw_x, hw_y, ev['t'], ev['p'])

                    # --- TAKE SNAPSHOT 1 (33%) ---
                    if ev['t'] >= t_33 and not frame1_captured:
                        on_m, off_m = ts_model.get_hardware_matrices()
                        gesture_feature_vector.extend(on_m.flatten())
                        gesture_feature_vector.extend(off_m.flatten())
                        frame1_captured = True

                    # --- TAKE SNAPSHOT 2 (66%) ---
                    if ev['t'] >= t_66 and not frame2_captured:
                        on_m, off_m = ts_model.get_hardware_matrices()
                        if class_id == 0 and len(X_features) == 0:
                            print("\n[Debug] Mid-Gesture Hardware Matrix:")
                            print(on_m)
                        gesture_feature_vector.extend(on_m.flatten())
                        gesture_feature_vector.extend(off_m.flatten())
                        frame2_captured = True

                    # Power Analytics
                    curr_cycle = int(ev['t'] / CLOCK_PERIOD_US)
                    gap = curr_cycle - last_cycle
                    if gap > IDLE_TIMEOUT_CYCLES:
                        total_awake += IDLE_TIMEOUT_CYCLES
                        total_asleep += (gap - IDLE_TIMEOUT_CYCLES)
                    else:
                        total_awake += gap
                    last_cycle = curr_cycle

                # --- TAKE SNAPSHOT 3 (100% - The End) ---
                on_m, off_m = ts_model.get_hardware_matrices()
                gesture_feature_vector.extend(on_m.flatten())
                gesture_feature_vector.extend(off_m.flatten())

                # Append the combined 384-element feature vector to our dataset
                expected_feature_length = GRID_X * GRID_Y * 2 * 3
                if len(gesture_feature_vector) == expected_feature_length:
                    X_features.append(gesture_feature_vector)
                    y_labels.append(class_id)

    total_cycles = total_awake + total_asleep
    savings = (total_asleep / total_cycles * 100) if total_cycles > 0 else 0

    return np.array(X_features), np.array(y_labels), savings

def evaluate(X, y, savings):
    if X is None or len(X) == 0:
        return

    print("\n==========================================================")
    print("  DVS128 DATASET: ACTUAL SILICON PERFORMANCE REPORT")
    print("==========================================================")
    print(f"Total Gestures Processed : {len(X)}")
    print(f"Power Gating Efficiency  : {savings:.2f}% Sleep Time")
    print("----------------------------------------------------------")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. SVM Classification
    t0 = time.time()
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test)) * 100
    print(f"1. SVM Accuracy (Linear) : {svm_acc:.2f}% (Time: {(time.time() - t0) * 1000:.2f}ms)")

    # 2. MLP Classification
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000)
    mlp.fit(X_train, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test)) * 100
    print(f"2. MLP Accuracy (Deep)   : {mlp_acc:.2f}% (Time: {(time.time() - t0) * 1000:.2f}ms)")
    print("==========================================================")


if __name__ == "__main__":
    X, y, savings = process_dvs_directory()
    evaluate(X, y, savings)