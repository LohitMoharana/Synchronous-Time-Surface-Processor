# import os
# import glob
# import numpy as np
# import time
# import scipy.io as sio
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier
#
# from src.time_surface import TimeSurfaceModel
# import src.config as config
#
# # --- FORCING HARDWARE CONFIG FOR IOT FOOTPRINT ---
# GRID_X = 8
# GRID_Y = 8
# CLOCK_PERIOD_US = 1.0 / 50.0  # 50 MHz
# IDLE_TIMEOUT_CYCLES = 100
#
# # --- HARDWARE ENVELOPE CONFIG ---
# config.TAU_US = 100_000.0
#
# # --- SET THIS TO YOUR SUBJECT FOLDER PATH ---
# DB5_SUBJECT_PATH = r"D:\Projects\Personal\MIOT_time_surface\datasets\Ninapro-DB5\Ninapro-DB5\s2"
#
#
# def load_ninapro_mat(filepath):
#     print(f"[Dataset] Loading {os.path.basename(filepath)}...")
#     mat_data = sio.loadmat(filepath)
#
#     emg_raw = mat_data['emg']
#     stimulus = mat_data['stimulus'].flatten()
#
#     X_trials = []
#     y_labels = []
#
#     in_gesture = False
#     start_idx = 0
#
#     for i in range(len(stimulus)):
#         if not in_gesture and stimulus[i] != 0:
#             in_gesture = True
#             start_idx = i
#         elif in_gesture and stimulus[i] == 0:
#             in_gesture = False
#             end_idx = i
#             if (end_idx - start_idx) > 100:
#                 X_trials.append(emg_raw[start_idx:end_idx, :])
#                 y_labels.append(stimulus[start_idx])
#
#     return X_trials, y_labels
#
#
# def delta_modulator(emg_matrix, threshold=25.0):
#     events = []
#     sample_period_us = 5000.0
#
#     last_v = emg_matrix[0, :]
#     for t_idx in range(1, emg_matrix.shape[0]):
#         current_v = emg_matrix[t_idx, :]
#         diffs = current_v - last_v
#
#         for ch in range(16):
#             if abs(diffs[ch]) >= threshold:
#                 # --- BRACELET TOPOLOGY ---
#                 x = ch % 8
#                 y = 2 if ch < 8 else 5
#
#                 events.append({
#                     't': float(t_idx * sample_period_us),
#                     'x': x,
#                     'y': y,
#                     'p': 1
#                 })
#                 last_v[ch] = current_v[ch]
#
#     return events
#
#
# def run_real_db5_benchmark():
#     mat_files = glob.glob(os.path.join(DB5_SUBJECT_PATH, "*E1*.mat"))
#
#     if not mat_files:
#         print(f"ERROR: Could not find any *E1*.mat files in {DB5_SUBJECT_PATH}")
#         return
#
#     X_features = []
#     y_all = []
#
#     total_awake = 0
#     total_asleep = 0
#     total_events = 0
#
#     # 50ms windows
#     WINDOW_STEP_US = 50_000.0
#
#     print("\n--- Processing NinaPro DB5 Real Datasets (Exercise 1 Only) ---")
#
#     for file in mat_files:
#         X_trials, y_labels = load_ninapro_mat(file)
#
#         for emg_data, label in zip(X_trials, y_labels):
#             events = delta_modulator(emg_data, threshold=22.0)
#             if len(events) < 50:
#                 continue
#
#             total_events += len(events)
#             ts_model = TimeSurfaceModel()
#             last_cycle = 0
#
#             t_start = events[0]['t']
#             t_end = events[-1]['t']
#             duration = t_end - t_start
#
#             extract_start = t_start + (duration * 0.2)
#             extract_end = t_start + (duration * 0.8)
#             next_snapshot_time = extract_start + WINDOW_STEP_US
#
#             history_buffer = []
#
#             for ev in events:
#                 ts_model.process_event(ev['x'], ev['y'], ev['t'], ev['p'])
#
#                 if ev['t'] >= next_snapshot_time and ev['t'] <= extract_end:
#                     on_m, _ = ts_model.get_hardware_matrices()
#                     history_buffer.append(on_m.flatten())
#
#                     # --- INCREASED TEMPORAL DEPTH: 8-FRAME (400ms History) ---
#                     if len(history_buffer) == 8:
#                         X_features.append(np.concatenate(history_buffer))
#                         y_all.append(label)
#                         history_buffer.pop(0)
#
#                     next_snapshot_time += WINDOW_STEP_US
#
#                 curr_cycle = int(ev['t'] / CLOCK_PERIOD_US)
#                 gap = curr_cycle - last_cycle
#                 if gap > IDLE_TIMEOUT_CYCLES:
#                     total_awake += IDLE_TIMEOUT_CYCLES
#                     total_asleep += (gap - IDLE_TIMEOUT_CYCLES)
#                 else:
#                     total_awake += gap
#                 last_cycle = curr_cycle
#
#     X_features = np.array(X_features)
#     y_all = np.array(y_all)
#
#     total_cycles = total_awake + total_asleep
#     savings = (total_asleep / total_cycles * 100) if total_cycles > 0 else 0
#
#     print("\n==========================================================")
#     print("  NINAPRO DB5 (E1 CORE GESTURES): SILICON PERFORMANCE REPORT")
#     print("==========================================================")
#     print(f"Data Snapshots Extracted : {len(X_features)} (8-Frame Temporal Windows)")
#     print(f"Matrix Grid Size         : {GRID_X}x{GRID_Y} (512 hardware features per snapshot)")
#     print(f"Hardware Decay Set To    : {config.TAU_US / 1000} ms")
#     print(f"Unique Gesture Classes   : {len(np.unique(y_all))}")
#     print(f"Total Spikes Fired       : {total_events:,}")
#     print(f"Power Gating Efficiency  : {savings:.2f}% Sleep Time")
#     print("----------------------------------------------------------")
#
#     if len(np.unique(y_all)) < 2:
#         print("Not enough diverse gestures found to train ML model.")
#         return
#
#     encoder = LabelEncoder()
#     y_encoded = encoder.fit_transform(y_all)
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#     )
#
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#
#     # --- FULLY GROWN RANDOM FOREST ---
#     t0 = time.time()
#     print("-> Training Random Forest Classifier (Synthesizable Edge Logic)...")
#     rf = RandomForestClassifier(n_estimators=500, max_depth=None, random_state=42, n_jobs=-1)
#     rf.fit(X_train, y_train)
#     rf_acc = accuracy_score(y_test, rf.predict(X_test)) * 100
#     print(f"1. Random Forest Accuracy: {rf_acc:.2f}% (Training Time: {(time.time() - t0) * 1000:.2f} ms)")
#
#     # --- XGBOOST ---
#     t0 = time.time()
#     print("-> Training XGBoost Classifier (Gradient Boosting)...")
#     xgb = XGBClassifier(
#         n_estimators=700,
#         max_depth=8,
#         learning_rate=0.05,
#         subsample=0.8,
#         colsample_bytree=0.7,
#         random_state=42,
#         n_jobs=-1,
#         eval_metric='mlogloss'
#     )
#     xgb.fit(X_train, y_train)
#     xgb_acc = accuracy_score(y_test, xgb.predict(X_test)) * 100
#     print(f"2. XGBoost Accuracy      : {xgb_acc:.2f}% (Training Time: {(time.time() - t0) * 1000:.2f} ms)")
#
#     # --- MULTI-LAYER PERCEPTRON ---
#     t0 = time.time()
#     print("-> Training Deep Multi-Layer Perceptron (MLP)...")
#     mlp = MLPClassifier(hidden_layer_sizes=(1024, 256, 256), max_iter=2000, random_state=42, learning_rate_init=0.005)
#     mlp.fit(X_train, y_train)
#     mlp_acc = accuracy_score(y_test, mlp.predict(X_test)) * 100
#     print(f"3. MLP Accuracy (Deep)   : {mlp_acc:.2f}% (Training Time: {(time.time() - t0) * 1000:.2f} ms)")
#     print("==========================================================")
#
#
# if __name__ == "__main__":
#     run_real_db5_benchmark()

import os, glob, time, scipy.io as sio, numpy as np
import joblib  # <-- CRITICAL: For saving the model!
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from src.time_surface import TimeSurfaceModel
import src.config as config

# --- CONFIG FOR MID-90S STABILITY ---
config.TAU_US = 75_000.0
GRID_X, GRID_Y = 8, 8


def delta_modulator(emg_matrix, threshold=22.0):
    events, last_v = [], emg_matrix[0, :]
    for t_idx in range(1, emg_matrix.shape[0]):
        current_v = emg_matrix[t_idx, :]
        diffs = current_v - last_v
        for ch in range(16):
            if abs(diffs[ch]) >= threshold:
                x, y = (ch % 8), (2 if ch < 8 else 5)
                events.append({'t': float(t_idx * 5000.0), 'x': x, 'y': y, 'p': 1})
                last_v[ch] = current_v[ch]
    return events


def run_real_db5_benchmark():
    mat_files = glob.glob(
        os.path.join(r"D:\Projects\Personal\MIOT_time_surface\datasets\Ninapro-DB5\Ninapro-DB5\s2", "*E1*.mat"))

    if not mat_files:
        print("ERROR: No .mat files found! Check your path.")
        return

    X, y = [], []

    print("\n--- 1. Extracting Data & Generating Time-Surfaces ---")
    for file in mat_files:
        print(f" -> Loading {os.path.basename(file)}...")
        mat_data = sio.loadmat(file)
        emg_raw, stimulus = mat_data['emg'], mat_data['stimulus'].flatten()

        # DOWN-SAMPLED OVERLAP: Step increased to 100 to halve dataset size and speed up extraction
        for i in range(0, len(stimulus) - 300, 100):
            if stimulus[i] != 0:
                events = delta_modulator(emg_raw[i:i + 300, :])
                if len(events) < 50: continue

                ts = TimeSurfaceModel()
                history = []
                for ev in events:
                    ts.process_event(ev['x'], ev['y'], ev['t'], ev['p'])
                    if ev['t'] % 30000.0 < 5000:
                        on_m, _ = ts.get_hardware_matrices()
                        history.append(on_m.flatten())
                        if len(history) == 8:
                            feat = np.concatenate(history)
                            X.append(np.hstack([feat, np.mean(history, axis=0), np.std(history, axis=0)]))
                            y.append(stimulus[i])
                            history.pop(0)

    X, y = np.array(X), np.array(y)
    print(f"\n[SUCCESS] Extracted {len(X)} stable snapshots. Feature size: {X.shape[1]}")

    print("\n--- 2. Splitting and Scaling Data ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    print("--- 3. Training Final MLP Model (One-Shot) ---")
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(1024, 1024, 512, 256),
                        activation='relu', solver='adam',
                        learning_rate='adaptive', max_iter=2000,
                        verbose=True, early_stopping=True).fit(X_train, y_train)

    acc = accuracy_score(y_test, mlp.predict(X_test))

    print("\n==========================================================")
    print(f"-> FINAL MODEL ACCURACY : {acc * 100:.2f}% (Time: {(time.time() - t0):.2f} sec)")
    print("==========================================================")

    # --- CRITICAL FIX: SAVE THE MODEL AND SCALER ---
    save_dir = r"D:\Projects\Personal\MIOT_time_surface\models"
    os.makedirs(save_dir, exist_ok=True)

    scaler_path = os.path.join(save_dir, "miot_scaler.pkl")
    model_path = os.path.join(save_dir, "miot_mlp_model.pkl")

    joblib.dump(scaler, scaler_path)
    joblib.dump(mlp, model_path)

    print(f"\n[SAVED] Scaler saved to: {scaler_path}")
    print(f"[SAVED] Model saved to : {model_path}")


if __name__ == "__main__":
    run_real_db5_benchmark()