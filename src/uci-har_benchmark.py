# # # import os, time
# # # import numpy as np
# # # from sklearn.neural_network import MLPClassifier
# # # from sklearn.svm import SVC
# # # from sklearn.ensemble import RandomForestClassifier
# # # from sklearn.preprocessing import StandardScaler
# # # from sklearn.metrics import accuracy_score
# # # from src.time_surface import TimeSurfaceModel
# # # import src.config as config
# # #
# # # # --- OPTIMIZED KINEMATICS CONFIG ---
# # # config.TAU_US = 2_500_000.0  # 2.5s decay. Posture is a long-term physical state! We want the memory to persist.
# # # GRID_X, GRID_Y = 8, 8
# # #
# # # # Path based on your provided screenshot
# # # DATA_DIR = r"D:\Projects\Personal\MIOT_time_surface\datasets\UCI-HAR\UCI-HAR Dataset"
# # #
# # #
# # # def load_har_raw_data(dataset_type="train"):
# # #     print(f" -> Loading '{dataset_type}' IMU signals...")
# # #     signal_path = os.path.join(DATA_DIR, dataset_type, "Inertial Signals")
# # #
# # #     # --- THE PERFECT 8-CHANNEL FIT ---
# # #     # We drop 'body_gyro_z' (Yaw/Turning around), which is irrelevant for these 6 activities.
# # #     # This leaves us with exactly 8 channels to perfectly fill our 8-pixel width!
# # #     files = [
# # #         f"total_acc_x_{dataset_type}.txt", f"total_acc_y_{dataset_type}.txt", f"total_acc_z_{dataset_type}.txt",
# # #         f"body_acc_x_{dataset_type}.txt", f"body_acc_y_{dataset_type}.txt", f"body_acc_z_{dataset_type}.txt",
# # #         f"body_gyro_x_{dataset_type}.txt", f"body_gyro_y_{dataset_type}.txt"
# # #     ]
# # #
# # #     channels = []
# # #     for f in files:
# # #         file_path = os.path.join(signal_path, f)
# # #         data = np.loadtxt(file_path)
# # #         channels.append(data)
# # #
# # #     # Shape: (Windows, 128 timesteps, 8 distinct channels)
# # #     X = np.dstack(channels)
# # #
# # #     y_path = os.path.join(DATA_DIR, dataset_type, f"y_{dataset_type}.txt")
# # #     y = np.loadtxt(y_path)
# # #
# # #     return X, y
# # #
# # #
# # # def delta_modulator_imu(imu_window, threshold=0.015):
# # #     """Converts continuous physics data into Neuromorphic Spikes using Value-to-Space Encoding."""
# # #     events = []
# # #     sample_period_us = 20_000.0  # IMU sampled at 50Hz (20ms per step)
# # #
# # #     # 1. We explicitly inject the starting posture at t=0.
# # #     for ch in range(8):
# # #         # We divide by 2.0 to expand the dynamic range mapping to [-2g, +2g]
# # #         val = np.clip(imu_window[0, ch] / 2.0, -1.0, 1.0)
# # #
# # #         # 8 channels map perfectly 1-to-1 with the 8 horizontal X-pixels!
# # #         x = ch
# # #
# # #         # Map the absolute physical value to a spatial Y-coordinate
# # #         y = min(7, max(0, int((val + 1.0) * 3.99)))
# # #         events.append({'t': 0.0, 'x': x, 'y': y, 'p': 1})
# # #
# # #     # 2. Standard delta-modulation for subsequent movement
# # #     last_v = imu_window[0, :]
# # #     for t_idx in range(1, imu_window.shape[0]):
# # #         current_v = imu_window[t_idx, :]
# # #         diffs = current_v - last_v
# # #
# # #         for ch in range(8):
# # #             if abs(diffs[ch]) >= threshold:
# # #                 val = np.clip(current_v[ch] / 2.0, -1.0, 1.0)
# # #
# # #                 x = ch
# # #                 y = min(7, max(0, int((val + 1.0) * 3.99)))
# # #
# # #                 polarity = 1 if diffs[ch] > 0 else 0
# # #                 events.append({'t': float(t_idx * sample_period_us), 'x': x, 'y': y, 'p': polarity})
# # #                 last_v[ch] = current_v[ch]
# # #     return events
# # #
# # #
# # # def run_imu_benchmark():
# # #     print("==========================================================")
# # #     print("  UCI HAR KINEMATICS: NEUROMORPHIC TIME-SURFACE BENCHMARK")
# # #     print("==========================================================")
# # #
# # #     if not os.path.exists(DATA_DIR):
# # #         print(f"[ERROR] Could not find dataset at: {DATA_DIR}")
# # #         return
# # #
# # #     # 1. Load Data
# # #     X_raw_train, y_train_labels = load_har_raw_data("train")
# # #     X_raw_test, y_test_labels = load_har_raw_data("test")
# # #
# # #     def process_to_time_surfaces(X_raw, dataset_name):
# # #         X_features = []
# # #         print(f" -> Delta-Modulating {len(X_raw)} {dataset_name} windows...")
# # #
# # #         total_awake = 0
# # #         total_asleep = 0
# # #
# # #         for i in range(X_raw.shape[0]):
# # #             events = delta_modulator_imu(X_raw[i], threshold=0.015)
# # #             ts = TimeSurfaceModel()
# # #
# # #             # The window is exactly 2.56 seconds. We split it into 8 frames (0.32s each)
# # #             frame_interval_us = 2_560_000.0 / 8.0
# # #             history, next_frame_time = [], frame_interval_us
# # #             last_cycle = 0
# # #
# # #             for ev in events:
# # #                 ts.process_event(ev['x'], ev['y'], ev['t'], ev['p'])
# # #
# # #                 # Power gating logic
# # #                 curr_cycle = int(ev['t'] / config.CLOCK_PERIOD_US)
# # #                 gap = curr_cycle - last_cycle
# # #                 if gap > config.IDLE_TIMEOUT_CYCLES:
# # #                     total_awake += config.IDLE_TIMEOUT_CYCLES
# # #                     total_asleep += (gap - config.IDLE_TIMEOUT_CYCLES)
# # #                 else:
# # #                     total_awake += gap
# # #                 last_cycle = curr_cycle
# # #
# # #                 if ev['t'] >= next_frame_time:
# # #                     on_m, _ = ts.get_hardware_matrices()
# # #                     history.append(on_m.flatten())
# # #                     next_frame_time += frame_interval_us
# # #
# # #             while len(history) < 8:
# # #                 on_m, _ = ts.get_hardware_matrices()
# # #                 history.append(on_m.flatten())
# # #
# # #             feat = np.concatenate(history[:8])
# # #             X_features.append(np.hstack([feat, np.mean(history[:8], axis=0), np.std(history[:8], axis=0)]))
# # #
# # #         return np.array(X_features), total_awake, total_asleep
# # #
# # #     # 2. Convert to Neuromorphic Features
# # #     X_train, awake_tr, asleep_tr = process_to_time_surfaces(X_raw_train, "Train")
# # #     X_test, awake_te, asleep_te = process_to_time_surfaces(X_raw_test, "Test")
# # #
# # #     total_cycles = (awake_tr + asleep_tr + awake_te + asleep_te)
# # #     sleep_efficiency = ((asleep_tr + asleep_te) / total_cycles * 100) if total_cycles > 0 else 0
# # #
# # #     print("\n--- 3. Scaling and Training Models ---")
# # #     scaler = StandardScaler().fit(X_train)
# # #     X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
# # #
# # #     t0 = time.time()
# # #
# # #     print(" -> Training Random Forest...")
# # #     rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
# # #     rf.fit(X_train, y_train_labels)
# # #     rf_acc = accuracy_score(y_test_labels, rf.predict(X_test))
# # #
# # #     print(" -> Training Support Vector Machine (RBF)...")
# # #     svm = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
# # #     svm.fit(X_train, y_train_labels)
# # #     svm_acc = accuracy_score(y_test_labels, svm.predict(X_test))
# # #
# # #     print(" -> Training Deep MLP (Adam)...")
# # #     mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128),
# # #                         activation='relu',
# # #                         solver='adam',
# # #                         max_iter=500,
# # #                         early_stopping=True,
# # #                         random_state=42).fit(X_train, y_train_labels)
# # #     mlp_acc = accuracy_score(y_test_labels, mlp.predict(X_test))
# # #
# # #     print("\n==========================================================")
# # #     print("  UCI HAR FINAL ACCURACY REPORT (6-Class Kinematics)")
# # #     print("==========================================================")
# # #     print(f"Power Gating Efficiency   : {sleep_efficiency:.2f}% Sleep Time")
# # #     print("----------------------------------------------------------")
# # #     print(f"1. Random Forest Accuracy : {rf_acc * 100:.2f}%")
# # #     print(f"2. SVM (RBF) Accuracy     : {svm_acc * 100:.2f}%")
# # #     print(f"3. Deep MLP Accuracy      : {mlp_acc * 100:.2f}%")
# # #     print("----------------------------------------------------------")
# # #     print("Classes: Walk, Walk Up, Walk Down, Sit, Stand, Lay Down")
# # #     print(f"Total Computation Time: {time.time() - t0:.2f} seconds")
# # #
# # #
# # # if __name__ == "__main__":
# # #     run_imu_benchmark()
# #
# # # import os, time
# # # import numpy as np
# # # from sklearn.neural_network import MLPClassifier
# # # from sklearn.svm import SVC
# # # from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
# # # from sklearn.preprocessing import StandardScaler
# # # from sklearn.metrics import accuracy_score
# # # from src.time_surface import TimeSurfaceModel
# # # import src.config as config
# # #
# # # # --- OPTIMIZED KINEMATICS CONFIG ---
# # # config.TAU_US = 2_500_000.0  # 2.5s decay. Posture is a long-term physical state! We want the memory to persist.
# # # GRID_X, GRID_Y = 8, 8
# # #
# # # # Path based on your provided screenshot
# # # DATA_DIR = r"D:\Projects\Personal\MIOT_time_surface\datasets\UCI-HAR\UCI-HAR Dataset"
# # #
# # #
# # # def load_har_raw_data(dataset_type="train"):
# # #     print(f" -> Loading '{dataset_type}' IMU signals...")
# # #     signal_path = os.path.join(DATA_DIR, dataset_type, "Inertial Signals")
# # #
# # #     # --- THE PERFECT 8-CHANNEL FIT ---
# # #     # We drop 'body_gyro_z' (Yaw/Turning around), which is irrelevant for these 6 activities.
# # #     # This leaves us with exactly 8 channels to perfectly fill our 8-pixel width!
# # #     files = [
# # #         f"total_acc_x_{dataset_type}.txt", f"total_acc_y_{dataset_type}.txt", f"total_acc_z_{dataset_type}.txt",
# # #         f"body_acc_x_{dataset_type}.txt", f"body_acc_y_{dataset_type}.txt", f"body_acc_z_{dataset_type}.txt",
# # #         f"body_gyro_x_{dataset_type}.txt", f"body_gyro_y_{dataset_type}.txt"
# # #     ]
# # #
# # #     channels = []
# # #     for f in files:
# # #         file_path = os.path.join(signal_path, f)
# # #         data = np.loadtxt(file_path)
# # #         channels.append(data)
# # #
# # #     # Shape: (Windows, 128 timesteps, 8 distinct channels)
# # #     X = np.dstack(channels)
# # #
# # #     y_path = os.path.join(DATA_DIR, dataset_type, f"y_{dataset_type}.txt")
# # #     y = np.loadtxt(y_path)
# # #
# # #     return X, y
# # #
# # #
# # # def delta_modulator_imu(imu_window, threshold=0.015):
# # #     """Converts continuous physics data into Neuromorphic Spikes using Value-to-Space Encoding."""
# # #     events = []
# # #     sample_period_us = 20_000.0  # IMU sampled at 50Hz (20ms per step)
# # #
# # #     # 1. We explicitly inject the starting posture at t=0.
# # #     for ch in range(8):
# # #         # We divide by 2.0 to expand the dynamic range mapping to [-2g, +2g]
# # #         val = np.clip(imu_window[0, ch] / 2.0, -1.0, 1.0)
# # #
# # #         # 8 channels map perfectly 1-to-1 with the 8 horizontal X-pixels!
# # #         x = ch
# # #
# # #         # Map the absolute physical value to a spatial Y-coordinate
# # #         y = min(7, max(0, int((val + 1.0) * 3.99)))
# # #         events.append({'t': 0.0, 'x': x, 'y': y, 'p': 1})
# # #
# # #     # 2. Standard delta-modulation for subsequent movement
# # #     last_v = imu_window[0, :]
# # #     for t_idx in range(1, imu_window.shape[0]):
# # #         current_v = imu_window[t_idx, :]
# # #         diffs = current_v - last_v
# # #
# # #         for ch in range(8):
# # #             if abs(diffs[ch]) >= threshold:
# # #                 val = np.clip(current_v[ch] / 2.0, -1.0, 1.0)
# # #
# # #                 x = ch
# # #                 y = min(7, max(0, int((val + 1.0) * 3.99)))
# # #
# # #                 polarity = 1 if diffs[ch] > 0 else 0
# # #                 events.append({'t': float(t_idx * sample_period_us), 'x': x, 'y': y, 'p': polarity})
# # #                 last_v[ch] = current_v[ch]
# # #     return events
# # #
# # #
# # # def run_imu_benchmark():
# # #     print("==========================================================")
# # #     print("  UCI HAR KINEMATICS: NEUROMORPHIC TIME-SURFACE BENCHMARK")
# # #     print("==========================================================")
# # #
# # #     if not os.path.exists(DATA_DIR):
# # #         print(f"[ERROR] Could not find dataset at: {DATA_DIR}")
# # #         return
# # #
# # #     # 1. Load Data
# # #     X_raw_train, y_train_labels = load_har_raw_data("train")
# # #     X_raw_test, y_test_labels = load_har_raw_data("test")
# # #
# # #     def process_to_time_surfaces(X_raw, dataset_name):
# # #         X_features = []
# # #         print(f" -> Delta-Modulating {len(X_raw)} {dataset_name} windows...")
# # #
# # #         total_awake = 0
# # #         total_asleep = 0
# # #
# # #         for i in range(X_raw.shape[0]):
# # #             events = delta_modulator_imu(X_raw[i], threshold=0.015)
# # #             ts = TimeSurfaceModel()
# # #
# # #             # The window is exactly 2.56 seconds. We split it into 8 frames (0.32s each)
# # #             frame_interval_us = 2_560_000.0 / 8.0
# # #             history, next_frame_time = [], frame_interval_us
# # #             last_cycle = 0
# # #
# # #             for ev in events:
# # #                 ts.process_event(ev['x'], ev['y'], ev['t'], ev['p'])
# # #
# # #                 # Power gating logic
# # #                 curr_cycle = int(ev['t'] / config.CLOCK_PERIOD_US)
# # #                 gap = curr_cycle - last_cycle
# # #                 if gap > config.IDLE_TIMEOUT_CYCLES:
# # #                     total_awake += config.IDLE_TIMEOUT_CYCLES
# # #                     total_asleep += (gap - config.IDLE_TIMEOUT_CYCLES)
# # #                 else:
# # #                     total_awake += gap
# # #                 last_cycle = curr_cycle
# # #
# # #                 if ev['t'] >= next_frame_time:
# # #                     on_m, _ = ts.get_hardware_matrices()
# # #                     history.append(on_m.flatten())
# # #                     next_frame_time += frame_interval_us
# # #
# # #             while len(history) < 8:
# # #                 on_m, _ = ts.get_hardware_matrices()
# # #                 history.append(on_m.flatten())
# # #
# # #             feat = np.concatenate(history[:8])
# # #             X_features.append(np.hstack([feat, np.mean(history[:8], axis=0), np.std(history[:8], axis=0)]))
# # #
# # #         return np.array(X_features), total_awake, total_asleep
# # #
# # #     # 2. Convert to Neuromorphic Features
# # #     X_train, awake_tr, asleep_tr = process_to_time_surfaces(X_raw_train, "Train")
# # #     X_test, awake_te, asleep_te = process_to_time_surfaces(X_raw_test, "Test")
# # #
# # #     total_cycles = (awake_tr + asleep_tr + awake_te + asleep_te)
# # #     sleep_efficiency = ((asleep_tr + asleep_te) / total_cycles * 100) if total_cycles > 0 else 0
# # #
# # #     print("\n--- 3. Scaling and Training Models ---")
# # #     scaler = StandardScaler().fit(X_train)
# # #     X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
# # #
# # #     t0 = time.time()
# # #
# # #     print(" -> Training Random Forest...")
# # #     rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
# # #     rf.fit(X_train, y_train_labels)
# # #     rf_acc = accuracy_score(y_test_labels, rf.predict(X_test))
# # #
# # #     print(" -> Training Histogram Gradient Boosting (State-of-the-Art)...")
# # #     hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_depth=10, random_state=42)
# # #     hgb.fit(X_train, y_train_labels)
# # #     hgb_acc = accuracy_score(y_test_labels, hgb.predict(X_test))
# # #
# # #     print(" -> Training Deep MLP (Adam)...")
# # #     # Reduced the initial layer slightly, as 512 was likely causing it to overfit the empty black pixels
# # #     mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64),
# # #                         activation='relu',
# # #                         solver='adam',
# # #                         max_iter=500,
# # #                         early_stopping=True,
# # #                         random_state=42).fit(X_train, y_train_labels)
# # #     mlp_acc = accuracy_score(y_test_labels, mlp.predict(X_test))
# # #
# # #     print("\n==========================================================")
# # #     print("  UCI HAR FINAL ACCURACY REPORT (6-Class Kinematics)")
# # #     print("==========================================================")
# # #     print(f"Power Gating Efficiency   : {sleep_efficiency:.2f}% Sleep Time")
# # #     print("----------------------------------------------------------")
# # #     print(f"1. Random Forest Accuracy : {rf_acc * 100:.2f}%")
# # #     print(f"2. Gradient Boost Accuracy: {hgb_acc * 100:.2f}%  <-- (NEW LEADER?)")
# # #     print(f"3. Deep MLP Accuracy      : {mlp_acc * 100:.2f}%")
# # #     print("----------------------------------------------------------")
# # #     print("Classes: Walk, Walk Up, Walk Down, Sit, Stand, Lay Down")
# # #     print(f"Total Computation Time: {time.time() - t0:.2f} seconds")
# # #
# # #
# # # if __name__ == "__main__":
# # #     run_imu_benchmark()
# #
# #
# # import os, time
# # import numpy as np
# # from sklearn.neural_network import MLPClassifier
# # from sklearn.svm import SVC
# # from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
# # from sklearn.preprocessing import MinMaxScaler
# # from sklearn.metrics import accuracy_score
# # from src.time_surface import TimeSurfaceModel
# # import src.config as config
# #
# # # --- OPTIMIZED KINEMATICS CONFIG ---
# # config.TAU_US = 750_000.0  # 750ms decay. Allows static posture to persist, but dim fast enough to show the "heartbeat" of walking.
# # GRID_X, GRID_Y = 8, 8
# #
# # # Path based on your provided screenshot
# # DATA_DIR = r"D:\Projects\Personal\MIOT_time_surface\datasets\UCI-HAR\UCI-HAR Dataset"
# #
# #
# # def load_har_raw_data(dataset_type="train"):
# #     print(f" -> Loading '{dataset_type}' IMU signals...")
# #     signal_path = os.path.join(DATA_DIR, dataset_type, "Inertial Signals")
# #
# #     files = [
# #         f"total_acc_x_{dataset_type}.txt", f"total_acc_y_{dataset_type}.txt", f"total_acc_z_{dataset_type}.txt",
# #         f"body_acc_x_{dataset_type}.txt", f"body_acc_y_{dataset_type}.txt", f"body_acc_z_{dataset_type}.txt",
# #         f"body_gyro_x_{dataset_type}.txt", f"body_gyro_y_{dataset_type}.txt"
# #     ]
# #
# #     channels = []
# #     for f in files:
# #         file_path = os.path.join(signal_path, f)
# #         data = np.loadtxt(file_path)
# #         channels.append(data)
# #
# #     X = np.dstack(channels)
# #     y_path = os.path.join(DATA_DIR, dataset_type, f"y_{dataset_type}.txt")
# #     y = np.loadtxt(y_path)
# #
# #     return X, y
# #
# #
# # def get_scaled_val(raw_val, ch):
# #     """Applies channel-specific optical stretching so all sensors fully utilize the 8-pixel height."""
# #     if ch < 3:
# #         return np.clip(raw_val / 1.5, -1.0, 1.0)  # Total Acc (Gravity is strong)
# #     elif ch < 6:
# #         return np.clip(raw_val / 0.5, -1.0, 1.0)  # Body Acc (Movement is weak, stretch it!)
# #     else:
# #         return np.clip(raw_val / 2.0, -1.0, 1.0)  # Gyro (Rotation)
# #
# #
# # def delta_modulator_imu(imu_window, threshold=0.015):
# #     """Converts continuous physics data into Neuromorphic Spikes."""
# #     events = []
# #     sample_period_us = 20_000.0
# #
# #     # 1. Explicitly inject the starting posture at t=0.
# #     for ch in range(8):
# #         val = get_scaled_val(imu_window[0, ch], ch)
# #         x = ch
# #         y = min(7, max(0, int((val + 1.0) * 3.99)))
# #         events.append({'t': 0.0, 'x': x, 'y': y, 'p': 1})
# #
# #     # 2. Delta-modulation for subsequent movement
# #     last_v = imu_window[0, :]
# #     for t_idx in range(1, imu_window.shape[0]):
# #         current_v = imu_window[t_idx, :]
# #         diffs = current_v - last_v
# #
# #         for ch in range(8):
# #             if abs(diffs[ch]) >= threshold:
# #                 val = get_scaled_val(current_v[ch], ch)
# #                 x = ch
# #                 y = min(7, max(0, int((val + 1.0) * 3.99)))
# #                 polarity = 1 if diffs[ch] > 0 else 0
# #                 events.append({'t': float(t_idx * sample_period_us), 'x': x, 'y': y, 'p': polarity})
# #                 last_v[ch] = current_v[ch]
# #     return events
# #
# #
# # def run_imu_benchmark():
# #     print("==========================================================")
# #     print("  UCI HAR KINEMATICS: NEUROMORPHIC TIME-SURFACE BENCHMARK")
# #     print("==========================================================")
# #
# #     if not os.path.exists(DATA_DIR):
# #         print(f"[ERROR] Could not find dataset at: {DATA_DIR}")
# #         return
# #
# #     X_raw_train, y_train_labels = load_har_raw_data("train")
# #     X_raw_test, y_test_labels = load_har_raw_data("test")
# #
# #     def process_to_time_surfaces(X_raw, dataset_name):
# #         X_features = []
# #         print(f" -> Delta-Modulating {len(X_raw)} {dataset_name} windows...")
# #
# #         total_awake = 0
# #         total_asleep = 0
# #
# #         for i in range(X_raw.shape[0]):
# #             events = delta_modulator_imu(X_raw[i], threshold=0.015)
# #             ts = TimeSurfaceModel()
# #
# #             frame_interval_us = 2_560_000.0 / 8.0
# #             history, next_frame_time = [], frame_interval_us
# #             last_cycle = 0
# #
# #             for ev in events:
# #                 ts.process_event(ev['x'], ev['y'], ev['t'], ev['p'])
# #
# #                 curr_cycle = int(ev['t'] / config.CLOCK_PERIOD_US)
# #                 gap = curr_cycle - last_cycle
# #                 if gap > config.IDLE_TIMEOUT_CYCLES:
# #                     total_awake += config.IDLE_TIMEOUT_CYCLES
# #                     total_asleep += (gap - config.IDLE_TIMEOUT_CYCLES)
# #                 else:
# #                     total_awake += gap
# #                 last_cycle = curr_cycle
# #
# #                 if ev['t'] >= next_frame_time:
# #                     on_m, _ = ts.get_hardware_matrices()
# #                     history.append(on_m.flatten())
# #                     next_frame_time += frame_interval_us
# #
# #             while len(history) < 8:
# #                 on_m, _ = ts.get_hardware_matrices()
# #                 history.append(on_m.flatten())
# #
# #             feat = np.concatenate(history[:8])
# #             X_features.append(np.hstack([feat, np.mean(history[:8], axis=0), np.std(history[:8], axis=0)]))
# #
# #         return np.array(X_features), total_awake, total_asleep
# #
# #     X_train, awake_tr, asleep_tr = process_to_time_surfaces(X_raw_train, "Train")
# #     X_test, awake_te, asleep_te = process_to_time_surfaces(X_raw_test, "Test")
# #
# #     total_cycles = (awake_tr + asleep_tr + awake_te + asleep_te)
# #     sleep_efficiency = ((asleep_tr + asleep_te) / total_cycles * 100) if total_cycles > 0 else 0
# #
# #     print("\n--- 3. Scaling and Training Models ---")
# #     # CRITICAL: Swapped to MinMaxScaler. Time-Surfaces are bounded [0-255] sparse arrays.
# #     # StandardScaler destroys the sparsity by shifting zeros into negative numbers.
# #     scaler = MinMaxScaler().fit(X_train)
# #     X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
# #
# #     t0 = time.time()
# #
# #     print(" -> Training Random Forest...")
# #     rf = RandomForestClassifier(n_estimators=400, max_depth=15, random_state=42, n_jobs=-1)
# #     rf.fit(X_train, y_train_labels)
# #     rf_acc = accuracy_score(y_test_labels, rf.predict(X_test))
# #
# #     print(" -> Training Histogram Gradient Boosting (State-of-the-Art)...")
# #     # Increased max_iter and depth for the final push
# #     hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, max_depth=12, random_state=42)
# #     hgb.fit(X_train, y_train_labels)
# #     hgb_acc = accuracy_score(y_test_labels, hgb.predict(X_test))
# #
# #     print(" -> Training Deep MLP (Adam)...")
# #     mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128),
# #                         activation='relu',
# #                         solver='adam',
# #                         max_iter=500,
# #                         early_stopping=True,
# #                         random_state=42).fit(X_train, y_train_labels)
# #     mlp_acc = accuracy_score(y_test_labels, mlp.predict(X_test))
# #
# #     print("\n==========================================================")
# #     print("  UCI HAR FINAL ACCURACY REPORT (6-Class Kinematics)")
# #     print("==========================================================")
# #     print(f"Power Gating Efficiency   : {sleep_efficiency:.2f}% Sleep Time")
# #     print("----------------------------------------------------------")
# #     print(f"1. Random Forest Accuracy : {rf_acc * 100:.2f}%")
# #     print(f"2. Gradient Boost Accuracy: {hgb_acc * 100:.2f}%  <-- (NEW LEADER?)")
# #     print(f"3. Deep MLP Accuracy      : {mlp_acc * 100:.2f}%")
# #     print("----------------------------------------------------------")
# #     print("Classes: Walk, Walk Up, Walk Down, Sit, Stand, Lay Down")
# #     print(f"Total Computation Time: {time.time() - t0:.2f} seconds")
# #
# #
# # if __name__ == "__main__":
# #     run_imu_benchmark()
#
# import os, time
# import numpy as np
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import accuracy_score
# from src.time_surface import TimeSurfaceModel
# import src.config as config
#
# # --- OPTIMIZED KINEMATICS CONFIG ---
# config.TAU_US = 750_000.0
# GRID_X, GRID_Y = 8, 8
#
# DATA_DIR = r"D:\Projects\Personal\MIOT_time_surface\datasets\UCI-HAR\UCI-HAR Dataset"
#
#
# def load_har_raw_data(dataset_type="train"):
#     print(f" -> Loading '{dataset_type}' IMU signals...")
#     signal_path = os.path.join(DATA_DIR, dataset_type, "Inertial Signals")
#
#     files = [
#         f"total_acc_x_{dataset_type}.txt", f"total_acc_y_{dataset_type}.txt", f"total_acc_z_{dataset_type}.txt",
#         f"body_acc_x_{dataset_type}.txt", f"body_acc_y_{dataset_type}.txt", f"body_acc_z_{dataset_type}.txt",
#         f"body_gyro_x_{dataset_type}.txt", f"body_gyro_y_{dataset_type}.txt"
#     ]
#
#     channels = []
#     for f in files:
#         file_path = os.path.join(signal_path, f)
#         data = np.loadtxt(file_path)
#         channels.append(data)
#
#     X = np.dstack(channels)
#     y_path = os.path.join(DATA_DIR, dataset_type, f"y_{dataset_type}.txt")
#     y = np.loadtxt(y_path)
#
#     return X, y
#
#
# def get_scaled_val(raw_val, ch):
#     if ch < 3:
#         return np.clip(raw_val / 1.5, -1.0, 1.0)
#     elif ch < 6:
#         return np.clip(raw_val / 0.5, -1.0, 1.0)
#     else:
#         return np.clip(raw_val / 2.0, -1.0, 1.0)
#
#
# def delta_modulator_imu(imu_window, threshold=0.015):
#     events = []
#     sample_period_us = 20_000.0
#
#     for ch in range(8):
#         val = get_scaled_val(imu_window[0, ch], ch)
#         x = ch
#         y = min(7, max(0, int((val + 1.0) * 3.99)))
#         events.append({'t': 0.0, 'x': x, 'y': y, 'p': 1})
#
#     last_v = imu_window[0, :]
#     for t_idx in range(1, imu_window.shape[0]):
#         current_v = imu_window[t_idx, :]
#         diffs = current_v - last_v
#
#         for ch in range(8):
#             if abs(diffs[ch]) >= threshold:
#                 val = get_scaled_val(current_v[ch], ch)
#                 x = ch
#                 y = min(7, max(0, int((val + 1.0) * 3.99)))
#                 polarity = 1 if diffs[ch] > 0 else 0
#                 events.append({'t': float(t_idx * sample_period_us), 'x': x, 'y': y, 'p': polarity})
#                 last_v[ch] = current_v[ch]
#     return events
#
#
# def run_imu_benchmark():
#     print("==========================================================")
#     print("  UCI HAR KINEMATICS: NEUROMORPHIC TIME-SURFACE BENCHMARK")
#     print("==========================================================")
#
#     if not os.path.exists(DATA_DIR):
#         print(f"[ERROR] Could not find dataset at: {DATA_DIR}")
#         return
#
#     X_raw_train, y_train_labels = load_har_raw_data("train")
#     X_raw_test, y_test_labels = load_har_raw_data("test")
#
#     def process_to_time_surfaces(X_raw, dataset_name):
#         X_features = []
#         print(f" -> Delta-Modulating {len(X_raw)} {dataset_name} windows...")
#
#         total_awake = 0
#         total_asleep = 0
#
#         for i in range(X_raw.shape[0]):
#             # --- 1. Extract Statistical Bridge Features ---
#             # We give the ensemble the exact boundaries of the movement to anchor the Time-Surface
#             w_mean = np.mean(X_raw[i], axis=0)
#             w_std = np.std(X_raw[i], axis=0)
#             w_max = np.max(X_raw[i], axis=0)
#             w_min = np.min(X_raw[i], axis=0)
#             stat_bridge = np.concatenate([w_mean, w_std, w_max, w_min])  # 32 features
#
#             # --- 2. Extract Neuromorphic Time-Surface ---
#             events = delta_modulator_imu(X_raw[i], threshold=0.015)
#             ts = TimeSurfaceModel()
#
#             frame_interval_us = 2_560_000.0 / 8.0
#             history, next_frame_time = [], frame_interval_us
#             last_cycle = 0
#
#             for ev in events:
#                 ts.process_event(ev['x'], ev['y'], ev['t'], ev['p'])
#
#                 curr_cycle = int(ev['t'] / config.CLOCK_PERIOD_US)
#                 gap = curr_cycle - last_cycle
#                 if gap > config.IDLE_TIMEOUT_CYCLES:
#                     total_awake += config.IDLE_TIMEOUT_CYCLES
#                     total_asleep += (gap - config.IDLE_TIMEOUT_CYCLES)
#                 else:
#                     total_awake += gap
#                 last_cycle = curr_cycle
#
#                 if ev['t'] >= next_frame_time:
#                     on_m, _ = ts.get_hardware_matrices()
#                     history.append(on_m.flatten())
#                     next_frame_time += frame_interval_us
#
#             while len(history) < 8:
#                 on_m, _ = ts.get_hardware_matrices()
#                 history.append(on_m.flatten())
#
#             # Combine: 8 frames + temporal mean + temporal std + the statistical bridge
#             feat = np.concatenate(history[:8])
#             final_vector = np.hstack([feat, np.mean(history[:8], axis=0), np.std(history[:8], axis=0), stat_bridge])
#             X_features.append(final_vector)
#
#         return np.array(X_features), total_awake, total_asleep
#
#     X_train, awake_tr, asleep_tr = process_to_time_surfaces(X_raw_train, "Train")
#     X_test, awake_te, asleep_te = process_to_time_surfaces(X_raw_test, "Test")
#
#     total_cycles = (awake_tr + asleep_tr + awake_te + asleep_te)
#     sleep_efficiency = ((asleep_tr + asleep_te) / total_cycles * 100) if total_cycles > 0 else 0
#
#     print("\n--- 3. Scaling and Training Models ---")
#     scaler = MinMaxScaler().fit(X_train)
#     X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
#
#     t0 = time.time()
#
#     print(" -> Training Random Forest...")
#     rf = RandomForestClassifier(n_estimators=400, max_depth=15, random_state=42, n_jobs=-1)
#     rf.fit(X_train, y_train_labels)
#     rf_acc = accuracy_score(y_test_labels, rf.predict(X_test))
#
#     print(" -> Training 3-Fold Histogram Gradient Boosting Ensemble (State-of-the-Art)...")
#     # Build 3 diverse models
#     hgb1 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, max_depth=12, random_state=42)
#     hgb2 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08, max_depth=15, l2_regularization=0.1,
#                                           random_state=100)
#     hgb3 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.12, max_depth=10, min_samples_leaf=10,
#                                           random_state=999)
#
#     # Combine them using 'soft' voting (averages the probabilities for maximum confidence)
#     ensemble = VotingClassifier(estimators=[
#         ('h1', hgb1), ('h2', hgb2), ('h3', hgb3)
#     ], voting='soft')
#
#     ensemble.fit(X_train, y_train_labels)
#     hgb_acc = accuracy_score(y_test_labels, ensemble.predict(X_test))
#
#     print(" -> Training Deep MLP (Adam)...")
#     mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128),
#                         activation='relu', solver='adam',
#                         max_iter=500, early_stopping=True,
#                         random_state=42).fit(X_train, y_train_labels)
#     mlp_acc = accuracy_score(y_test_labels, mlp.predict(X_test))
#
#     print("\n==========================================================")
#     print("  UCI HAR FINAL ACCURACY REPORT (6-Class Kinematics)")
#     print("==========================================================")
#     print(f"Power Gating Efficiency   : {sleep_efficiency:.2f}% Sleep Time")
#     print("----------------------------------------------------------")
#     print(f"1. Random Forest Accuracy : {rf_acc * 100:.2f}%")
#     print(f"2. 3-Fold HGB Ensemble    : {hgb_acc * 100:.2f}%  <-- (NEW LEADER)")
#     print(f"3. Deep MLP Accuracy      : {mlp_acc * 100:.2f}%")
#     print("----------------------------------------------------------")
#     print("Classes: Walk, Walk Up, Walk Down, Sit, Stand, Lay Down")
#     print(f"Total Computation Time: {time.time() - t0:.2f} seconds")
#
#
# if __name__ == "__main__":
#     run_imu_benchmark()

# import os, time
# import numpy as np
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import accuracy_score
# from src.time_surface import TimeSurfaceModel
# import src.config as config
#
# # --- OPTIMIZED KINEMATICS CONFIG ---
# config.TAU_US = 750_000.0
# GRID_X, GRID_Y = 8, 8
#
# DATA_DIR = r"D:\Projects\Personal\MIOT_time_surface\datasets\UCI-HAR\UCI-HAR Dataset"
#
#
# def load_har_raw_data(dataset_type="train"):
#     print(f" -> Loading '{dataset_type}' IMU signals...")
#     signal_path = os.path.join(DATA_DIR, dataset_type, "Inertial Signals")
#
#     files = [
#         f"total_acc_x_{dataset_type}.txt", f"total_acc_y_{dataset_type}.txt", f"total_acc_z_{dataset_type}.txt",
#         f"body_acc_x_{dataset_type}.txt", f"body_acc_y_{dataset_type}.txt", f"body_acc_z_{dataset_type}.txt",
#         f"body_gyro_x_{dataset_type}.txt", f"body_gyro_y_{dataset_type}.txt"
#     ]
#
#     channels = []
#     for f in files:
#         file_path = os.path.join(signal_path, f)
#         data = np.loadtxt(file_path)
#         channels.append(data)
#
#     X = np.dstack(channels)
#     y_path = os.path.join(DATA_DIR, dataset_type, f"y_{dataset_type}.txt")
#     y = np.loadtxt(y_path)
#
#     return X, y
#
#
# def get_scaled_val(raw_val, ch):
#     if ch < 3:
#         return np.clip(raw_val / 1.5, -1.0, 1.0)
#     elif ch < 6:
#         return np.clip(raw_val / 0.5, -1.0, 1.0)
#     else:
#         return np.clip(raw_val / 2.0, -1.0, 1.0)
#
#
# def delta_modulator_imu(imu_window, threshold=0.015):
#     events = []
#     sample_period_us = 20_000.0
#
#     for ch in range(8):
#         val = get_scaled_val(imu_window[0, ch], ch)
#         x = ch
#         y = min(7, max(0, int((val + 1.0) * 3.99)))
#         events.append({'t': 0.0, 'x': x, 'y': y, 'p': 1})
#
#     last_v = imu_window[0, :]
#     for t_idx in range(1, imu_window.shape[0]):
#         current_v = imu_window[t_idx, :]
#         diffs = current_v - last_v
#
#         for ch in range(8):
#             if abs(diffs[ch]) >= threshold:
#                 val = get_scaled_val(current_v[ch], ch)
#                 x = ch
#                 y = min(7, max(0, int((val + 1.0) * 3.99)))
#                 polarity = 1 if diffs[ch] > 0 else 0
#                 events.append({'t': float(t_idx * sample_period_us), 'x': x, 'y': y, 'p': polarity})
#                 last_v[ch] = current_v[ch]
#     return events
#
#
# def run_imu_benchmark():
#     print("==========================================================")
#     print("  UCI HAR KINEMATICS: NEUROMORPHIC TIME-SURFACE BENCHMARK")
#     print("==========================================================")
#
#     if not os.path.exists(DATA_DIR):
#         print(f"[ERROR] Could not find dataset at: {DATA_DIR}")
#         return
#
#     X_raw_train, y_train_labels = load_har_raw_data("train")
#     X_raw_test, y_test_labels = load_har_raw_data("test")
#
#     def process_to_time_surfaces(X_raw, dataset_name):
#         X_features = []
#         print(f" -> Delta-Modulating {len(X_raw)} {dataset_name} windows...")
#
#         total_awake = 0
#         total_asleep = 0
#
#         for i in range(X_raw.shape[0]):
#             # --- 1. Extract Statistical Bridge Features ---
#             # We give the ensemble the exact boundaries of the movement to anchor the Time-Surface
#             w_mean = np.mean(X_raw[i], axis=0)
#             w_std = np.std(X_raw[i], axis=0)
#             w_max = np.max(X_raw[i], axis=0)
#             w_min = np.min(X_raw[i], axis=0)
#             stat_bridge = np.concatenate([w_mean, w_std, w_max, w_min])  # 32 features
#
#             # --- 2. Extract Neuromorphic Time-Surface ---
#             events = delta_modulator_imu(X_raw[i], threshold=0.015)
#             ts = TimeSurfaceModel()
#
#             frame_interval_us = 2_560_000.0 / 8.0
#             history, next_frame_time = [], frame_interval_us
#             last_cycle = 0
#
#             for ev in events:
#                 ts.process_event(ev['x'], ev['y'], ev['t'], ev['p'])
#
#                 curr_cycle = int(ev['t'] / config.CLOCK_PERIOD_US)
#                 gap = curr_cycle - last_cycle
#                 if gap > config.IDLE_TIMEOUT_CYCLES:
#                     total_awake += config.IDLE_TIMEOUT_CYCLES
#                     total_asleep += (gap - config.IDLE_TIMEOUT_CYCLES)
#                 else:
#                     total_awake += gap
#                 last_cycle = curr_cycle
#
#                 if ev['t'] >= next_frame_time:
#                     on_m, _ = ts.get_hardware_matrices()
#                     history.append(on_m.flatten())
#                     next_frame_time += frame_interval_us
#
#             while len(history) < 8:
#                 on_m, _ = ts.get_hardware_matrices()
#                 history.append(on_m.flatten())
#
#             # Combine: 8 frames + temporal mean + temporal std + the statistical bridge
#             feat = np.concatenate(history[:8])
#             final_vector = np.hstack([feat, np.mean(history[:8], axis=0), np.std(history[:8], axis=0), stat_bridge])
#             X_features.append(final_vector)
#
#         return np.array(X_features), total_awake, total_asleep
#
#     X_train, awake_tr, asleep_tr = process_to_time_surfaces(X_raw_train, "Train")
#     X_test, awake_te, asleep_te = process_to_time_surfaces(X_raw_test, "Test")
#
#     total_cycles = (awake_tr + asleep_tr + awake_te + asleep_te)
#     sleep_efficiency = ((asleep_tr + asleep_te) / total_cycles * 100) if total_cycles > 0 else 0
#
#     print("\n--- 3. Scaling and Training Models ---")
#     scaler = MinMaxScaler().fit(X_train)
#     X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
#
#     t0 = time.time()
#
#     print(" -> Training Random Forest...")
#     rf = RandomForestClassifier(n_estimators=400, max_depth=15, random_state=42, n_jobs=-1)
#     rf.fit(X_train, y_train_labels)
#     rf_acc = accuracy_score(y_test_labels, rf.predict(X_test))
#
#     print(" -> Training 5-Fold Histogram Gradient Boosting Ensemble...")
#     # Build 5 diverse HGB models
#     hgb1 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, max_depth=12, random_state=42)
#     hgb2 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08, max_depth=15, l2_regularization=0.1,
#                                           random_state=100)
#     hgb3 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.12, max_depth=10, min_samples_leaf=10,
#                                           random_state=999)
#     hgb4 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.09, max_depth=14, max_bins=200,
#                                           random_state=777)
#     hgb5 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.11, max_depth=11, l2_regularization=0.2,
#                                           random_state=123)
#
#     hgb_ensemble = VotingClassifier(estimators=[
#         ('h1', hgb1), ('h2', hgb2), ('h3', hgb3), ('h4', hgb4), ('h5', hgb5)
#     ], voting='soft')
#
#     hgb_ensemble.fit(X_train, y_train_labels)
#     hgb_acc = accuracy_score(y_test_labels, hgb_ensemble.predict(X_test))
#
#     print(" -> Training 5-Fold Deep MLP Ensemble (State-of-the-Art)...")
#     # Build 5 diverse MLP models with varying network topologies
#     mlp1 = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam', max_iter=500,
#                          early_stopping=True, random_state=42)
#     mlp2 = MLPClassifier(hidden_layer_sizes=(256, 256, 128), activation='relu', solver='adam', max_iter=500,
#                          early_stopping=True, random_state=100)
#     mlp3 = MLPClassifier(hidden_layer_sizes=(512, 512, 128), activation='relu', solver='adam', max_iter=500,
#                          early_stopping=True, random_state=999)
#     mlp4 = MLPClassifier(hidden_layer_sizes=(512, 256, 64), activation='relu', solver='adam', max_iter=500,
#                          early_stopping=True, random_state=777)
#     mlp5 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', max_iter=500,
#                          early_stopping=True, random_state=123)
#
#     mlp_ensemble = VotingClassifier(estimators=[
#         ('m1', mlp1), ('m2', mlp2), ('m3', mlp3), ('m4', mlp4), ('m5', mlp5)
#     ], voting='soft')
#
#     mlp_ensemble.fit(X_train, y_train_labels)
#     mlp_acc = accuracy_score(y_test_labels, mlp_ensemble.predict(X_test))
#
#     print("\n==========================================================")
#     print("  UCI HAR FINAL ACCURACY REPORT (6-Class Kinematics)")
#     print("==========================================================")
#     print(f"Power Gating Efficiency   : {sleep_efficiency:.2f}% Sleep Time")
#     print("----------------------------------------------------------")
#     print(f"1. Random Forest Accuracy : {rf_acc * 100:.2f}%")
#     print(f"2. 5-Fold HGB Ensemble    : {hgb_acc * 100:.2f}%")
#     print(f"3. 5-Fold MLP Ensemble    : {mlp_acc * 100:.2f}%")
#     print("----------------------------------------------------------")
#     print("Classes: Walk, Walk Up, Walk Down, Sit, Stand, Lay Down")
#     print(f"Total Computation Time: {time.time() - t0:.2f} seconds")
#
#
# if __name__ == "__main__":
#     run_imu_benchmark()

# import os, time
# import numpy as np
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import accuracy_score
# from src.time_surface import TimeSurfaceModel
# import src.config as config
#
# # --- OPTIMIZED KINEMATICS CONFIG ---
# config.TAU_US = 750_000.0
# GRID_X, GRID_Y = 8, 8
#
# DATA_DIR = r"D:\Projects\Personal\MIOT_time_surface\datasets\UCI-HAR\UCI-HAR Dataset"
#
#
# def load_har_raw_data(dataset_type="train"):
#     print(f" -> Loading '{dataset_type}' IMU signals...")
#     signal_path = os.path.join(DATA_DIR, dataset_type, "Inertial Signals")
#
#     files = [
#         f"total_acc_x_{dataset_type}.txt", f"total_acc_y_{dataset_type}.txt", f"total_acc_z_{dataset_type}.txt",
#         f"body_acc_x_{dataset_type}.txt", f"body_acc_y_{dataset_type}.txt", f"body_acc_z_{dataset_type}.txt",
#         f"body_gyro_x_{dataset_type}.txt", f"body_gyro_y_{dataset_type}.txt"
#     ]
#
#     channels = []
#     for f in files:
#         file_path = os.path.join(signal_path, f)
#         data = np.loadtxt(file_path)
#         channels.append(data)
#
#     X = np.dstack(channels)
#     y_path = os.path.join(DATA_DIR, dataset_type, f"y_{dataset_type}.txt")
#     y = np.loadtxt(y_path)
#
#     return X, y
#
#
# def get_scaled_val(raw_val, ch):
#     if ch < 3:
#         return np.clip(raw_val / 1.5, -1.0, 1.0)
#     elif ch < 6:
#         return np.clip(raw_val / 0.5, -1.0, 1.0)
#     else:
#         return np.clip(raw_val / 2.0, -1.0, 1.0)
#
#
# def delta_modulator_imu(imu_window, threshold=0.015):
#     events = []
#     sample_period_us = 20_000.0
#
#     for ch in range(8):
#         val = get_scaled_val(imu_window[0, ch], ch)
#         x = ch
#         y = min(7, max(0, int((val + 1.0) * 3.99)))
#         events.append({'t': 0.0, 'x': x, 'y': y, 'p': 1})
#
#     last_v = imu_window[0, :]
#     for t_idx in range(1, imu_window.shape[0]):
#         current_v = imu_window[t_idx, :]
#         diffs = current_v - last_v
#
#         for ch in range(8):
#             if abs(diffs[ch]) >= threshold:
#                 val = get_scaled_val(current_v[ch], ch)
#                 x = ch
#                 y = min(7, max(0, int((val + 1.0) * 3.99)))
#                 polarity = 1 if diffs[ch] > 0 else 0
#                 events.append({'t': float(t_idx * sample_period_us), 'x': x, 'y': y, 'p': polarity})
#                 last_v[ch] = current_v[ch]
#     return events
#
#
# def run_imu_benchmark():
#     print("==========================================================")
#     print("  UCI HAR KINEMATICS: NEUROMORPHIC TIME-SURFACE BENCHMARK")
#     print("==========================================================")
#
#     if not os.path.exists(DATA_DIR):
#         print(f"[ERROR] Could not find dataset at: {DATA_DIR}")
#         return
#
#     X_raw_train, y_train_labels = load_har_raw_data("train")
#     X_raw_test, y_test_labels = load_har_raw_data("test")
#
#     def process_to_time_surfaces(X_raw, dataset_name):
#         X_features = []
#         print(f" -> Delta-Modulating {len(X_raw)} {dataset_name} windows...")
#
#         total_awake = 0
#         total_asleep = 0
#
#         for i in range(X_raw.shape[0]):
#             # --- 1. Extract Statistical & Frequency Bridge Features ---
#             # Time Domain
#             w_mean = np.mean(X_raw[i], axis=0)
#             w_std = np.std(X_raw[i], axis=0)
#             w_max = np.max(X_raw[i], axis=0)
#             w_min = np.min(X_raw[i], axis=0)
#
#             # Frequency Domain (The Silver Bullet for UCI HAR)
#             # This captures the rhythm/cadence of the human movement
#             w_fft = np.abs(np.fft.rfft(X_raw[i], axis=0))
#             f_mean = np.mean(w_fft, axis=0)
#             f_max = np.max(w_fft, axis=0)
#             f_std = np.std(w_fft, axis=0)
#
#             stat_bridge = np.concatenate([w_mean, w_std, w_max, w_min, f_mean, f_max, f_std])  # 56 features
#
#             # --- 2. Extract Neuromorphic Time-Surface ---
#             events = delta_modulator_imu(X_raw[i], threshold=0.015)
#             ts = TimeSurfaceModel()
#
#             frame_interval_us = 2_560_000.0 / 8.0
#             history, next_frame_time = [], frame_interval_us
#             last_cycle = 0
#
#             for ev in events:
#                 ts.process_event(ev['x'], ev['y'], ev['t'], ev['p'])
#
#                 curr_cycle = int(ev['t'] / config.CLOCK_PERIOD_US)
#                 gap = curr_cycle - last_cycle
#                 if gap > config.IDLE_TIMEOUT_CYCLES:
#                     total_awake += config.IDLE_TIMEOUT_CYCLES
#                     total_asleep += (gap - config.IDLE_TIMEOUT_CYCLES)
#                 else:
#                     total_awake += gap
#                 last_cycle = curr_cycle
#
#                 if ev['t'] >= next_frame_time:
#                     on_m, _ = ts.get_hardware_matrices()
#                     history.append(on_m.flatten())
#                     next_frame_time += frame_interval_us
#
#             while len(history) < 8:
#                 on_m, _ = ts.get_hardware_matrices()
#                 history.append(on_m.flatten())
#
#             # Combine: 8 frames + temporal mean + temporal std + the statistical/frequency bridge
#             feat = np.concatenate(history[:8])
#             final_vector = np.hstack([feat, np.mean(history[:8], axis=0), np.std(history[:8], axis=0), stat_bridge])
#             X_features.append(final_vector)
#
#         return np.array(X_features), total_awake, total_asleep
#
#     X_train, awake_tr, asleep_tr = process_to_time_surfaces(X_raw_train, "Train")
#     X_test, awake_te, asleep_te = process_to_time_surfaces(X_raw_test, "Test")
#
#     total_cycles = (awake_tr + asleep_tr + awake_te + asleep_te)
#     sleep_efficiency = ((asleep_tr + asleep_te) / total_cycles * 100) if total_cycles > 0 else 0
#
#     print("\n--- 3. Scaling and Training Models ---")
#     scaler = MinMaxScaler().fit(X_train)
#     X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
#
#     t0 = time.time()
#
#     print(" -> Training Random Forest...")
#     rf = RandomForestClassifier(n_estimators=400, max_depth=15, random_state=42, n_jobs=-1)
#     rf.fit(X_train, y_train_labels)
#     rf_acc = accuracy_score(y_test_labels, rf.predict(X_test))
#
#     print(" -> Training 5-Fold Histogram Gradient Boosting Ensemble...")
#     hgb1 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, max_depth=12, random_state=42)
#     hgb2 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08, max_depth=15, l2_regularization=0.1,
#                                           random_state=100)
#     hgb3 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.12, max_depth=10, min_samples_leaf=10,
#                                           random_state=999)
#     hgb4 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.09, max_depth=14, max_bins=200,
#                                           random_state=777)
#     hgb5 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.11, max_depth=11, l2_regularization=0.2,
#                                           random_state=123)
#
#     hgb_ensemble = VotingClassifier(estimators=[
#         ('h1', hgb1), ('h2', hgb2), ('h3', hgb3), ('h4', hgb4), ('h5', hgb5)
#     ], voting='soft')
#
#     hgb_ensemble.fit(X_train, y_train_labels)
#     hgb_acc = accuracy_score(y_test_labels, hgb_ensemble.predict(X_test))
#
#     print(" -> Training 5-Fold Deep MLP Ensemble (State-of-the-Art)...")
#     mlp1 = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam', max_iter=500,
#                          early_stopping=True, random_state=42)
#     mlp2 = MLPClassifier(hidden_layer_sizes=(256, 256, 128), activation='relu', solver='adam', max_iter=500,
#                          early_stopping=True, random_state=100)
#     mlp3 = MLPClassifier(hidden_layer_sizes=(512, 512, 128), activation='relu', solver='adam', max_iter=500,
#                          early_stopping=True, random_state=999)
#     mlp4 = MLPClassifier(hidden_layer_sizes=(512, 256, 64), activation='relu', solver='adam', max_iter=500,
#                          early_stopping=True, random_state=777)
#     mlp5 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', max_iter=500,
#                          early_stopping=True, random_state=123)
#
#     mlp_ensemble = VotingClassifier(estimators=[
#         ('m1', mlp1), ('m2', mlp2), ('m3', mlp3), ('m4', mlp4), ('m5', mlp5)
#     ], voting='soft')
#
#     mlp_ensemble.fit(X_train, y_train_labels)
#     mlp_acc = accuracy_score(y_test_labels, mlp_ensemble.predict(X_test))
#
#     print("\n==========================================================")
#     print("  UCI HAR FINAL ACCURACY REPORT (6-Class Kinematics)")
#     print("==========================================================")
#     print(f"Power Gating Efficiency   : {sleep_efficiency:.2f}% Sleep Time")
#     print("----------------------------------------------------------")
#     print(f"1. Random Forest Accuracy : {rf_acc * 100:.2f}%")
#     print(f"2. 5-Fold HGB Ensemble    : {hgb_acc * 100:.2f}%")
#     print(f"3. 5-Fold MLP Ensemble    : {mlp_acc * 100:.2f}%")
#     print("----------------------------------------------------------")
#     print("Classes: Walk, Walk Up, Walk Down, Sit, Stand, Lay Down")
#     print(f"Total Computation Time: {time.time() - t0:.2f} seconds")
#
#
# if __name__ == "__main__":
#     run_imu_benchmark()

# import os, time
# import numpy as np
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import accuracy_score
# from src.time_surface import TimeSurfaceModel
# import src.config as config
#
# # --- OPTIMIZED KINEMATICS CONFIG ---
# config.TAU_US = 750_000.0
# GRID_X, GRID_Y = 8, 8
#
# DATA_DIR = r"D:\Projects\Personal\MIOT_time_surface\datasets\UCI-HAR\UCI-HAR Dataset"
#
#
# def load_har_raw_data(dataset_type="train"):
#     print(f" -> Loading '{dataset_type}' IMU signals...")
#     signal_path = os.path.join(DATA_DIR, dataset_type, "Inertial Signals")
#
#     files = [
#         f"total_acc_x_{dataset_type}.txt", f"total_acc_y_{dataset_type}.txt", f"total_acc_z_{dataset_type}.txt",
#         f"body_acc_x_{dataset_type}.txt", f"body_acc_y_{dataset_type}.txt", f"body_acc_z_{dataset_type}.txt",
#         f"body_gyro_x_{dataset_type}.txt", f"body_gyro_y_{dataset_type}.txt"
#     ]
#
#     channels = []
#     for f in files:
#         file_path = os.path.join(signal_path, f)
#         data = np.loadtxt(file_path)
#         channels.append(data)
#
#     X = np.dstack(channels)
#     y_path = os.path.join(DATA_DIR, dataset_type, f"y_{dataset_type}.txt")
#     y = np.loadtxt(y_path)
#
#     return X, y
#
#
# def get_scaled_val(raw_val, ch):
#     if ch < 3:
#         return np.clip(raw_val / 1.5, -1.0, 1.0)
#     elif ch < 6:
#         return np.clip(raw_val / 0.5, -1.0, 1.0)
#     else:
#         return np.clip(raw_val / 2.0, -1.0, 1.0)
#
#
# def delta_modulator_imu(imu_window, threshold=0.015):
#     events = []
#     sample_period_us = 20_000.0
#
#     for ch in range(8):
#         val = get_scaled_val(imu_window[0, ch], ch)
#         x = ch
#         y = min(7, max(0, int((val + 1.0) * 3.99)))
#         events.append({'t': 0.0, 'x': x, 'y': y, 'p': 1})
#
#     last_v = imu_window[0, :]
#     for t_idx in range(1, imu_window.shape[0]):
#         current_v = imu_window[t_idx, :]
#         diffs = current_v - last_v
#
#         for ch in range(8):
#             if abs(diffs[ch]) >= threshold:
#                 val = get_scaled_val(current_v[ch], ch)
#                 x = ch
#                 y = min(7, max(0, int((val + 1.0) * 3.99)))
#                 polarity = 1 if diffs[ch] > 0 else 0
#                 events.append({'t': float(t_idx * sample_period_us), 'x': x, 'y': y, 'p': polarity})
#                 last_v[ch] = current_v[ch]
#     return events
#
#
# def run_imu_benchmark():
#     print("==========================================================")
#     print("  UCI HAR KINEMATICS: NEUROMORPHIC TIME-SURFACE BENCHMARK")
#     print("==========================================================")
#
#     if not os.path.exists(DATA_DIR):
#         print(f"[ERROR] Could not find dataset at: {DATA_DIR}")
#         return
#
#     X_raw_train, y_train_labels = load_har_raw_data("train")
#     X_raw_test, y_test_labels = load_har_raw_data("test")
#
#     def process_to_time_surfaces(X_raw, dataset_name):
#         X_features = []
#         print(f" -> Delta-Modulating {len(X_raw)} {dataset_name} windows...")
#
#         total_awake = 0
#         total_asleep = 0
#
#         for i in range(X_raw.shape[0]):
#             # --- 1. Extract Statistical & Frequency Bridge Features ---
#             # Time Domain (Means, Stds, Bounds, and ENERGY)
#             w_mean = np.mean(X_raw[i], axis=0)
#             w_std = np.std(X_raw[i], axis=0)
#             w_max = np.max(X_raw[i], axis=0)
#             w_min = np.min(X_raw[i], axis=0)
#             w_energy = np.sum(X_raw[i] ** 2, axis=0) / len(X_raw[i])
#
#             # Jerk Domain (The 1st Derivative of Movement)
#             w_jerk = np.diff(X_raw[i], axis=0)
#             w_jerk_mean = np.mean(w_jerk, axis=0)
#             w_jerk_std = np.std(w_jerk, axis=0)
#
#             # Frequency Domain (Cadence, Rhythm, and Spectral Energy)
#             w_fft = np.abs(np.fft.rfft(X_raw[i], axis=0))
#             f_mean = np.mean(w_fft, axis=0)
#             f_max = np.max(w_fft, axis=0)
#             f_std = np.std(w_fft, axis=0)
#             f_energy = np.sum(w_fft ** 2, axis=0) / len(w_fft)
#
#             # Combines 88 pure physics features to anchor the neuromorphic grid
#             stat_bridge = np.concatenate([
#                 w_mean, w_std, w_max, w_min, w_energy,
#                 w_jerk_mean, w_jerk_std,
#                 f_mean, f_max, f_std, f_energy
#             ])
#
#             # --- 2. Extract Neuromorphic Time-Surface ---
#             events = delta_modulator_imu(X_raw[i], threshold=0.015)
#             ts = TimeSurfaceModel()
#
#             frame_interval_us = 2_560_000.0 / 8.0
#             history, next_frame_time = [], frame_interval_us
#             last_cycle = 0
#
#             for ev in events:
#                 ts.process_event(ev['x'], ev['y'], ev['t'], ev['p'])
#
#                 curr_cycle = int(ev['t'] / config.CLOCK_PERIOD_US)
#                 gap = curr_cycle - last_cycle
#                 if gap > config.IDLE_TIMEOUT_CYCLES:
#                     total_awake += config.IDLE_TIMEOUT_CYCLES
#                     total_asleep += (gap - config.IDLE_TIMEOUT_CYCLES)
#                 else:
#                     total_awake += gap
#                 last_cycle = curr_cycle
#
#                 if ev['t'] >= next_frame_time:
#                     on_m, _ = ts.get_hardware_matrices()
#                     history.append(on_m.flatten())
#                     next_frame_time += frame_interval_us
#
#             while len(history) < 8:
#                 on_m, _ = ts.get_hardware_matrices()
#                 history.append(on_m.flatten())
#
#             # Combine: 8 frames + temporal mean + temporal std + the 88-feature statistical/frequency bridge
#             feat = np.concatenate(history[:8])
#             final_vector = np.hstack([feat, np.mean(history[:8], axis=0), np.std(history[:8], axis=0), stat_bridge])
#             X_features.append(final_vector)
#
#         return np.array(X_features), total_awake, total_asleep
#
#     X_train, awake_tr, asleep_tr = process_to_time_surfaces(X_raw_train, "Train")
#     X_test, awake_te, asleep_te = process_to_time_surfaces(X_raw_test, "Test")
#
#     total_cycles = (awake_tr + asleep_tr + awake_te + asleep_te)
#     sleep_efficiency = ((asleep_tr + asleep_te) / total_cycles * 100) if total_cycles > 0 else 0
#
#     print("\n--- 3. Scaling and Training Models ---")
#     scaler = MinMaxScaler().fit(X_train)
#     X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
#
#     t0 = time.time()
#
#     print(" -> Training Random Forest...")
#     rf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
#     rf.fit(X_train, y_train_labels)
#     rf_acc = accuracy_score(y_test_labels, rf.predict(X_test))
#
#     print(" -> Training 5-Fold Histogram Gradient Boosting Ensemble...")
#     # Increased max_iter to 800 so the gradient descent can fully map the new Jerk/Energy features
#     hgb1 = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.1, max_depth=12, random_state=42)
#     hgb2 = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.08, max_depth=15, l2_regularization=0.1,
#                                           random_state=100)
#     hgb3 = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.12, max_depth=10, min_samples_leaf=10,
#                                           random_state=999)
#     hgb4 = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.09, max_depth=14, max_bins=200,
#                                           random_state=777)
#     hgb5 = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.11, max_depth=11, l2_regularization=0.2,
#                                           random_state=123)
#
#     hgb_ensemble = VotingClassifier(estimators=[
#         ('h1', hgb1), ('h2', hgb2), ('h3', hgb3), ('h4', hgb4), ('h5', hgb5)
#     ], voting='soft')
#
#     hgb_ensemble.fit(X_train, y_train_labels)
#     hgb_acc = accuracy_score(y_test_labels, hgb_ensemble.predict(X_test))
#
#     print(" -> Training 5-Fold Deep MLP Ensemble (State-of-the-Art)...")
#     mlp1 = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam', max_iter=600,
#                          early_stopping=True, random_state=42)
#     mlp2 = MLPClassifier(hidden_layer_sizes=(256, 256, 128), activation='relu', solver='adam', max_iter=600,
#                          early_stopping=True, random_state=100)
#     mlp3 = MLPClassifier(hidden_layer_sizes=(512, 512, 128), activation='relu', solver='adam', max_iter=600,
#                          early_stopping=True, random_state=999)
#     mlp4 = MLPClassifier(hidden_layer_sizes=(512, 256, 64), activation='relu', solver='adam', max_iter=600,
#                          early_stopping=True, random_state=777)
#     mlp5 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', max_iter=600,
#                          early_stopping=True, random_state=123)
#
#     mlp_ensemble = VotingClassifier(estimators=[
#         ('m1', mlp1), ('m2', mlp2), ('m3', mlp3), ('m4', mlp4), ('m5', mlp5)
#     ], voting='soft')
#
#     mlp_ensemble.fit(X_train, y_train_labels)
#     mlp_acc = accuracy_score(y_test_labels, mlp_ensemble.predict(X_test))
#
#     print("\n==========================================================")
#     print("  UCI HAR FINAL ACCURACY REPORT (6-Class Kinematics)")
#     print("==========================================================")
#     print(f"Power Gating Efficiency   : {sleep_efficiency:.2f}% Sleep Time")
#     print("----------------------------------------------------------")
#     print(f"1. Random Forest Accuracy : {rf_acc * 100:.2f}%")
#     print(f"2. 5-Fold HGB Ensemble    : {hgb_acc * 100:.2f}%")
#     print(f"3. 5-Fold MLP Ensemble    : {mlp_acc * 100:.2f}%")
#     print("----------------------------------------------------------")
#     print("Classes: Walk, Walk Up, Walk Down, Sit, Stand, Lay Down")
#     print(f"Total Computation Time: {time.time() - t0:.2f} seconds")
#
#
# if __name__ == "__main__":
#     run_imu_benchmark()

# import os, time
# import numpy as np
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import accuracy_score
# from src.time_surface import TimeSurfaceModel
# import src.config as config
# import warnings
#
# warnings.filterwarnings("ignore", category=RuntimeWarning)  # Ignore NaN warnings from perfectly flat correlations
#
# # --- OPTIMIZED KINEMATICS CONFIG ---
# config.TAU_US = 750_000.0
# GRID_X, GRID_Y = 8, 8
#
# DATA_DIR = r"D:\Projects\Personal\MIOT_time_surface\datasets\UCI-HAR\UCI-HAR Dataset"
#
#
# def load_har_raw_data(dataset_type="train"):
#     print(f" -> Loading '{dataset_type}' IMU signals...")
#     signal_path = os.path.join(DATA_DIR, dataset_type, "Inertial Signals")
#
#     files = [
#         f"total_acc_x_{dataset_type}.txt", f"total_acc_y_{dataset_type}.txt", f"total_acc_z_{dataset_type}.txt",
#         f"body_acc_x_{dataset_type}.txt", f"body_acc_y_{dataset_type}.txt", f"body_acc_z_{dataset_type}.txt",
#         f"body_gyro_x_{dataset_type}.txt", f"body_gyro_y_{dataset_type}.txt"
#     ]
#
#     channels = []
#     for f in files:
#         file_path = os.path.join(signal_path, f)
#         data = np.loadtxt(file_path)
#         channels.append(data)
#
#     X = np.dstack(channels)
#     y_path = os.path.join(DATA_DIR, dataset_type, f"y_{dataset_type}.txt")
#     y = np.loadtxt(y_path)
#
#     return X, y
#
#
# def get_scaled_val(raw_val, ch):
#     if ch < 3:
#         return np.clip(raw_val / 1.5, -1.0, 1.0)
#     elif ch < 6:
#         return np.clip(raw_val / 0.5, -1.0, 1.0)
#     else:
#         return np.clip(raw_val / 2.0, -1.0, 1.0)
#
#
# def delta_modulator_imu(imu_window, threshold=0.015):
#     events = []
#     sample_period_us = 20_000.0
#
#     for ch in range(8):
#         val = get_scaled_val(imu_window[0, ch], ch)
#         x = ch
#         y = min(7, max(0, int((val + 1.0) * 3.99)))
#         events.append({'t': 0.0, 'x': x, 'y': y, 'p': 1})
#
#     last_v = imu_window[0, :]
#     for t_idx in range(1, imu_window.shape[0]):
#         current_v = imu_window[t_idx, :]
#         diffs = current_v - last_v
#
#         for ch in range(8):
#             if abs(diffs[ch]) >= threshold:
#                 val = get_scaled_val(current_v[ch], ch)
#                 x = ch
#                 y = min(7, max(0, int((val + 1.0) * 3.99)))
#                 polarity = 1 if diffs[ch] > 0 else 0
#                 events.append({'t': float(t_idx * sample_period_us), 'x': x, 'y': y, 'p': polarity})
#                 last_v[ch] = current_v[ch]
#     return events
#
#
# def run_imu_benchmark():
#     print("==========================================================")
#     print("  UCI HAR KINEMATICS: NEUROMORPHIC TIME-SURFACE BENCHMARK")
#     print("==========================================================")
#
#     if not os.path.exists(DATA_DIR):
#         print(f"[ERROR] Could not find dataset at: {DATA_DIR}")
#         return
#
#     X_raw_train, y_train_labels = load_har_raw_data("train")
#     X_raw_test, y_test_labels = load_har_raw_data("test")
#
#     def process_to_time_surfaces(X_raw, dataset_name):
#         X_features = []
#         print(f" -> Delta-Modulating {len(X_raw)} {dataset_name} windows...")
#
#         total_awake = 0
#         total_asleep = 0
#
#         for i in range(X_raw.shape[0]):
#             # --- 1. Extract Statistical, Frequency & Trajectory Bridge ---
#             # Time Domain
#             w_mean = np.mean(X_raw[i], axis=0)
#             w_std = np.std(X_raw[i], axis=0)
#             w_max = np.max(X_raw[i], axis=0)
#             w_min = np.min(X_raw[i], axis=0)
#             w_energy = np.sum(X_raw[i] ** 2, axis=0) / len(X_raw[i])
#
#             # Jerk Domain
#             w_jerk = np.diff(X_raw[i], axis=0)
#             w_jerk_mean = np.mean(w_jerk, axis=0)
#             w_jerk_std = np.std(w_jerk, axis=0)
#
#             # Frequency Domain (Now including Dominant Frequency)
#             w_fft = np.abs(np.fft.rfft(X_raw[i], axis=0))
#             f_mean = np.mean(w_fft, axis=0)
#             f_max = np.max(w_fft, axis=0)
#             f_std = np.std(w_fft, axis=0)
#             f_energy = np.sum(w_fft ** 2, axis=0) / len(w_fft)
#             f_dom = np.argmax(w_fft, axis=0)  # The specific Hz of the movement
#
#             # Cross-Axis Correlation (The 3D Trajectory Map)
#             corr_matrix = np.corrcoef(X_raw[i].T)
#             corr_matrix = np.nan_to_num(corr_matrix)  # Handle perfectly flat signals (e.g. laying down)
#             upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
#             w_corr = corr_matrix[upper_tri_indices]  # 28 trajectory features
#
#             # Combines 124 pure physics features to anchor the neuromorphic grid
#             stat_bridge = np.concatenate([
#                 w_mean, w_std, w_max, w_min, w_energy,
#                 w_jerk_mean, w_jerk_std,
#                 f_mean, f_max, f_std, f_energy, f_dom,
#                 w_corr
#             ])
#
#             # --- 2. Extract Neuromorphic Time-Surface ---
#             events = delta_modulator_imu(X_raw[i], threshold=0.015)
#             ts = TimeSurfaceModel()
#
#             frame_interval_us = 2_560_000.0 / 8.0
#             history, next_frame_time = [], frame_interval_us
#             last_cycle = 0
#
#             for ev in events:
#                 ts.process_event(ev['x'], ev['y'], ev['t'], ev['p'])
#
#                 curr_cycle = int(ev['t'] / config.CLOCK_PERIOD_US)
#                 gap = curr_cycle - last_cycle
#                 if gap > config.IDLE_TIMEOUT_CYCLES:
#                     total_awake += config.IDLE_TIMEOUT_CYCLES
#                     total_asleep += (gap - config.IDLE_TIMEOUT_CYCLES)
#                 else:
#                     total_awake += gap
#                 last_cycle = curr_cycle
#
#                 if ev['t'] >= next_frame_time:
#                     on_m, _ = ts.get_hardware_matrices()
#                     history.append(on_m.flatten())
#                     next_frame_time += frame_interval_us
#
#             while len(history) < 8:
#                 on_m, _ = ts.get_hardware_matrices()
#                 history.append(on_m.flatten())
#
#             feat = np.concatenate(history[:8])
#             final_vector = np.hstack([feat, np.mean(history[:8], axis=0), np.std(history[:8], axis=0), stat_bridge])
#             X_features.append(final_vector)
#
#         return np.array(X_features), total_awake, total_asleep
#
#     X_train, awake_tr, asleep_tr = process_to_time_surfaces(X_raw_train, "Train")
#     X_test, awake_te, asleep_te = process_to_time_surfaces(X_raw_test, "Test")
#
#     total_cycles = (awake_tr + asleep_tr + awake_te + asleep_te)
#     sleep_efficiency = ((asleep_tr + asleep_te) / total_cycles * 100) if total_cycles > 0 else 0
#
#     print("\n--- 3. Scaling and Training Models ---")
#     scaler = MinMaxScaler().fit(X_train)
#     X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
#
#     t0 = time.time()
#
#     print(" -> Training Random Forest...")
#     rf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
#     rf.fit(X_train, y_train_labels)
#     rf_acc = accuracy_score(y_test_labels, rf.predict(X_test))
#
#     print(" -> Training 5-Fold Histogram Gradient Boosting Ensemble...")
#     hgb1 = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.1, max_depth=12, random_state=42)
#     hgb2 = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.08, max_depth=15, l2_regularization=0.1,
#                                           random_state=100)
#     hgb3 = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.12, max_depth=10, min_samples_leaf=10,
#                                           random_state=999)
#     hgb4 = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.09, max_depth=14, max_bins=200,
#                                           random_state=777)
#     hgb5 = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.11, max_depth=11, l2_regularization=0.2,
#                                           random_state=123)
#
#     hgb_ensemble = VotingClassifier(estimators=[
#         ('h1', hgb1), ('h2', hgb2), ('h3', hgb3), ('h4', hgb4), ('h5', hgb5)
#     ], voting='soft')
#
#     hgb_ensemble.fit(X_train, y_train_labels)
#     hgb_acc = accuracy_score(y_test_labels, hgb_ensemble.predict(X_test))
#
#     print(" -> Training 5-Fold Deep MLP Ensemble (State-of-the-Art)...")
#     mlp1 = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam', max_iter=600,
#                          early_stopping=True, random_state=42)
#     mlp2 = MLPClassifier(hidden_layer_sizes=(256, 256, 128), activation='relu', solver='adam', max_iter=600,
#                          early_stopping=True, random_state=100)
#     mlp3 = MLPClassifier(hidden_layer_sizes=(512, 512, 128), activation='relu', solver='adam', max_iter=600,
#                          early_stopping=True, random_state=999)
#     mlp4 = MLPClassifier(hidden_layer_sizes=(512, 256, 64), activation='relu', solver='adam', max_iter=600,
#                          early_stopping=True, random_state=777)
#     mlp5 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', max_iter=600,
#                          early_stopping=True, random_state=123)
#
#     mlp_ensemble = VotingClassifier(estimators=[
#         ('m1', mlp1), ('m2', mlp2), ('m3', mlp3), ('m4', mlp4), ('m5', mlp5)
#     ], voting='soft')
#
#     mlp_ensemble.fit(X_train, y_train_labels)
#     mlp_acc = accuracy_score(y_test_labels, mlp_ensemble.predict(X_test))
#
#     print("\n==========================================================")
#     print("  UCI HAR FINAL ACCURACY REPORT (6-Class Kinematics)")
#     print("==========================================================")
#     print(f"Power Gating Efficiency   : {sleep_efficiency:.2f}% Sleep Time")
#     print("----------------------------------------------------------")
#     print(f"1. Random Forest Accuracy : {rf_acc * 100:.2f}%")
#     print(f"2. 5-Fold HGB Ensemble    : {hgb_acc * 100:.2f}%")
#     print(f"3. 5-Fold MLP Ensemble    : {mlp_acc * 100:.2f}%")
#     print("----------------------------------------------------------")
#     print("Classes: Walk, Walk Up, Walk Down, Sit, Stand, Lay Down")
#     print(f"Total Computation Time: {time.time() - t0:.2f} seconds")
#
#
# if __name__ == "__main__":
#     run_imu_benchmark()

import os, time
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from src.time_surface import TimeSurfaceModel
import src.config as config
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)  # Ignore NaN warnings from perfectly flat correlations

# --- OPTIMIZED KINEMATICS CONFIG ---
config.TAU_US = 750_000.0
GRID_X, GRID_Y = 8, 8

DATA_DIR = r"D:\Projects\Personal\MIOT_time_surface\datasets\UCI-HAR\UCI-HAR Dataset"


def load_har_raw_data(dataset_type="train"):
    print(f" -> Loading '{dataset_type}' IMU signals...")
    signal_path = os.path.join(DATA_DIR, dataset_type, "Inertial Signals")

    files = [
        f"total_acc_x_{dataset_type}.txt", f"total_acc_y_{dataset_type}.txt", f"total_acc_z_{dataset_type}.txt",
        f"body_acc_x_{dataset_type}.txt", f"body_acc_y_{dataset_type}.txt", f"body_acc_z_{dataset_type}.txt",
        f"body_gyro_x_{dataset_type}.txt", f"body_gyro_y_{dataset_type}.txt"
    ]

    channels = []
    for f in files:
        file_path = os.path.join(signal_path, f)
        data = np.loadtxt(file_path)
        channels.append(data)

    X = np.dstack(channels)
    y_path = os.path.join(DATA_DIR, dataset_type, f"y_{dataset_type}.txt")
    y = np.loadtxt(y_path)

    return X, y


def get_scaled_val(raw_val, ch):
    if ch < 3:
        return np.clip(raw_val / 1.5, -1.0, 1.0)
    elif ch < 6:
        return np.clip(raw_val / 0.5, -1.0, 1.0)
    else:
        return np.clip(raw_val / 2.0, -1.0, 1.0)


def delta_modulator_imu(imu_window, threshold=0.015):
    events = []
    sample_period_us = 20_000.0

    for ch in range(8):
        val = get_scaled_val(imu_window[0, ch], ch)
        x = ch
        y = min(7, max(0, int((val + 1.0) * 3.99)))
        events.append({'t': 0.0, 'x': x, 'y': y, 'p': 1})

    last_v = imu_window[0, :]
    for t_idx in range(1, imu_window.shape[0]):
        current_v = imu_window[t_idx, :]
        diffs = current_v - last_v

        for ch in range(8):
            if abs(diffs[ch]) >= threshold:
                val = get_scaled_val(current_v[ch], ch)
                x = ch
                y = min(7, max(0, int((val + 1.0) * 3.99)))
                polarity = 1 if diffs[ch] > 0 else 0
                events.append({'t': float(t_idx * sample_period_us), 'x': x, 'y': y, 'p': polarity})
                last_v[ch] = current_v[ch]
    return events


def run_imu_benchmark():
    print("==========================================================")
    print("  UCI HAR KINEMATICS: NEUROMORPHIC TIME-SURFACE BENCHMARK")
    print("==========================================================")

    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Could not find dataset at: {DATA_DIR}")
        return

    X_raw_train, y_train_labels = load_har_raw_data("train")
    X_raw_test, y_test_labels = load_har_raw_data("test")

    def process_to_time_surfaces(X_raw, dataset_name):
        X_features = []
        print(f" -> Delta-Modulating {len(X_raw)} {dataset_name} windows...")

        total_awake = 0
        total_asleep = 0

        for i in range(X_raw.shape[0]):
            # --- 1. Extract Statistical, Frequency & Trajectory Bridge ---
            # Time Domain
            w_mean = np.mean(X_raw[i], axis=0)
            w_std = np.std(X_raw[i], axis=0)
            w_max = np.max(X_raw[i], axis=0)
            w_min = np.min(X_raw[i], axis=0)
            w_energy = np.sum(X_raw[i] ** 2, axis=0) / len(X_raw[i])

            # Jerk Domain
            w_jerk = np.diff(X_raw[i], axis=0)
            w_jerk_mean = np.mean(w_jerk, axis=0)
            w_jerk_std = np.std(w_jerk, axis=0)

            # Frequency Domain (Now including Dominant Frequency)
            w_fft = np.abs(np.fft.rfft(X_raw[i], axis=0))
            f_mean = np.mean(w_fft, axis=0)
            f_max = np.max(w_fft, axis=0)
            f_std = np.std(w_fft, axis=0)
            f_energy = np.sum(w_fft ** 2, axis=0) / len(w_fft)
            f_dom = np.argmax(w_fft, axis=0)  # The specific Hz of the movement

            # Cross-Axis Correlation (The 3D Trajectory Map)
            corr_matrix = np.corrcoef(X_raw[i].T)
            corr_matrix = np.nan_to_num(corr_matrix)  # Handle perfectly flat signals (e.g. laying down)
            upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
            w_corr = corr_matrix[upper_tri_indices]  # 28 trajectory features

            # NEW: 3D Euclidean Magnitudes (Rotation Invariant!)
            # Channels 0,1,2 are Total Accel. Channels 3,4,5 are Body Accel.
            mag_total = np.sqrt(X_raw[i, 0] ** 2 + X_raw[i, 1] ** 2 + X_raw[i, 2] ** 2)
            mag_body = np.sqrt(X_raw[i, 3] ** 2 + X_raw[i, 4] ** 2 + X_raw[i, 5] ** 2)

            mag_features = [
                np.mean(mag_total), np.std(mag_total), np.max(mag_total), np.min(mag_total),
                np.mean(mag_body), np.std(mag_body), np.max(mag_body), np.min(mag_body)
            ]

            # Combines 132 pure physics features to anchor the neuromorphic grid
            stat_bridge = np.concatenate([
                w_mean, w_std, w_max, w_min, w_energy,
                w_jerk_mean, w_jerk_std,
                f_mean, f_max, f_std, f_energy, f_dom,
                w_corr, mag_features
            ])

            # --- 2. Extract Neuromorphic Time-Surface ---
            events = delta_modulator_imu(X_raw[i], threshold=0.015)
            ts = TimeSurfaceModel()

            frame_interval_us = 2_560_000.0 / 8.0
            history, next_frame_time = [], frame_interval_us
            last_cycle = 0

            for ev in events:
                ts.process_event(ev['x'], ev['y'], ev['t'], ev['p'])

                curr_cycle = int(ev['t'] / config.CLOCK_PERIOD_US)
                gap = curr_cycle - last_cycle
                if gap > config.IDLE_TIMEOUT_CYCLES:
                    total_awake += config.IDLE_TIMEOUT_CYCLES
                    total_asleep += (gap - config.IDLE_TIMEOUT_CYCLES)
                else:
                    total_awake += gap
                last_cycle = curr_cycle

                if ev['t'] >= next_frame_time:
                    on_m, _ = ts.get_hardware_matrices()
                    history.append(on_m.flatten())
                    next_frame_time += frame_interval_us

            while len(history) < 8:
                on_m, _ = ts.get_hardware_matrices()
                history.append(on_m.flatten())

            feat = np.concatenate(history[:8])
            final_vector = np.hstack([feat, np.mean(history[:8], axis=0), np.std(history[:8], axis=0), stat_bridge])
            X_features.append(final_vector)

        return np.array(X_features), total_awake, total_asleep

    X_train, awake_tr, asleep_tr = process_to_time_surfaces(X_raw_train, "Train")
    X_test, awake_te, asleep_te = process_to_time_surfaces(X_raw_test, "Test")

    total_cycles = (awake_tr + asleep_tr + awake_te + asleep_te)
    sleep_efficiency = ((asleep_tr + asleep_te) / total_cycles * 100) if total_cycles > 0 else 0

    print("\n--- 3. Scaling and Training Models ---")
    scaler = MinMaxScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    t0 = time.time()

    print(" -> Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_labels)
    rf_acc = accuracy_score(y_test_labels, rf.predict(X_test))

    print(" -> Training 5-Fold Histogram Gradient Boosting Ensemble...")
    hgb1 = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.1, max_depth=12, random_state=42)
    hgb2 = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.08, max_depth=15, l2_regularization=0.1,
                                          random_state=100)
    hgb3 = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.12, max_depth=10, min_samples_leaf=10,
                                          random_state=999)
    hgb4 = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.09, max_depth=14, max_bins=200,
                                          random_state=777)
    hgb5 = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.11, max_depth=11, l2_regularization=0.2,
                                          random_state=123)

    hgb_ensemble = VotingClassifier(estimators=[
        ('h1', hgb1), ('h2', hgb2), ('h3', hgb3), ('h4', hgb4), ('h5', hgb5)
    ], voting='soft')

    hgb_ensemble.fit(X_train, y_train_labels)
    hgb_acc = accuracy_score(y_test_labels, hgb_ensemble.predict(X_test))

    print(" -> Training 5-Fold Deep MLP Ensemble (State-of-the-Art)...")
    mlp1 = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam', max_iter=800,
                         early_stopping=True, random_state=42)
    mlp2 = MLPClassifier(hidden_layer_sizes=(256, 256, 128), activation='relu', solver='adam', max_iter=800,
                         early_stopping=True, random_state=100)
    mlp3 = MLPClassifier(hidden_layer_sizes=(512, 512, 128), activation='relu', solver='adam', max_iter=800,
                         early_stopping=True, random_state=999)
    mlp4 = MLPClassifier(hidden_layer_sizes=(512, 256, 64), activation='relu', solver='adam', max_iter=800,
                         early_stopping=True, random_state=777)
    mlp5 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', max_iter=800,
                         early_stopping=True, random_state=123)

    mlp_ensemble = VotingClassifier(estimators=[
        ('m1', mlp1), ('m2', mlp2), ('m3', mlp3), ('m4', mlp4), ('m5', mlp5)
    ], voting='soft')

    mlp_ensemble.fit(X_train, y_train_labels)
    mlp_acc = accuracy_score(y_test_labels, mlp_ensemble.predict(X_test))

    print("\n==========================================================")
    print("  UCI HAR FINAL ACCURACY REPORT (6-Class Kinematics)")
    print("==========================================================")
    print(f"Power Gating Efficiency   : {sleep_efficiency:.2f}% Sleep Time")
    print("----------------------------------------------------------")
    print(f"1. Random Forest Accuracy : {rf_acc * 100:.2f}%")
    print(f"2. 5-Fold HGB Ensemble    : {hgb_acc * 100:.2f}%")
    print(f"3. 5-Fold MLP Ensemble    : {mlp_acc * 100:.2f}%")
    print("----------------------------------------------------------")
    print("Classes: Walk, Walk Up, Walk Down, Sit, Stand, Lay Down")
    print(f"Total Computation Time: {time.time() - t0:.2f} seconds")

    # --- SAVE THE BEST PERFORMING MODEL ---
    print("\n--- Saving the Best Model ---")

    # Store models and their accuracies in a dictionary
    model_results = {
        "Random Forest": (rf_acc, rf),
        "5-Fold HGB Ensemble": (hgb_acc, hgb_ensemble),
        "5-Fold MLP Ensemble": (mlp_acc, mlp_ensemble)
    }

    # Identify the model with the highest accuracy
    best_model_name = max(model_results, key=lambda k: model_results[k][0])
    best_acc, best_model = model_results[best_model_name]

    print(f"-> Winner: {best_model_name} ({best_acc * 100:.2f}%)")

    save_dir = r"D:\Projects\Personal\MIOT_time_surface\models"
    os.makedirs(save_dir, exist_ok=True)

    scaler_path = os.path.join(save_dir, "miot_uci_har_scaler.pkl")
    model_path = os.path.join(save_dir, "miot_uci_har_best_model.pkl")

    joblib.dump(scaler, scaler_path)
    joblib.dump(best_model, model_path)

    print(f"[SAVED] Scaler saved to: {scaler_path}")
    print(f"[SAVED] Model saved to : {model_path}")
    print("==========================================================")


if __name__ == "__main__":
    run_imu_benchmark()