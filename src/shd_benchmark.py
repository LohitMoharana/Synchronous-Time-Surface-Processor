# # # # import os, time, sys
# # # # import numpy as np
# # # # from sklearn.neural_network import MLPClassifier
# # # # from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
# # # # from sklearn.preprocessing import StandardScaler
# # # # from sklearn.metrics import accuracy_score
# # # # from src.time_surface import TimeSurfaceModel
# # # # import src.config as config
# # # #
# # # # try:
# # # #     import h5py
# # # # except ImportError:
# # # #     print("[ERROR] The 'h5py' library is required to read SHD datasets. Please run: pip install h5py")
# # # #     sys.exit(1)
# # # #
# # # # # --- OPTIMIZED AUDIO CONFIG ---
# # # # # Speech is much faster than human walking. Syllables last roughly 50-150ms.
# # # # # We set the Time-Surface decay to 100ms so formants fade just in time for the next syllable.
# # # # config.TAU_US = 100_000.0
# # # # GRID_X, GRID_Y = 8, 8
# # # #
# # # # # Path to the unzipped SHD dataset HDF5 files
# # # # DATA_DIR = r"D:\Projects\Personal\MIOT_time_surface\datasets\SHD"
# # # #
# # # #
# # # # def load_shd_data(file_name):
# # # #     """Reads native spiking data directly from the SHD HDF5 archives."""
# # # #     file_path = os.path.join(DATA_DIR, file_name)
# # # #     if not os.path.exists(file_path):
# # # #         print(f"[ERROR] Could not find SHD dataset at: {file_path}")
# # # #         print("Please ensure 'shd_train.h5' and 'shd_test.h5' are downloaded and extracted.")
# # # #         sys.exit(1)
# # # #
# # # #     print(f" -> Loading native spikes from '{file_name}'...")
# # # #
# # # #     with h5py.File(file_path, 'r') as f:
# # # #         # Load labels cleanly
# # # #         labels = np.array(f['labels'], dtype=np.int32)
# # # #
# # # #         # References to the Variable-Length (jagged) HDF5 datasets
# # # #         times_ds = f['spikes']['times']
# # # #         units_ds = f['spikes']['units']
# # # #
# # # #         samples = []
# # # #         for i in range(len(labels)):
# # # #             # Explicitly extract each jagged array and cast to native float64/int32
# # # #             # This completely prevents the Numpy 'object' array NaN overflow bug.
# # # #             t_arr = np.array(times_ds[i], dtype=np.float64)
# # # #             u_arr = np.array(units_ds[i], dtype=np.int32)
# # # #
# # # #             # Bulletproof check to filter out any corrupted NaNs in the raw dataset
# # # #             valid_mask = ~np.isnan(t_arr)
# # # #
# # # #             samples.append({
# # # #                 'times': t_arr[valid_mask],
# # # #                 'units': u_arr[valid_mask]
# # # #             })
# # # #
# # # #     return samples, labels
# # # #
# # # #
# # # # def process_shd_to_surfaces(samples, dataset_name):
# # # #     print(f" -> Routing {len(samples)} {dataset_name} audio spikes to the 8x8 Grid...")
# # # #     X_features = []
# # # #
# # # #     total_awake = 0
# # # #     total_asleep = 0
# # # #
# # # #     for i, sample in enumerate(samples):
# # # #         ts = TimeSurfaceModel()
# # # #
# # # #         spike_times = sample['times']  # in seconds
# # # #         spike_units = sample['units']  # 0 to 699 frequencies
# # # #
# # # #         # Audio samples in SHD are typically ~1.0 second long.
# # # #         # We split the 1-second audio into 8 visual frames (125ms per frame)
# # # #         duration = float(np.max(spike_times)) if len(spike_times) > 0 else 1.0
# # # #         frame_interval_us = (duration * 1_000_000.0) / 8.0
# # # #         next_frame_time = frame_interval_us
# # # #         history = []
# # # #         last_cycle = 0
# # # #
# # # #         for t_sec, unit in zip(spike_times, spike_units):
# # # #             t_us = float(t_sec) * 1_000_000.0
# # # #
# # # #             # --- THE SILICON COCHLEA MAPPING ---
# # # #             # Compress 700 frequencies down to our 64 pixels.
# # # #             # 700 / 11 ~ 63. So dividing by 11 safely maps all frequencies into 0-63.
# # # #             pixel_id = min(63, int(unit) // 11)
# # # #             x = pixel_id % 8
# # # #             y = pixel_id // 8
# # # #
# # # #             # Process the raw biological spike directly!
# # # #             ts.process_event(x, y, t_us, polarity=1)
# # # #
# # # #             # Power Gating Logic
# # # #             curr_cycle = int(t_us / config.CLOCK_PERIOD_US)
# # # #             gap = curr_cycle - last_cycle
# # # #             if gap > config.IDLE_TIMEOUT_CYCLES:
# # # #                 total_awake += config.IDLE_TIMEOUT_CYCLES
# # # #                 total_asleep += (gap - config.IDLE_TIMEOUT_CYCLES)
# # # #             else:
# # # #                 total_awake += gap
# # # #             last_cycle = curr_cycle
# # # #
# # # #             if t_us >= next_frame_time:
# # # #                 on_m, _ = ts.get_hardware_matrices()
# # # #                 history.append(on_m.flatten())
# # # #                 next_frame_time += frame_interval_us
# # # #
# # # #         while len(history) < 8:
# # # #             on_m, _ = ts.get_hardware_matrices()
# # # #             history.append(on_m.flatten())
# # # #
# # # #         # Combine the 8 frames (Spatio-temporal shape of the spoken word)
# # # #         feat = np.concatenate(history[:8])
# # # #         # We also extract basic event density (how "loud" it was)
# # # #         total_spikes = len(spike_times)
# # # #         avg_hz = np.mean(spike_units) if total_spikes > 0 else 0.0
# # # #
# # # #         final_vector = np.hstack(
# # # #             [feat, np.mean(history[:8], axis=0), np.std(history[:8], axis=0), [total_spikes, avg_hz]])
# # # #         X_features.append(final_vector)
# # # #
# # # #     return np.array(X_features), total_awake, total_asleep
# # # #
# # # #
# # # # def run_shd_benchmark():
# # # #     print("==========================================================")
# # # #     print("  SHD AUDIO: NEUROMORPHIC SILICON COCHLEA BENCHMARK")
# # # #     print("==========================================================")
# # # #
# # # #     # 1. Load native Spikes
# # # #     samples_train, y_train = load_shd_data("shd_train.h5")
# # # #     samples_test, y_test = load_shd_data("shd_test.h5")
# # # #
# # # #     # 2. Map to 8x8 Time-Surfaces
# # # #     X_train, awake_tr, asleep_tr = process_shd_to_surfaces(samples_train, "Train")
# # # #     X_test, awake_te, asleep_te = process_shd_to_surfaces(samples_test, "Test")
# # # #
# # # #     total_cycles = (awake_tr + asleep_tr + awake_te + asleep_te)
# # # #     sleep_efficiency = ((asleep_tr + asleep_te) / total_cycles * 100) if total_cycles > 0 else 0
# # # #
# # # #     print("\n--- 3. Scaling and Training Models (20-Class Digits) ---")
# # # #     scaler = StandardScaler().fit(X_train)
# # # #     X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
# # # #
# # # #     t0 = time.time()
# # # #
# # # #     print(" -> Training Random Forest...")
# # # #     rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
# # # #     rf.fit(X_train, y_train)
# # # #     rf_acc = accuracy_score(y_test, rf.predict(X_test))
# # # #
# # # #     print(" -> Training Histogram Gradient Boosting (Edge AI)...")
# # # #     hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_depth=10, random_state=42)
# # # #     hgb.fit(X_train, y_train)
# # # #     hgb_acc = accuracy_score(y_test, hgb.predict(X_test))
# # # #
# # # #     print(" -> Training Deep MLP (Adam)...")
# # # #     mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128),
# # # #                         activation='relu',
# # # #                         solver='adam',
# # # #                         max_iter=400,
# # # #                         early_stopping=True,
# # # #                         random_state=42).fit(X_train, y_train)
# # # #     mlp_acc = accuracy_score(y_test, mlp.predict(X_test))
# # # #
# # # #     print("\n==========================================================")
# # # #     print("  SHD AUDIO FINAL ACCURACY REPORT (20-Class Digits)")
# # # #     print("==========================================================")
# # # #     print(f"Power Gating Efficiency   : {sleep_efficiency:.2f}% Sleep Time")
# # # #     print("----------------------------------------------------------")
# # # #     print(f"1. Random Forest Accuracy : {rf_acc * 100:.2f}%")
# # # #     print(f"2. Gradient Boost Accuracy: {hgb_acc * 100:.2f}%")
# # # #     print(f"3. Deep MLP Accuracy      : {mlp_acc * 100:.2f}%")
# # # #     print("----------------------------------------------------------")
# # # #     print("Classes: Spoken Digits (0-9) in English and German")
# # # #     print(f"Total Computation Time: {time.time() - t0:.2f} seconds")
# # # #
# # # #
# # # # if __name__ == "__main__":
# # # #     run_shd_benchmark()
# # #
# # # import os, time, sys
# # # import numpy as np
# # # from sklearn.neural_network import MLPClassifier
# # # from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
# # # from sklearn.preprocessing import StandardScaler, MinMaxScaler
# # # from sklearn.metrics import accuracy_score
# # # from src.time_surface import TimeSurfaceModel
# # # import src.config as config
# # # import warnings
# # #
# # # warnings.filterwarnings("ignore")
# # #
# # # try:
# # #     import h5py
# # # except ImportError:
# # #     print("[ERROR] The 'h5py' library is required to read SHD datasets. Please run: pip install h5py")
# # #     sys.exit(1)
# # #
# # # # --- OPTIMIZED AUDIO CONFIG ---
# # # config.TAU_US = 100_000.0
# # # GRID_X, GRID_Y = 8, 8
# # #
# # # DATA_DIR = r"D:\Projects\Personal\MIOT_time_surface\datasets\SHD"
# # #
# # #
# # # def load_shd_data(file_name):
# # #     file_path = os.path.join(DATA_DIR, file_name)
# # #     if not os.path.exists(file_path):
# # #         print(f"[ERROR] Could not find SHD dataset at: {file_path}")
# # #         sys.exit(1)
# # #
# # #     print(f" -> Loading native spikes from '{file_name}'...")
# # #
# # #     with h5py.File(file_path, 'r') as f:
# # #         labels = np.array(f['labels'], dtype=np.int32)
# # #         times_ds = f['spikes']['times']
# # #         units_ds = f['spikes']['units']
# # #
# # #         samples = []
# # #         for i in range(len(labels)):
# # #             t_arr = np.array(times_ds[i], dtype=np.float64)
# # #             u_arr = np.array(units_ds[i], dtype=np.int32)
# # #             valid_mask = ~np.isnan(t_arr)
# # #
# # #             samples.append({
# # #                 'times': t_arr[valid_mask],
# # #                 'units': u_arr[valid_mask]
# # #             })
# # #
# # #     return samples, labels
# # #
# # #
# # # def process_shd_to_surfaces(samples, dataset_name):
# # #     print(f" -> Routing {len(samples)} {dataset_name} audio spikes to the 8x8 Grid...")
# # #     X_features = []
# # #
# # #     total_awake = 0
# # #     total_asleep = 0
# # #
# # #     for i, sample in enumerate(samples):
# # #         ts = TimeSurfaceModel()
# # #
# # #         spike_times = sample['times']
# # #         spike_units = sample['units']
# # #
# # #         duration = float(np.max(spike_times)) if len(spike_times) > 0 else 1.0
# # #         frame_interval_us = (duration * 1_000_000.0) / 8.0
# # #         next_frame_time = frame_interval_us
# # #         history = []
# # #         last_cycle = 0
# # #
# # #         for t_sec, unit in zip(spike_times, spike_units):
# # #             t_us = float(t_sec) * 1_000_000.0
# # #
# # #             pixel_id = min(63, int(unit) // 11)
# # #             x = pixel_id % 8
# # #             y = pixel_id // 8
# # #
# # #             ts.process_event(x, y, t_us, polarity=1)
# # #
# # #             curr_cycle = int(t_us / config.CLOCK_PERIOD_US)
# # #             gap = curr_cycle - last_cycle
# # #             if gap > config.IDLE_TIMEOUT_CYCLES:
# # #                 total_awake += config.IDLE_TIMEOUT_CYCLES
# # #                 total_asleep += (gap - config.IDLE_TIMEOUT_CYCLES)
# # #             else:
# # #                 total_awake += gap
# # #             last_cycle = curr_cycle
# # #
# # #             if t_us >= next_frame_time:
# # #                 on_m, _ = ts.get_hardware_matrices()
# # #                 history.append(on_m.flatten())
# # #                 next_frame_time += frame_interval_us
# # #
# # #         while len(history) < 8:
# # #             on_m, _ = ts.get_hardware_matrices()
# # #             history.append(on_m.flatten())
# # #
# # #         feat = np.concatenate(history[:8])
# # #
# # #         # --- THE ACOUSTIC BRIDGE ---
# # #         # A 64-bin histogram of the raw spike frequencies. This gives the AI a perfect
# # #         # spectral footprint of the word to supplement the temporal Time-Surface visual.
# # #         freq_hist, _ = np.histogram(spike_units, bins=64, range=(0, 700))
# # #         freq_hist = freq_hist / (np.max(freq_hist) + 1e-6)  # Normalize the histogram
# # #
# # #         total_spikes = len(spike_times)
# # #         avg_hz = np.mean(spike_units) if total_spikes > 0 else 0.0
# # #
# # #         final_vector = np.hstack(
# # #             [feat, np.mean(history[:8], axis=0), np.std(history[:8], axis=0), freq_hist, [total_spikes, avg_hz]])
# # #         X_features.append(final_vector)
# # #
# # #     return np.array(X_features), total_awake, total_asleep
# # #
# # #
# # # def run_shd_benchmark():
# # #     print("==========================================================")
# # #     print("  SHD AUDIO: NEUROMORPHIC SILICON COCHLEA BENCHMARK")
# # #     print("==========================================================")
# # #
# # #     samples_train, y_train = load_shd_data("shd_train.h5")
# # #     samples_test, y_test = load_shd_data("shd_test.h5")
# # #
# # #     X_train, awake_tr, asleep_tr = process_shd_to_surfaces(samples_train, "Train")
# # #     X_test, awake_te, asleep_te = process_shd_to_surfaces(samples_test, "Test")
# # #
# # #     total_cycles = (awake_tr + asleep_tr + awake_te + asleep_te)
# # #     sleep_efficiency = ((asleep_tr + asleep_te) / total_cycles * 100) if total_cycles > 0 else 0
# # #
# # #     print("\n--- 3. Scaling and Training Models (20-Class Digits) ---")
# # #     scaler = MinMaxScaler().fit(X_train)
# # #     X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
# # #
# # #     t0 = time.time()
# # #
# # #     print(" -> Training Random Forest...")
# # #     rf = RandomForestClassifier(n_estimators=400, max_depth=20, random_state=42, n_jobs=-1)
# # #     rf.fit(X_train, y_train)
# # #     rf_acc = accuracy_score(y_test, rf.predict(X_test))
# # #
# # #     print(" -> Training 5-Fold Histogram Gradient Boosting Ensemble...")
# # #     hgb1 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, max_depth=12, random_state=42)
# # #     hgb2 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08, max_depth=15, l2_regularization=0.1,
# # #                                           random_state=100)
# # #     hgb3 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.12, max_depth=10, min_samples_leaf=10,
# # #                                           random_state=999)
# # #     hgb4 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.09, max_depth=14, max_bins=200,
# # #                                           random_state=777)
# # #     hgb5 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.11, max_depth=11, l2_regularization=0.2,
# # #                                           random_state=123)
# # #
# # #     hgb_ensemble = VotingClassifier(estimators=[
# # #         ('h1', hgb1), ('h2', hgb2), ('h3', hgb3), ('h4', hgb4), ('h5', hgb5)
# # #     ], voting='soft').fit(X_train, y_train)
# # #     hgb_acc = accuracy_score(y_test, hgb_ensemble.predict(X_test))
# # #
# # #     print(" -> Training 5-Fold Deep MLP Ensemble (State-of-the-Art)...")
# # #     mlp1 = MLPClassifier(hidden_layer_sizes=(1024, 512, 256), activation='relu', solver='adam', max_iter=500,
# # #                          early_stopping=True, random_state=42)
# # #     mlp2 = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam', max_iter=500,
# # #                          early_stopping=True, random_state=100)
# # #     mlp3 = MLPClassifier(hidden_layer_sizes=(1024, 1024, 256), activation='relu', solver='adam', max_iter=500,
# # #                          early_stopping=True, random_state=999)
# # #     mlp4 = MLPClassifier(hidden_layer_sizes=(512, 512, 256), activation='relu', solver='adam', max_iter=500,
# # #                          early_stopping=True, random_state=777)
# # #     mlp5 = MLPClassifier(hidden_layer_sizes=(1024, 256, 128), activation='relu', solver='adam', max_iter=500,
# # #                          early_stopping=True, random_state=123)
# # #
# # #     mlp_ensemble = VotingClassifier(estimators=[
# # #         ('m1', mlp1), ('m2', mlp2), ('m3', mlp3), ('m4', mlp4), ('m5', mlp5)
# # #     ], voting='soft').fit(X_train, y_train)
# # #     mlp_acc = accuracy_score(y_test, mlp_ensemble.predict(X_test))
# # #
# # #     print("\n==========================================================")
# # #     print("  SHD AUDIO FINAL ACCURACY REPORT (20-Class Digits)")
# # #     print("==========================================================")
# # #     print(f"Power Gating Efficiency   : {sleep_efficiency:.2f}% Sleep Time")
# # #     print("----------------------------------------------------------")
# # #     print(f"1. Random Forest Accuracy : {rf_acc * 100:.2f}%")
# # #     print(f"2. 5-Fold HGB Ensemble    : {hgb_acc * 100:.2f}%")
# # #     print(f"3. 5-Fold MLP Ensemble    : {mlp_acc * 100:.2f}%")
# # #     print("----------------------------------------------------------")
# # #     print("Classes: Spoken Digits (0-9) in English and German")
# # #     print(f"Total Computation Time: {time.time() - t0:.2f} seconds")
# # #
# # #
# # # if __name__ == "__main__":
# # #     run_shd_benchmark()
# #
# # import os, time, sys
# # import numpy as np
# # from sklearn.neural_network import MLPClassifier
# # from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
# # from sklearn.preprocessing import StandardScaler, MinMaxScaler
# # from sklearn.metrics import accuracy_score
# # from src.time_surface import TimeSurfaceModel
# # import src.config as config
# # import warnings
# #
# # warnings.filterwarnings("ignore")
# #
# # try:
# #     import h5py
# # except ImportError:
# #     print("[ERROR] The 'h5py' library is required to read SHD datasets. Please run: pip install h5py")
# #     sys.exit(1)
# #
# # # --- OPTIMIZED AUDIO CONFIG ---
# # config.TAU_US = 100_000.0
# # GRID_X, GRID_Y = 8, 8
# #
# # DATA_DIR = r"D:\Projects\Personal\MIOT_time_surface\datasets\SHD"
# #
# #
# # def load_shd_data(file_name):
# #     file_path = os.path.join(DATA_DIR, file_name)
# #     if not os.path.exists(file_path):
# #         print(f"[ERROR] Could not find SHD dataset at: {file_path}")
# #         sys.exit(1)
# #
# #     print(f" -> Loading native spikes from '{file_name}'...")
# #
# #     with h5py.File(file_path, 'r') as f:
# #         labels = np.array(f['labels'], dtype=np.int32)
# #         times_ds = f['spikes']['times']
# #         units_ds = f['spikes']['units']
# #
# #         samples = []
# #         for i in range(len(labels)):
# #             t_arr = np.array(times_ds[i], dtype=np.float64)
# #             u_arr = np.array(units_ds[i], dtype=np.int32)
# #             valid_mask = ~np.isnan(t_arr)
# #
# #             samples.append({
# #                 'times': t_arr[valid_mask],
# #                 'units': u_arr[valid_mask]
# #             })
# #
# #     return samples, labels
# #
# #
# # def process_shd_to_surfaces(samples, dataset_name):
# #     print(f" -> Routing {len(samples)} {dataset_name} audio spikes to the 8x8 Grid...")
# #     X_features = []
# #
# #     total_awake = 0
# #     total_asleep = 0
# #
# #     for i, sample in enumerate(samples):
# #         ts = TimeSurfaceModel()
# #
# #         spike_times = sample['times']
# #         spike_units = sample['units']
# #
# #         duration = float(np.max(spike_times)) if len(spike_times) > 0 else 1.0
# #         frame_duration = duration / 8.0
# #         frame_interval_us = frame_duration * 1_000_000.0
# #
# #         next_frame_time = frame_interval_us
# #         history = []
# #         last_cycle = 0
# #
# #         # --- HIGH-RES SPECTRO-TEMPORAL BRIDGE ---
# #         # 8 Frames x 70 Frequency Bins (10Hz per bin)
# #         spectrogram = np.zeros((8, 70))
# #
# #         for t_sec, unit in zip(spike_times, spike_units):
# #             t_us = float(t_sec) * 1_000_000.0
# #
# #             # 1. Build the High-Res Phoneme Spectrogram
# #             frame_idx = min(7, int(t_sec / frame_duration))
# #             bin_idx = min(69, int(unit) // 10)
# #             spectrogram[frame_idx, bin_idx] += 1.0
# #
# #             # 2. Route to 8x8 Hardware Simulator
# #             pixel_id = min(63, int(unit) // 11)
# #             x = pixel_id % 8
# #             y = pixel_id // 8
# #
# #             ts.process_event(x, y, t_us, polarity=1)
# #
# #             curr_cycle = int(t_us / config.CLOCK_PERIOD_US)
# #             gap = curr_cycle - last_cycle
# #             if gap > config.IDLE_TIMEOUT_CYCLES:
# #                 total_awake += config.IDLE_TIMEOUT_CYCLES
# #                 total_asleep += (gap - config.IDLE_TIMEOUT_CYCLES)
# #             else:
# #                 total_awake += gap
# #             last_cycle = curr_cycle
# #
# #             # Use 'while' to properly capture frames even if there is audio silence
# #             while t_us >= next_frame_time and len(history) < 8:
# #                 on_m, _ = ts.get_hardware_matrices()
# #                 history.append(on_m.flatten())
# #                 next_frame_time += frame_interval_us
# #
# #         while len(history) < 8:
# #             on_m, _ = ts.get_hardware_matrices()
# #             history.append(on_m.flatten())
# #
# #         # --- FEATURE COMPILATION ---
# #         # A) Spatial Neuromorphic Data (512 + 64 + 64 features)
# #         feat_ts = np.concatenate(history[:8])
# #         ts_mean = np.mean(history[:8], axis=0)
# #         ts_std = np.std(history[:8], axis=0)
# #
# #         # B) Acoustic Spectrogram Data
# #         # Normalize the spectrogram to prevent volume (loudness) bias
# #         spec_norm = spectrogram / (np.max(spectrogram) + 1e-6)
# #
# #         # The Delta maps the Formant Shifts (How frequencies change over time - crucial for speech!)
# #         spec_delta = np.diff(spec_norm, axis=0).flatten()  # 7 frames * 70 bins = 490 features
# #         spec_flat = spec_norm.flatten()  # 8 frames * 70 bins = 560 features
# #         spec_mean = np.mean(spec_norm, axis=0)  # 70 features
# #
# #         total_spikes = len(spike_times)
# #
# #         final_vector = np.hstack([
# #             feat_ts, ts_mean, ts_std,
# #             spec_flat, spec_delta, spec_mean,
# #             [total_spikes]
# #         ])
# #
# #         X_features.append(final_vector)
# #
# #     return np.array(X_features), total_awake, total_asleep
# #
# #
# # def run_shd_benchmark():
# #     print("==========================================================")
# #     print("  SHD AUDIO: NEUROMORPHIC SILICON COCHLEA BENCHMARK")
# #     print("==========================================================")
# #
# #     samples_train, y_train = load_shd_data("shd_train.h5")
# #     samples_test, y_test = load_shd_data("shd_test.h5")
# #
# #     X_train, awake_tr, asleep_tr = process_shd_to_surfaces(samples_train, "Train")
# #     X_test, awake_te, asleep_te = process_shd_to_surfaces(samples_test, "Test")
# #
# #     total_cycles = (awake_tr + asleep_tr + awake_te + asleep_te)
# #     sleep_efficiency = ((asleep_tr + asleep_te) / total_cycles * 100) if total_cycles > 0 else 0
# #
# #     print("\n--- 3. Scaling and Training High-Fidelity Single Models ---")
# #     scaler = MinMaxScaler().fit(X_train)
# #     X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
# #
# #     t0 = time.time()
# #
# #     print(" -> Training Random Forest...")
# #     rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
# #     rf.fit(X_train, y_train)
# #     rf_acc = accuracy_score(y_test, rf.predict(X_test))
# #
# #     print(" -> Training Fast Histogram Gradient Boosting (Single Model)...")
# #     hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_depth=12, random_state=42)
# #     hgb.fit(X_train, y_train)
# #     hgb_acc = accuracy_score(y_test, hgb.predict(X_test))
# #
# #     print(" -> Training Deep MLP (Adam, Single Architecture)...")
# #     mlp = MLPClassifier(hidden_layer_sizes=(1024, 512, 256),
# #                         activation='relu',
# #                         solver='adam',
# #                         max_iter=400,
# #                         early_stopping=True,
# #                         random_state=42).fit(X_train, y_train)
# #     mlp_acc = accuracy_score(y_test, mlp.predict(X_test))
# #
# #     print("\n==========================================================")
# #     print("  SHD AUDIO FINAL ACCURACY REPORT (20-Class Digits)")
# #     print("==========================================================")
# #     print(f"Power Gating Efficiency   : {sleep_efficiency:.2f}% Sleep Time")
# #     print("----------------------------------------------------------")
# #     print(f"1. Random Forest Accuracy : {rf_acc * 100:.2f}%")
# #     print(f"2. Single HGB Accuracy    : {hgb_acc * 100:.2f}%")
# #     print(f"3. Single MLP Accuracy    : {mlp_acc * 100:.2f}%")
# #     print("----------------------------------------------------------")
# #     print("Classes: Spoken Digits (0-9) in English and German")
# #     print(f"Total Computation Time: {time.time() - t0:.2f} seconds")
# #
# #
# # if __name__ == "__main__":
# #     run_shd_benchmark()

# import os, time, sys
# import numpy as np
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.metrics import accuracy_score
# from src.time_surface import TimeSurfaceModel
# import src.config as config
# import warnings
#
# warnings.filterwarnings("ignore")
#
# try:
#     import h5py
# except ImportError:
#     print("[ERROR] The 'h5py' library is required to read SHD datasets. Please run: pip install h5py")
#     sys.exit(1)
#
# # --- OPTIMIZED AUDIO CONFIG ---
# config.TAU_US = 100_000.0
# GRID_X, GRID_Y = 8, 8
#
# DATA_DIR = r"D:\Projects\Personal\MIOT_time_surface\datasets\SHD"
#
#
# def load_shd_data(file_name):
#     file_path = os.path.join(DATA_DIR, file_name)
#     if not os.path.exists(file_path):
#         print(f"[ERROR] Could not find SHD dataset at: {file_path}")
#         sys.exit(1)
#
#     print(f" -> Loading native spikes from '{file_name}'...")
#
#     with h5py.File(file_path, 'r') as f:
#         labels = np.array(f['labels'], dtype=np.int32)
#         times_ds = f['spikes']['times']
#         units_ds = f['spikes']['units']
#
#         samples = []
#         for i in range(len(labels)):
#             t_arr = np.array(times_ds[i], dtype=np.float64)
#             u_arr = np.array(units_ds[i], dtype=np.int32)
#             valid_mask = ~np.isnan(t_arr)
#
#             samples.append({
#                 'times': t_arr[valid_mask],
#                 'units': u_arr[valid_mask]
#             })
#
#     return samples, labels
#
#
# def process_shd_to_surfaces(samples, dataset_name):
#     print(f" -> Routing {len(samples)} {dataset_name} audio spikes to the 8x8 Grid...")
#     X_features = []
#
#     total_awake = 0
#     total_asleep = 0
#
#     for i, sample in enumerate(samples):
#         ts = TimeSurfaceModel()
#
#         spike_times = sample['times']
#         spike_units = sample['units']
#
#         duration = float(np.max(spike_times)) if len(spike_times) > 0 else 1.0
#         frame_duration = duration / 8.0
#         frame_interval_us = frame_duration * 1_000_000.0
#
#         next_frame_time = frame_interval_us
#         history = []
#         last_cycle = 0
#
#         # --- HIGH-RES SPECTRO-TEMPORAL BRIDGE ---
#         spectrogram = np.zeros((8, 70))
#
#         for t_sec, unit in zip(spike_times, spike_units):
#             t_us = float(t_sec) * 1_000_000.0
#
#             # 1. Build the High-Res Phoneme Spectrogram
#             frame_idx = min(7, int(t_sec / frame_duration))
#             bin_idx = min(69, int(unit) // 10)
#             spectrogram[frame_idx, bin_idx] += 1.0
#
#             # 2. Route to 8x8 Hardware Simulator
#             pixel_id = min(63, int(unit) // 11)
#             x = pixel_id % 8
#             y = pixel_id // 8
#
#             ts.process_event(x, y, t_us, polarity=1)
#
#             curr_cycle = int(t_us / config.CLOCK_PERIOD_US)
#             gap = curr_cycle - last_cycle
#             if gap > config.IDLE_TIMEOUT_CYCLES:
#                 total_awake += config.IDLE_TIMEOUT_CYCLES
#                 total_asleep += (gap - config.IDLE_TIMEOUT_CYCLES)
#             else:
#                 total_awake += gap
#             last_cycle = curr_cycle
#
#             while t_us >= next_frame_time and len(history) < 8:
#                 on_m, _ = ts.get_hardware_matrices()
#                 history.append(on_m.flatten())
#                 next_frame_time += frame_interval_us
#
#         while len(history) < 8:
#             on_m, _ = ts.get_hardware_matrices()
#             history.append(on_m.flatten())
#
#         # --- FEATURE COMPILATION ---
#         feat_ts = np.concatenate(history[:8])
#         ts_mean = np.mean(history[:8], axis=0)
#         ts_std = np.std(history[:8], axis=0)
#
#         # B) Acoustic Spectrogram Data
#         spec_norm = spectrogram / (np.max(spectrogram) + 1e-6)
#         spec_delta = np.diff(spec_norm, axis=0).flatten()
#         spec_flat = spec_norm.flatten()
#         spec_mean = np.mean(spec_norm, axis=0)
#         spec_var = np.var(spec_norm, axis=0)  # NEW: Formant modulation variance
#
#         total_spikes = len(spike_times)
#
#         # NEW: Temporal Spike Statistics (Where does the word peak?)
#         if total_spikes > 0:
#             t_mean = np.mean(spike_times)
#             t_std = np.std(spike_times)
#             t_median = np.median(spike_times)
#             t_q1 = np.percentile(spike_times, 25)
#             t_q3 = np.percentile(spike_times, 75)
#         else:
#             t_mean, t_std, t_median, t_q1, t_q3 = 0, 0, 0, 0, 0
#
#         time_stats = [t_mean, t_std, t_median, t_q1, t_q3]
#
#         final_vector = np.hstack([
#             feat_ts, ts_mean, ts_std,
#             spec_flat, spec_delta, spec_mean, spec_var,
#             [total_spikes], time_stats
#         ])
#
#         X_features.append(final_vector)
#
#     return np.array(X_features), total_awake, total_asleep
#
#
# def run_shd_benchmark():
#     print("==========================================================")
#     print("  SHD AUDIO: NEUROMORPHIC SILICON COCHLEA BENCHMARK")
#     print("==========================================================")
#
#     samples_train, y_train = load_shd_data("shd_train.h5")
#     samples_test, y_test = load_shd_data("shd_test.h5")
#
#     X_train, awake_tr, asleep_tr = process_shd_to_surfaces(samples_train, "Train")
#     X_test, awake_te, asleep_te = process_shd_to_surfaces(samples_test, "Test")
#
#     total_cycles = (awake_tr + asleep_tr + awake_te + asleep_te)
#     sleep_efficiency = ((asleep_tr + asleep_te) / total_cycles * 100) if total_cycles > 0 else 0
#
#     print("\n--- 3. Scaling and Training High-Fidelity Models ---")
#     scaler = MinMaxScaler().fit(X_train)
#     X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
#
#     t0 = time.time()
#
#     print(" -> Training Random Forest...")
#     rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
#     rf.fit(X_train, y_train)
#     rf_acc = accuracy_score(y_test, rf.predict(X_test))
#
#     print(" -> Training Fast 3-Fold HGB Ensemble (Push to 90s)...")
#     hgb1 = HistGradientBoostingClassifier(max_iter=350, learning_rate=0.08, max_depth=12, random_state=42)
#     hgb2 = HistGradientBoostingClassifier(max_iter=350, learning_rate=0.1, max_depth=15, l2_regularization=0.1,
#                                           random_state=100)
#     hgb3 = HistGradientBoostingClassifier(max_iter=350, learning_rate=0.09, max_depth=10, min_samples_leaf=10,
#                                           random_state=999)
#
#     hgb_ensemble = VotingClassifier(estimators=[('h1', hgb1), ('h2', hgb2), ('h3', hgb3)], voting='soft')
#     hgb_ensemble.fit(X_train, y_train)
#     hgb_acc = accuracy_score(y_test, hgb_ensemble.predict(X_test))
#
#     print(" -> Training Deep MLP (Adam, Single Architecture)...")
#     mlp = MLPClassifier(hidden_layer_sizes=(1024, 512, 256),
#                         activation='relu',
#                         solver='adam',
#                         max_iter=400,
#                         early_stopping=True,
#                         random_state=42).fit(X_train, y_train)
#     mlp_acc = accuracy_score(y_test, mlp.predict(X_test))
#
#     print("\n==========================================================")
#     print("  SHD AUDIO FINAL ACCURACY REPORT (20-Class Digits)")
#     print("==========================================================")
#     print(f"Power Gating Efficiency   : {sleep_efficiency:.2f}% Sleep Time")
#     print("----------------------------------------------------------")
#     print(f"1. Random Forest Accuracy : {rf_acc * 100:.2f}%")
#     print(f"2. 3-Fold HGB Ensemble    : {hgb_acc * 100:.2f}%")
#     print(f"3. Single MLP Accuracy    : {mlp_acc * 100:.2f}%")
#     print("----------------------------------------------------------")
#     print("Classes: Spoken Digits (0-9) in English and German")
#     print(f"Total Computation Time: {time.time() - t0:.2f} seconds")
#
#
# if __name__ == "__main__":
#     run_shd_benchmark()

import os, time, sys
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from src.time_surface import TimeSurfaceModel
import src.config as config
import warnings

warnings.filterwarnings("ignore")

try:
    import h5py
except ImportError:
    print("[ERROR] The 'h5py' library is required to read SHD datasets. Please run: pip install h5py")
    sys.exit(1)

# --- OPTIMIZED AUDIO CONFIG ---
config.TAU_US = 100_000.0
GRID_X, GRID_Y = 8, 8

DATA_DIR = r"D:\Projects\Personal\MIOT_time_surface\datasets\SHD"


def load_shd_data(file_name):
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"[ERROR] Could not find SHD dataset at: {file_path}")
        sys.exit(1)

    print(f" -> Loading native spikes from '{file_name}'...")

    with h5py.File(file_path, 'r') as f:
        labels = np.array(f['labels'], dtype=np.int32)
        times_ds = f['spikes']['times']
        units_ds = f['spikes']['units']

        samples = []
        for i in range(len(labels)):
            t_arr = np.array(times_ds[i], dtype=np.float64)
            u_arr = np.array(units_ds[i], dtype=np.int32)
            valid_mask = ~np.isnan(t_arr)

            samples.append({
                'times': t_arr[valid_mask],
                'units': u_arr[valid_mask]
            })

    return samples, labels


def process_shd_to_surfaces(samples, dataset_name):
    print(f" -> Routing {len(samples)} {dataset_name} audio spikes to the 8x8 Grid...")
    X_features = []

    total_awake = 0
    total_asleep = 0

    for i, sample in enumerate(samples):
        ts = TimeSurfaceModel()

        spike_times = sample['times']
        spike_units = sample['units']

        duration = float(np.max(spike_times)) if len(spike_times) > 0 else 1.0

        frame_interval_us = (duration * 1_000_000.0) / 8.0
        next_frame_time = frame_interval_us
        history = []
        last_cycle = 0

        for t_sec, unit in zip(spike_times, spike_units):
            t_us = float(t_sec) * 1_000_000.0

            pixel_id = min(63, int(unit) // 11)
            x = pixel_id % 8
            y = pixel_id // 8

            ts.process_event(x, y, t_us, polarity=1)

            curr_cycle = int(t_us / config.CLOCK_PERIOD_US)
            gap = curr_cycle - last_cycle
            if gap > config.IDLE_TIMEOUT_CYCLES:
                total_awake += config.IDLE_TIMEOUT_CYCLES
                total_asleep += (gap - config.IDLE_TIMEOUT_CYCLES)
            else:
                total_awake += gap
            last_cycle = curr_cycle

            while t_us >= next_frame_time and len(history) < 8:
                on_m, _ = ts.get_hardware_matrices()
                history.append(on_m.flatten())
                next_frame_time += frame_interval_us

        while len(history) < 8:
            on_m, _ = ts.get_hardware_matrices()
            history.append(on_m.flatten())

        feat_ts = np.concatenate(history[:8])
        ts_mean = np.mean(history[:8], axis=0)

        unit_counts = np.bincount(spike_units, minlength=700)[:700]
        unit_fingerprint = unit_counts / (np.max(unit_counts) + 1e-6)

        if len(spike_times) > 0:
            time_counts, _ = np.histogram(spike_times, bins=25)
            time_envelope = time_counts / (np.max(time_counts) + 1e-6)
        else:
            time_envelope = np.zeros(25)

        spectrogram = np.zeros((10, 70))
        frame_duration_10 = duration / 10.0
        for t_sec, unit in zip(spike_times, spike_units):
            f_idx = min(9, int(t_sec / frame_duration_10))
            b_idx = min(69, int(unit) // 10)
            spectrogram[f_idx, b_idx] += 1.0

        spec_flat = (spectrogram / (np.max(spectrogram) + 1e-6)).flatten()

        final_vector = np.hstack([
            feat_ts, ts_mean,
            unit_fingerprint,
            time_envelope,
            spec_flat,
            [len(spike_times)]
        ])

        X_features.append(final_vector)

    return np.array(X_features), total_awake, total_asleep


def run_shd_benchmark():
    print("==========================================================")
    print("  SHD AUDIO: NEUROMORPHIC SILICON COCHLEA (SOTA EDITION)")
    print("==========================================================")

    samples_train, y_train = load_shd_data("shd_train.h5")
    samples_test, y_test = load_shd_data("shd_test.h5")

    X_train, awake_tr, asleep_tr = process_shd_to_surfaces(samples_train, "Train")
    X_test, awake_te, asleep_te = process_shd_to_surfaces(samples_test, "Test")

    total_cycles = (awake_tr + asleep_tr + awake_te + asleep_te)
    sleep_efficiency = ((asleep_tr + asleep_te) / total_cycles * 100) if total_cycles > 0 else 0

    print("\n--- Scaling and Training High-Fidelity Models ---")
    scaler = MinMaxScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    t0 = time.time()

    print("\n -> Training Fast 3-Fold HGB Ensemble (With Early Stopping)...")
    hgb1 = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.08, max_depth=12, early_stopping=True,
                                          n_iter_no_change=15, random_state=42, verbose=2)
    hgb2 = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.1, max_depth=15, l2_regularization=0.01,
                                          early_stopping=True, n_iter_no_change=15, random_state=100, verbose=2)
    hgb3 = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.09, max_depth=10, min_samples_leaf=15,
                                          early_stopping=True, n_iter_no_change=15, random_state=999, verbose=2)

    hgb_ensemble = VotingClassifier(estimators=[('h1', hgb1), ('h2', hgb2), ('h3', hgb3)], voting='soft')
    hgb_ensemble.fit(X_train, y_train)
    hgb_acc = accuracy_score(y_test, hgb_ensemble.predict(X_test))

    print("\n -> Training Deep MLP (Adam, With Early Stopping & L2 Reg)...")
    mlp = MLPClassifier(hidden_layer_sizes=(1024, 512, 256),
                        activation='relu',
                        solver='adam',
                        alpha=0.001,
                        max_iter=500,
                        early_stopping=True,
                        n_iter_no_change=15,
                        verbose=True,
                        random_state=42).fit(X_train, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test))

    print("\n==========================================================")
    print("  SHD AUDIO FINAL ACCURACY REPORT (20-Class Digits)")
    print("==========================================================")
    print(f"Power Gating Efficiency   : {sleep_efficiency:.2f}% Sleep Time")
    print("----------------------------------------------------------")
    print(f"1. 3-Fold HGB Ensemble    : {hgb_acc * 100:.2f}%")
    print(f"2. Single Deep MLP        : {mlp_acc * 100:.2f}%")
    print("----------------------------------------------------------")
    print("Classes: Spoken Digits (0-9) in English and German")
    print(f"Total Computation Time: {time.time() - t0:.2f} seconds")

    # --- SAVE THE BEST PERFORMING MODEL ---
    print("\n--- Saving the Best Model ---")

    model_results = {
        "3-Fold HGB Ensemble": (hgb_acc, hgb_ensemble),
        "Single Deep MLP": (mlp_acc, mlp)
    }

    # Identify the model with the highest accuracy
    best_model_name = max(model_results, key=lambda k: model_results[k][0])
    best_acc, best_model = model_results[best_model_name]

    print(f"-> Winner: {best_model_name} ({best_acc * 100:.2f}%)")

    save_dir = r"D:\Projects\Personal\MIOT_time_surface\models"
    os.makedirs(save_dir, exist_ok=True)

    scaler_path = os.path.join(save_dir, "miot_shd_scaler.pkl")
    model_path = os.path.join(save_dir, "miot_shd_best_model.pkl")

    joblib.dump(scaler, scaler_path)
    joblib.dump(best_model, model_path)

    print(f"[SAVED] Scaler saved to: {scaler_path}")
    print(f"[SAVED] Model saved to : {model_path}")
    print("==========================================================")


if __name__ == "__main__":
    run_shd_benchmark()