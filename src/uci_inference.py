import os
import time
import numpy as np
import joblib
import warnings
from src.time_surface import TimeSurfaceModel
import src.config as config

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- CONFIGURATION ---
config.TAU_US = 750_000.0
MODEL_DIR = r"D:\Projects\Personal\MIOT_time_surface\models"
SCALER_PATH = os.path.join(MODEL_DIR, "miot_uci_har_scaler.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "miot_uci_har_best_model.pkl")

# Human-readable labels
ACTIVITY_MAP = {
    0: "Walking",
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Sitting",
    4: "Standing",
    5: "Laying Down"
}


def get_scaled_val(raw_val, ch):
    if ch < 3:
        return np.clip(raw_val / 1.5, -1.0, 1.0)
    elif ch < 6:
        return np.clip(raw_val / 0.5, -1.0, 1.0)
    else:
        return np.clip(raw_val / 2.0, -1.0, 1.0)


def generate_mock_streaming_window():
    """Simulates a 2.56s (128 timestep) stream of physical walking data from an IMU."""
    t = np.linspace(0, 2.56, 128)
    window = np.zeros((128, 8))
    # Dummy physics for walking (1.8Hz cadence)
    window[:, 0] = np.sin(t * 2 * np.pi * 1.8) + 0.9  # Total Acc X
    window[:, 1] = np.cos(t * 2 * np.pi * 1.8)  # Total Acc Y
    window[:, 2] = np.sin(t * 2 * np.pi * 1.8 * 2)  # Total Acc Z
    window[:, 3:6] = np.random.normal(0, 0.2, (128, 3))  # Body Accel (jitter)
    window[:, 6:8] = np.random.normal(0, 0.5, (128, 2))  # Gyro (rotation)
    return window


def run_realtime_inference_benchmark():
    print("==========================================================")
    print("  UCI HAR KINEMATICS: REAL-TIME EDGE INFERENCE ENGINE")
    print("==========================================================")

    # 1. LOAD ARTIFACTS
    try:
        print(" -> Waking up Edge AI (Loading Scaler & Model)...")
        scaler = joblib.load(SCALER_PATH)
        loaded_model = joblib.load(MODEL_PATH)

        # --- THE SPEED FIX: EXTRACT A SINGLE LIGHTWEIGHT MODEL ---
        # The 5-Fold Ensemble is too heavy for microsecond edge inference.
        # We crack open the VotingClassifier and extract the first underlying model.
        if hasattr(loaded_model, 'estimators_'):
            print(" -> [WARNING] Ensemble detected. Extracting a single HGB model for ultra-low latency...")
            model = loaded_model.estimators_[0]
        else:
            model = loaded_model

        # --- THE FIX: COLD-START WARMUP ---
        # Sklearn has a massive memory-allocation penalty on the very first predict() call.
        # We run a dummy prediction to load the trees into active RAM.
        print(" -> Warming up Silicon (Bypassing Cold-Start Penalty)...")
        dummy_features = np.zeros((1, 772))  # 640 TS + 132 Stats = 772
        _ = model.predict(scaler.transform(dummy_features))

    except FileNotFoundError:
        print(f"[ERROR] Could not find models in {MODEL_DIR}. Did you run the training script?")
        return

    # 2. GENERATE INCOMING DATA
    print(" -> Awaiting IMU Buffer (2.56 seconds of physics data)...\n")
    window = generate_mock_streaming_window()

    print("----------------------------------------------------------")
    print("  [SYSTEM WAKE] EXECUTING INFERENCE PIPELINE (100 Iterations)")
    print("----------------------------------------------------------")

    iterations = 100
    lat_transducer_total = 0.0
    lat_stat_total = 0.0
    lat_ml_total = 0.0

    prediction = None

    for _ in range(iterations):
        # =====================================================================
        # TIMING BLOCK START
        # =====================================================================

        # --- A. NEUROMORPHIC TRANSDUCER (Spikes & Time-Surface) ---
        t_transducer_start = time.perf_counter()

        events = []
        sample_period_us = 20_000.0

        for ch in range(8):
            val = get_scaled_val(window[0, ch], ch)
            y = min(7, max(0, int((val + 1.0) * 3.99)))
            events.append({'t': 0.0, 'x': ch, 'y': y, 'p': 1})

        last_v = window[0, :]
        for t_idx in range(1, window.shape[0]):
            current_v = window[t_idx, :]
            diffs = current_v - last_v
            for ch in range(8):
                if abs(diffs[ch]) >= 0.015:
                    val = get_scaled_val(current_v[ch], ch)
                    y = min(7, max(0, int((val + 1.0) * 3.99)))
                    events.append(
                        {'t': float(t_idx * sample_period_us), 'x': ch, 'y': y, 'p': 1 if diffs[ch] > 0 else 0})
                    last_v[ch] = current_v[ch]

        ts = TimeSurfaceModel()
        frame_interval_us = 2_560_000.0 / 8.0
        history, next_frame_time = [], frame_interval_us

        for ev in events:
            ts.process_event(ev['x'], ev['y'], ev['t'], ev['p'])
            if ev['t'] >= next_frame_time:
                on_m, _ = ts.get_hardware_matrices()
                history.append(on_m.flatten())
                next_frame_time += frame_interval_us

        while len(history) < 8:
            on_m, _ = ts.get_hardware_matrices()
            history.append(on_m.flatten())

        ts_feat = np.concatenate(history[:8])
        ts_final = np.hstack([ts_feat, np.mean(history[:8], axis=0), np.std(history[:8], axis=0)])

        t_transducer_end = time.perf_counter()

        # --- B. STATISTICAL & KINEMATIC BRIDGE ---
        t_stat_start = time.perf_counter()

        w_mean, w_std, w_max, w_min = np.mean(window, axis=0), np.std(window, axis=0), np.max(window, axis=0), np.min(
            window, axis=0)
        w_energy = np.sum(window ** 2, axis=0) / len(window)

        w_jerk = np.diff(window, axis=0)
        w_jerk_mean, w_jerk_std = np.mean(w_jerk, axis=0), np.std(w_jerk, axis=0)

        w_fft = np.abs(np.fft.rfft(window, axis=0))
        f_mean, f_max, f_std = np.mean(w_fft, axis=0), np.max(w_fft, axis=0), np.std(w_fft, axis=0)
        f_energy = np.sum(w_fft ** 2, axis=0) / len(w_fft)
        f_dom = np.argmax(w_fft, axis=0)

        corr_matrix = np.nan_to_num(np.corrcoef(window.T))
        w_corr = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

        mag_total = np.sqrt(window[:, 0] ** 2 + window[:, 1] ** 2 + window[:, 2] ** 2)
        mag_body = np.sqrt(window[:, 3] ** 2 + window[:, 4] ** 2 + window[:, 5] ** 2)
        mag_features = [np.mean(mag_total), np.std(mag_total), np.max(mag_total), np.min(mag_total),
                        np.mean(mag_body), np.std(mag_body), np.max(mag_body), np.min(mag_body)]

        stat_bridge = np.concatenate([
            w_mean, w_std, w_max, w_min, w_energy, w_jerk_mean, w_jerk_std,
            f_mean, f_max, f_std, f_energy, f_dom, w_corr, mag_features
        ])

        final_feature_vector = np.hstack([ts_final, stat_bridge]).reshape(1, -1)

        t_stat_end = time.perf_counter()

        # --- C. MACHINE LEARNING INFERENCE ---
        t_ml_start = time.perf_counter()

        scaled_vector = scaler.transform(final_feature_vector)
        prediction = model.predict(scaled_vector)[0]

        t_ml_end = time.perf_counter()

        # --- ACCUMULATE ---
        lat_transducer_total += (t_transducer_end - t_transducer_start) * 1000
        lat_stat_total += (t_stat_end - t_stat_start) * 1000
        lat_ml_total += (t_ml_end - t_ml_start) * 1000

    # Calculate Average Latencies
    avg_transducer = lat_transducer_total / iterations
    avg_stat = lat_stat_total / iterations
    avg_ml = lat_ml_total / iterations
    avg_total = avg_transducer + avg_stat + avg_ml

    # Calculate Real-Time Factor
    real_time_factor = 2560.0 / avg_total

    print(f" -> Predicted Activity : [{ACTIVITY_MAP.get(prediction, 'Unknown')}]")
    print("\n==========================================================")
    print(f"  LATENCY REPORT (Averaged over {iterations} runs)")
    print("==========================================================")
    print(f" 1. Neuromorphic Transducer : {avg_transducer:.3f} ms")
    print(f" 2. Kinematic Feature Bridge: {avg_stat:.3f} ms")
    print(f" 3. Lightweight AI Inference: {avg_ml:.3f} ms")
    print("----------------------------------------------------------")
    print(f" TOTAL LATENCY              : {avg_total:.3f} ms")
    print("----------------------------------------------------------")
    print(f" Buffer Window Size         : 2560.00 ms (2.56s)")
    print(f" Real-Time Processing Speed : {real_time_factor:.1f}x Faster Than Real-Time")
    print("==========================================================")


if __name__ == "__main__":
    run_realtime_inference_benchmark()