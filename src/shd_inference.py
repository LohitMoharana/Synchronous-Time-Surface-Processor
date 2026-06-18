import os
import time
import numpy as np
import joblib
import warnings
from src.time_surface import TimeSurfaceModel
import src.config as config

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
config.TAU_US = 100_000.0
MODEL_DIR = r"D:\Projects\Personal\MIOT_time_surface\models"
SCALER_PATH = os.path.join(MODEL_DIR, "miot_shd_scaler.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "miot_shd_best_model.pkl")

# SHD has 20 classes (0-9 English, 0-9 German)
# For the sake of the benchmark output, we'll map a few for demonstration
MOCK_LABELS = {0: "Zero (EN)", 1: "One (EN)", 2: "Two (EN)", 9: "Nine (EN)", 10: "Null (DE)", 11: "Eins (DE)",
               19: "Neun (DE)"}


def generate_mock_audio_spikes(num_spikes=500, duration=1.0):
    """
    Simulates a 1-second stream of an artificial cochlea hearing a spoken word.
    Generates realistic biological jitter across the 700 frequency channels.
    """
    times = np.sort(np.random.uniform(0.0, duration, num_spikes))
    # Focus spikes around a "formant" (e.g., channel 350) with some spread
    units = np.clip(np.random.normal(loc=350, scale=100, size=num_spikes), 0, 699).astype(np.int32)
    return times, units


def run_realtime_audio_benchmark():
    print("==========================================================")
    print("  SHD AUDIO: REAL-TIME EDGE INFERENCE ENGINE")
    print("==========================================================")

    try:
        print(" -> Waking up Silicon Cochlea (Loading Models)...")
        scaler = joblib.load(SCALER_PATH)
        loaded_model = joblib.load(MODEL_PATH)

        # --- THE SPEED FIX: EXTRACT A SINGLE LIGHTWEIGHT MODEL ---
        if hasattr(loaded_model, 'estimators_'):
            print(" -> [WARNING] Ensemble detected. Extracting a single HGB model for ultra-low latency...")
            model = loaded_model.estimators_[0]
        else:
            model = loaded_model

        # Warmup the CPU cache (Bypass Sklearn cold-start)
        print(" -> Warming up Silicon (Bypassing Cold-Start Penalty)...")
        dummy_features = np.zeros((1, 2002))  # 512 + 64 + 700 + 25 + 700 + 1 = 2002
        _ = model.predict(scaler.transform(dummy_features))

    except FileNotFoundError:
        print(f"[ERROR] Models not found in {MODEL_DIR}.")
        return

    print(" -> Awaiting Audio Buffer (1.00 seconds of Cochlear Spikes)...\n")

    iterations = 100
    lat_transducer_total = 0.0
    lat_ml_total = 0.0

    prediction = None

    print("----------------------------------------------------------")
    print(f"  [SYSTEM WAKE] EXECUTING INFERENCE PIPELINE ({iterations} Iterations)")
    print("----------------------------------------------------------")

    for _ in range(iterations):
        # 1. Simulate incoming biological audio spikes
        spike_times, spike_units = generate_mock_audio_spikes()
        duration = 1.0

        # =====================================================================
        # TIMING BLOCK START
        # =====================================================================
        t_transducer_start = time.perf_counter()

        # --- A. NEUROMORPHIC TRANSDUCER & FEATURE EXTRACTOR ---
        ts = TimeSurfaceModel()
        frame_interval_us = (duration * 1_000_000.0) / 8.0
        next_frame_time = frame_interval_us
        history = []

        # 1. Spatial Time-Surface Logic
        for t_sec, unit in zip(spike_times, spike_units):
            t_us = t_sec * 1_000_000.0
            pixel_id = min(63, unit // 11)
            x, y = pixel_id % 8, pixel_id // 8

            ts.process_event(x, y, t_us, polarity=1)

            while t_us >= next_frame_time and len(history) < 8:
                on_m, _ = ts.get_hardware_matrices()
                history.append(on_m.flatten())
                next_frame_time += frame_interval_us

        while len(history) < 8:
            on_m, _ = ts.get_hardware_matrices()
            history.append(on_m.flatten())

        feat_ts = np.concatenate(history[:8])
        ts_mean = np.mean(history[:8], axis=0)

        # 2. Biological Fingerprint (700 channels)
        unit_counts = np.bincount(spike_units, minlength=700)[:700]
        unit_fingerprint = unit_counts / (np.max(unit_counts) + 1e-6)

        # 3. Amplitude Envelope
        time_counts, _ = np.histogram(spike_times, bins=25, range=(0.0, duration))
        time_envelope = time_counts / (np.max(time_counts) + 1e-6)

        # 4. Spectro-Temporal Hash
        spectrogram = np.zeros((10, 70))
        frame_duration_10 = duration / 10.0
        for t_sec, unit in zip(spike_times, spike_units):
            f_idx = min(9, int(t_sec / frame_duration_10))
            b_idx = min(69, unit // 10)
            spectrogram[f_idx, b_idx] += 1.0

        spec_flat = (spectrogram / (np.max(spectrogram) + 1e-6)).flatten()

        # Combine Features (Must total exactly 2002 features for the Scaler)
        final_vector = np.hstack([
            feat_ts, ts_mean,
            unit_fingerprint,
            time_envelope,
            spec_flat,
            [len(spike_times)]
        ]).reshape(1, -1)

        t_transducer_end = time.perf_counter()

        # --- B. MACHINE LEARNING INFERENCE ---
        t_ml_start = time.perf_counter()

        scaled_vector = scaler.transform(final_vector)
        prediction = model.predict(scaled_vector)[0]

        t_ml_end = time.perf_counter()

        # --- ACCUMULATE ---
        lat_transducer_total += (t_transducer_end - t_transducer_start) * 1000
        lat_ml_total += (t_ml_end - t_ml_start) * 1000

    # Calculate Averages
    avg_transducer = lat_transducer_total / iterations
    avg_ml = lat_ml_total / iterations
    avg_total = avg_transducer + avg_ml

    real_time_factor = 1000.0 / avg_total  # 1000ms audio duration / latency

    pred_label = MOCK_LABELS.get(prediction, f"Class ID: {prediction}")

    print(f" -> Mock Prediction Output : [{pred_label}]")
    print("\n==========================================================")
    print(f"  LATENCY REPORT (Averaged over {iterations} runs)")
    print("==========================================================")
    print(f" 1. Neuromorphic Transducer & Features : {avg_transducer:.3f} ms")
    print(f" 2. Lightweight HGB Inference          : {avg_ml:.3f} ms")
    print("----------------------------------------------------------")
    print(f" TOTAL INFERENCE LATENCY               : {avg_total:.3f} ms")
    print("----------------------------------------------------------")
    print(f" Audio Buffer Size                     : 1000.00 ms (1.0s)")
    print(f" Real-Time Processing Speed            : {real_time_factor:.1f}x Faster Than Real-Time")
    print("==========================================================")


if __name__ == "__main__":
    run_realtime_audio_benchmark()