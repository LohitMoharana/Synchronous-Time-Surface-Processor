import os
import time
import numpy as np
import joblib

# --- PATHS TO YOUR SAVED MODEL ---
SAVE_DIR = r"D:\Projects\Personal\MIOT_time_surface\models"
SCALER_PATH = os.path.join(SAVE_DIR, "miot_s8_scaler_c.pkl")
MODEL_PATH = os.path.join(SAVE_DIR, "miot_s8_mlp_model_c.pkl")


def run_latency_test():
    print("--- MIOT Time-Surface: Latency & Inference Test ---")

    # 1. Load the pre-trained pipeline
    print("Loading Model and Scaler...")
    try:
        scaler = joblib.load(SCALER_PATH)
        mlp = joblib.load(MODEL_PATH)
        print("[SUCCESS] Pipeline loaded successfully.")
    except FileNotFoundError:
        print(f"[ERROR] Could not find models in {SAVE_DIR}.")
        print("Please ensure step1_benchmark.py finished saving the .pkl files.")
        return

    # 2. Generate a simulated Hardware Output Vector
    # Size: 8 frames of 8x8 (512) + mean (64) + std (64) = 640 features
    dummy_hardware_output = np.random.rand(1, 640) * 255.0

    print("\n--- Running Microsecond Speed Test ---")

    # WARM-UP: Python and CPU caches sometimes take a few milliseconds
    # to wake up on the very first execution. We run it once blindly.
    _ = mlp.predict(scaler.transform(dummy_hardware_output))

    # 3. Measure True Latency (Averaged over 1,000 continuous streams)
    iterations = 1000
    total_time_ms = 0.0

    for _ in range(iterations):
        start_t = time.perf_counter()

        # Step A: Scale the incoming hardware values
        scaled_input = scaler.transform(dummy_hardware_output)

        # Step B: Neural Network Inference (Gesture Prediction)
        prediction = mlp.predict(scaled_input)

        end_t = time.perf_counter()
        total_time_ms += (end_t - start_t) * 1000.0

    avg_latency = total_time_ms / iterations

    print("==========================================================")
    print(f"  PRODUCTION LATENCY REPORT (Averaged over {iterations} runs)")
    print("==========================================================")
    print(f"Feature Vector Size : 640 (Neuromorphic Compressed Data)")
    print(f"Model Architecture  : 1024x1024x512x256 Deep MLP")
    print(f"Average Latency     : {avg_latency:.4f} milliseconds per gesture")
    print("----------------------------------------------------------")

    if avg_latency < 10.0:
        print("[VERDICT]: REAL-TIME READY.")
        print("The system operates well under the 10ms human-perception")
        print("threshold required for fluid prosthetic responsiveness.")
    elif avg_latency < 50.0:
        print("[VERDICT]: ACCEPTABLE.")
        print("The system is fast enough for general IOT control, but may")
        print("feel slightly sluggish for high-speed prosthetics.")
    else:
        print("[VERDICT]: LAGGING.")
        print("The deep MLP is too heavy for the CPU. We need to prune layers.")
    print("==========================================================")


if __name__ == "__main__":
    run_latency_test()