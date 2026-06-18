import os, glob, time, scipy.io as sio, numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from src.time_surface import TimeSurfaceModel
import src.config as config

# --- OPTIMIZED HARDWARE CONFIG ---
config.TAU_US = 75_000.0
GRID_X, GRID_Y = 8, 8

# --- TARGETING THE HARDEST SUBJECT (SUBJECT 8) ---
DB5_SUBJECT_PATH = r"D:\Projects\Personal\MIOT_time_surface\datasets\Ninapro-DB5\Ninapro-DB5\s8"


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


def run_worst_case_benchmark():
    mat_files =  glob.glob(os.path.join(DB5_SUBJECT_PATH, "*.mat"))

    if not mat_files:
        print(f"ERROR: Could not find any .mat files for Subject 8 at {DB5_SUBJECT_PATH}")
        return

    X, y = [], []

    print("\n==========================================================")
    print("  NINAPRO DB5: WORST-CASE SCENARIO TEST (SUBJECT 8)")
    print("==========================================================")
    print(" -> Extracting noisy data & generating Time-Surfaces...")

    for file in mat_files:
        mat_data = sio.loadmat(file)
        emg_raw, stimulus = mat_data['emg'], mat_data['stimulus'].flatten()

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
    print(f"[SUCCESS] Extracted {len(X)} snapshots from S8.")

    print("\n--- Splitting and Scaling Data ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    print("--- Training Deep MLP on 'Hard Mode' Data ---")
    t0 = time.time()

    mlp = MLPClassifier(hidden_layer_sizes=(1024, 1024, 512, 256),
                        activation='relu', solver='adam',
                        learning_rate='adaptive', max_iter=50,
                        verbose=True, early_stopping=True).fit(X_train, y_train)

    acc = accuracy_score(y_test, mlp.predict(X_test))

    print("\n==========================================================")
    print(f"-> WORST-CASE MODEL ACCURACY (S8) : {acc * 100:.2f}%")
    print("==========================================================")

    if acc > 0.85:
        print("[VERDICT] INCREDIBLY ROBUST.")
        print("Scoring >85% on Subject 8 proves the Time-Surface pipeline")
        print("can slice through extreme biological noise and fatigue.")
    else:
        print("[VERDICT] HARDWARE DEGRADATION EXPECTED.")
        print("The noise floor of S8 was too high for the current threshold.")

    # --- SAVE THE MODEL AND SCALER ---
    save_dir = r"D:\Projects\Personal\MIOT_time_surface\models"
    os.makedirs(save_dir, exist_ok=True)

    scaler_path = os.path.join(save_dir, "miot_s8_scaler_c.pkl")
    model_path = os.path.join(save_dir, "miot_s8_mlp_model_c.pkl")

    joblib.dump(scaler, scaler_path)
    joblib.dump(mlp, model_path)

    print(f"\n[SAVED] Scaler saved to: {scaler_path}")
    print(f"[SAVED] Model saved to : {model_path}")


if __name__ == "__main__":
    run_worst_case_benchmark()