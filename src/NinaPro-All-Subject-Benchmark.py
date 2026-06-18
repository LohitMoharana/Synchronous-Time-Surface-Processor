import os, glob, time, scipy.io as sio, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from src.time_surface import TimeSurfaceModel
import src.config as config

# --- OPTIMIZED HARDWARE CONFIG ---
config.TAU_US = 75_000.0
GRID_X, GRID_Y = 8, 8

# --- BASE DATASET PATH ---
DB5_BASE_PATH = r"D:\Projects\Personal\MIOT_time_surface\datasets\Ninapro-DB5\Ninapro-DB5"


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


def evaluate_subject(subject_folder):
    mat_files = glob.glob(os.path.join(subject_folder, "*E1*.mat"))
    if not mat_files:
        return None

    X, y = [], []
    subject_id = os.path.basename(subject_folder).upper()
    print(f"\n[{subject_id}] Extracting Event-Driven Time-Surfaces...")

    # 1. Feature Extraction
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
    if len(X) == 0:
        return None

    # 2. Train/Test Split & Scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    # 3. Training the Deep MLP
    print(f"[{subject_id}] Training Deep MLP on {len(X_train)} hardware snapshots...")
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(1024, 1024, 512, 256),
                        activation='relu', solver='adam',
                        learning_rate='adaptive', max_iter=50,
                        early_stopping=True, random_state=42).fit(X_train, y_train)

    acc = accuracy_score(y_test, mlp.predict(X_test))
    train_time = time.time() - t0
    print(f"[{subject_id}] -> Accuracy: {acc * 100:.2f}% (Training Time: {train_time / 60:.1f} mins)")
    return acc * 100


def run_global_benchmark():
    print("==========================================================")
    print("  NINAPRO DB5: GLOBAL ALL-SUBJECT HARDWARE BENCHMARK")
    print("==========================================================")

    subject_accuracies = {}

    # Iterate through subject folders s1 to s10
    for s_num in range(1, 11):
        subject_dir = os.path.join(DB5_BASE_PATH, f"s{s_num}")
        if os.path.exists(subject_dir):
            acc = evaluate_subject(subject_dir)
            if acc is not None:
                subject_accuracies[f"S{s_num}"] = acc
        else:
            print(f"\n[WARNING] Folder for Subject {s_num} not found at {subject_dir}")

    print("\n==========================================================")
    print("                 FINAL GLOBAL REPORT")
    print("==========================================================")
    if not subject_accuracies:
        print("No subjects processed. Check dataset paths.")
        return

    for subj, acc in subject_accuracies.items():
        print(f" {subj} : {acc:.2f}%")

    global_mean = np.mean(list(subject_accuracies.values()))
    global_std = np.std(list(subject_accuracies.values()))

    print("----------------------------------------------------------")
    print(f" GLOBAL MEAN ACCURACY : {global_mean:.2f}%  (+/- {global_std:.2f}%)")
    print("==========================================================")


if __name__ == "__main__":
    run_global_benchmark()