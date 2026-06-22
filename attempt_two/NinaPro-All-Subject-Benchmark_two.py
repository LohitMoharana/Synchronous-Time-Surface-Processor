# src/NinaPro-All-Subject-Benchmark_two.py
import os, glob, time, scipy.io as sio, numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from attempt_two.time_surface_two import TimeSurfaceModel, TimeSurfaceModelRTL
import attempt_two.config_two as config

DB5_BASE_PATH = r"C:\Users\MAHESH MISHRA\Documents\MyProjects\SynchEMGFilter\datasets\datasets\Ninapro-DB5\Ninapro-DB5"


def delta_modulator(emg_matrix, threshold=config.EMG_DELTA_THRESHOLD):
    events, last_v = [], emg_matrix[0, :].copy()
    for t_idx in range(1, emg_matrix.shape[0]):
        current_v = emg_matrix[t_idx, :]
        diffs = current_v - last_v
        for ch in range(16):
            if abs(diffs[ch]) >= threshold:
                events.append({
                    't': float(t_idx * config.EMG_SAMPLE_PERIOD_US),
                    'x': ch % 8,
                    'y': 2 if ch < 8 else 5,
                    'p': 1 if diffs[ch] > 0 else 0
                })
                last_v[ch] = current_v[ch]
    return events


def extract_trial_features(trial_emg, use_rtl=False):
    """
    Extract 8-snapshot temporal feature vector from one gesture trial.
    Returns 640-element vector: 8x64 snapshots + 64 mean + 64 std.
    """
    events = delta_modulator(trial_emg)
    if len(events) < 10:
        return None

    ts = TimeSurfaceModelRTL() if use_rtl else TimeSurfaceModel()
    snapshots = []

    trial_duration_us = trial_emg.shape[0] * config.EMG_SAMPLE_PERIOD_US
    snapshot_interval = trial_duration_us / 8.0
    next_snapshot_t = snapshot_interval

    for ev in events:
        ts.process_event(ev['x'], ev['y'], ev['t'], ev['p'])
        if ev['t'] >= next_snapshot_t:
            on_m, _ = ts.get_hardware_matrices()
            snapshots.append(on_m.flatten().astype(float))
            next_snapshot_t += snapshot_interval

    # Ensure exactly 8 snapshots
    while len(snapshots) < 8:
        snapshots.append(
            snapshots[-1].copy() if snapshots else np.zeros(64))
    snapshots = snapshots[:8]

    return np.concatenate([
        np.concatenate(snapshots),          # 512 values
        np.mean(snapshots, axis=0),          # 64 values
        np.std(snapshots, axis=0)            # 64 values
    ])  # total: 640 features


def load_subject_trials(subject_folder, use_rtl=False):
    """Extract one feature vector per gesture trial."""
    mat_files = glob.glob(os.path.join(subject_folder, "*E1*.mat"))
    if not mat_files:
        return None, None

    trials_X, trials_y = [], []

    for mat_file in mat_files:
        mat = sio.loadmat(mat_file)
        emg_raw = mat['emg']
        stimulus = mat['stimulus'].flatten()

        in_gesture = False
        start_idx = 0
        current_label = 0

        for i in range(len(stimulus)):
            if not in_gesture and stimulus[i] != 0:
                in_gesture = True
                start_idx = i
                current_label = int(stimulus[i])
            elif in_gesture and stimulus[i] == 0:
                in_gesture = False
                trial_len = i - start_idx
                if trial_len < 50:
                    continue

                features = extract_trial_features(
                    emg_raw[start_idx:i, :], use_rtl=use_rtl)
                if features is not None:
                    trials_X.append(features)
                    trials_y.append(current_label)

    if not trials_X:
        return None, None
    return np.array(trials_X), np.array(trials_y)


def run_loo(X, y):
    """
    Leave-one-out cross validation.
    Returns accuracy and list of per-fold correct/wrong.
    """
    correct = 0
    n = len(X)

    for test_idx in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_idx:test_idx+1]
        y_test = y[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=False,  # remove this
            random_state=42
        ).fit(X_train_s, y_train)

        pred = mlp.predict(X_test_s)[0]
        if pred == y_test:
            correct += 1

    return correct / n * 100


def evaluate_subject(subject_folder):
    subject_id = os.path.basename(subject_folder).upper()
    print(f"\n[{subject_id}] Loading trials...")

    # Python golden model
    X_py, y_py = load_subject_trials(subject_folder, use_rtl=False)
    if X_py is None:
        print(f"[{subject_id}] No data found.")
        return None, None

    print(f"[{subject_id}] {len(X_py)} trials, "
          f"{len(np.unique(y_py))} classes")

    t0 = time.time()
    acc_py = run_loo(X_py, y_py)
    print(f"[{subject_id}] Python model:  {acc_py:.2f}% "
          f"({time.time()-t0:.0f}s)")

    # RTL equivalent model
    X_rtl, y_rtl = load_subject_trials(subject_folder, use_rtl=True)
    t0 = time.time()
    acc_rtl = run_loo(X_rtl, y_rtl)
    print(f"[{subject_id}] RTL model:     {acc_rtl:.2f}% "
          f"(loss: {acc_py-acc_rtl:.2f}%) "
          f"({time.time()-t0:.0f}s)")

    return acc_py, acc_rtl


def run_global_benchmark():
    print("=" * 60)
    print("  NINAPRO DB5: GLOBAL ALL-SUBJECT BENCHMARK")
    print("  Python vs RTL-Equivalent Time Surface")
    print("=" * 60)
    print(f"  TAU = {config.TAU_US/1000:.1f}ms | "
          f"RTL: factor={config.RTL_DECAY_FACTOR}, "
          f"period={config.RTL_DECAY_PERIOD_CYCLES} cycles")

    results_py = {}
    results_rtl = {}

    for s_num in range(1, 11):
        subject_dir = os.path.join(DB5_BASE_PATH, f"s{s_num}")
        if not os.path.exists(subject_dir):
            print(f"\n[WARNING] s{s_num} not found")
            continue

        acc_py, acc_rtl = evaluate_subject(subject_dir)
        if acc_py is not None:
            results_py[f"S{s_num}"] = acc_py
            results_rtl[f"S{s_num}"] = acc_rtl

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"{'Subject':<10} {'Python':>10} {'RTL':>10} {'Loss':>10}")
    print("-" * 42)

    losses = []
    for s in sorted(results_py.keys()):
        loss = results_py[s] - results_rtl[s]
        losses.append(loss)
        print(f"{s:<10} {results_py[s]:>9.2f}% "
              f"{results_rtl[s]:>9.2f}% "
              f"{loss:>+9.2f}%")

    print("-" * 42)
    mean_py = np.mean(list(results_py.values()))
    std_py = np.std(list(results_py.values()))
    mean_rtl = np.mean(list(results_rtl.values()))
    std_rtl = np.std(list(results_rtl.values()))
    mean_loss = np.mean(losses)

    print(f"{'MEAN':<10} {mean_py:>9.2f}% "
          f"{mean_rtl:>9.2f}% "
          f"{mean_loss:>+9.2f}%")
    print(f"{'STD':<10} {std_py:>9.2f}% "
          f"{std_rtl:>9.2f}%")
    print("=" * 60)
    print(f"\nQuantisation accuracy loss: {mean_loss:.2f}%")
    print("(Cost of fixed-point hardware vs continuous exponential)")


if __name__ == "__main__":
    run_global_benchmark()