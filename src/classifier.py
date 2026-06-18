# src/classifier.py
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Import our hardware simulator components
from src.config import GRID_X, GRID_Y
from src.synthetic_emg import generate_left_to_right_swipe, inject_sensor_noise
from src.time_surface import TimeSurfaceModel


def generate_dummy_dataset(num_samples=200):
    """
    Simulates a dataset of physical gestures.
    Label 0 = Swipe Right, Label 1 = Swipe Left
    Returns: X (flattened 64-bit matrices), y (labels)
    """
    print(f"[Dataset] Generating {num_samples} simulated hardware heatmaps...")
    X = []
    y = []

    for i in range(num_samples):
        # Alternate between Gesture 0 and Gesture 1
        label = i % 2

        # In a real scenario, you would load different CSV files here.
        # For now, we use our synthetic swipe and reverse it for the second gesture.
        events = generate_left_to_right_swipe()
        if label == 1:
            # Mirror the X coordinates to simulate a right-to-left swipe
            for e in events:
                e['x'] = (GRID_X - 1) - e['x']

                # Add random biological noise so the AI actually has to "learn"
        events = inject_sensor_noise(events, noise_ratio=0.20)

        # Run it through the hardware calculator
        ts_model = TimeSurfaceModel()
        for event in events:
            ts_model.process_event(event['x'], event['y'], event['t'], event['p'])

        on_matrix, _ = ts_model.get_hardware_matrices()

        # FLATTEN the 8x8 matrix into a 1D 64-element array for the AI
        X.append(on_matrix.flatten())
        y.append(label)

    return np.array(X), np.array(y)


def evaluate_classifiers():
    print("--- Starting Downstream AI Hardware Evaluation ---")

    # 1. Get the hardware features
    X, y = generate_dummy_dataset(num_samples=200)

    # Split into 80% Training and 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[Training Phase] Teaching models to read Time-Surfaces...")

    # ---------------------------------------------------------
    # MODEL 1: Support Vector Machine (Ultra-Low Power Target)
    # ---------------------------------------------------------
    start_time = time.time()
    svm_clf = SVC(kernel='linear')  # A simple linear line separates the data
    svm_clf.fit(X_train, y_train)
    svm_preds = svm_clf.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_preds) * 100
    svm_time = (time.time() - start_time) * 1000  # Convert to ms

    # ---------------------------------------------------------
    # MODEL 2: Multi-Layer Perceptron (High Accuracy Target)
    # ---------------------------------------------------------
    start_time = time.time()
    # A tiny neural net: Input(64) -> Hidden(32) -> Output(Classes)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)
    mlp_clf.fit(X_train, y_train)
    mlp_preds = mlp_clf.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_preds) * 100
    mlp_time = (time.time() - start_time) * 1000

    # --- PRINT THE PAPER RESULTS ---
    print("\n==========================================================")
    print("  DESIGN SPACE EXPLORATION: DOWNSTREAM AI CLASSIFIERS")
    print("==========================================================")
    print(f"Hardware Output Grid  : {GRID_X}x{GRID_Y} ({GRID_X * GRID_Y} input features)")
    print(f"Test Set Size         : {len(X_test)} Gestures")
    print("----------------------------------------------------------")
    print(f"1. Support Vector Machine (Linear)")
    print(f"   Accuracy        : {svm_acc:.2f}%")
    print(f"   Train+Test Time : {svm_time:.2f} ms")
    print(f"   Hardware Cost   : Extremely Low (Dot-Products only)")
    print("----------------------------------------------------------")
    print(f"2. Multi-Layer Perceptron (32 Hidden Neurons)")
    print(f"   Accuracy        : {mlp_acc:.2f}%")
    print(f"   Train+Test Time : {mlp_time:.2f} ms")
    print(f"   Hardware Cost   : Medium (MAC operations & Weight Memory)")
    print("==========================================================")


if __name__ == "__main__":
    evaluate_classifiers()