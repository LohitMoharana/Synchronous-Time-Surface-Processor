import os
import glob
import csv
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.aedat_parser import load_aedat_file
from src.time_surface import TimeSurfaceModel
from src.config import GRID_X, GRID_Y, CLOCK_PERIOD_US, IDLE_TIMEOUT_CYCLES

# --- SET THIS TO YOUR EXACT WINDOWS PATH ---
DATASET_PATH = r"D:\Projects\Personal\MIOT_time_surface\datasets\DVSGesture128\DvsGesture\DvsGesture"


def process_dvs_directory(max_files=None):
    """ Runs the hardware simulator (CPU) to extract 32x32 time-surfaces. """
    X_features = []
    y_labels = []
    total_awake = 0
    total_asleep = 0

    label_files = glob.glob(os.path.join(DATASET_PATH, "*_labels.csv"))
    if not label_files:
        print(f"ERROR: Could not find any _labels.csv files in {DATASET_PATH}")
        return None, None, None

    if max_files is not None:
        label_files = label_files[:max_files]
        print(f"--- [Notice] Limiting execution to the first {max_files} trial files ---")

    print(f"--- Running 32x32 Silicon Simulator on {len(label_files)} Trial Recordings ---")

    for label_file in label_files:
        aedat_file = label_file.replace('_labels.csv', '.aedat')
        if not os.path.exists(aedat_file): continue

        all_events = load_aedat_file(aedat_file)

        with open(label_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                try:
                    class_id = int(row[0])
                    start_time = float(row[1])
                    end_time = float(row[2])
                except:
                    continue

                gesture_events = [e for e in all_events if start_time <= e['t'] <= end_time]
                gesture_events.sort(key=lambda x: x['t'])

                if len(gesture_events) < 100: continue

                ts_model = TimeSurfaceModel()
                last_cycle = int(start_time / CLOCK_PERIOD_US)

                scale_x = 128 / GRID_X
                scale_y = 128 / GRID_Y

                duration = end_time - start_time
                t_33 = start_time + (duration * 0.33)
                t_66 = start_time + (duration * 0.66)

                frame1_captured, frame2_captured = False, False
                gesture_feature_vector = []

                for ev in gesture_events:
                    hw_x = min(int(ev['x'] / scale_x), GRID_X - 1)
                    hw_y = min(int(ev['y'] / scale_y), GRID_Y - 1)

                    ts_model.process_event(hw_x, hw_y, ev['t'], ev['p'])

                    if ev['t'] >= t_33 and not frame1_captured:
                        on_m, off_m = ts_model.get_hardware_matrices()
                        gesture_feature_vector.extend(on_m.flatten())
                        gesture_feature_vector.extend(off_m.flatten())
                        frame1_captured = True

                    if ev['t'] >= t_66 and not frame2_captured:
                        on_m, off_m = ts_model.get_hardware_matrices()
                        gesture_feature_vector.extend(on_m.flatten())
                        gesture_feature_vector.extend(off_m.flatten())
                        frame2_captured = True

                    curr_cycle = int(ev['t'] / CLOCK_PERIOD_US)
                    gap = curr_cycle - last_cycle
                    if gap > IDLE_TIMEOUT_CYCLES:
                        total_awake += IDLE_TIMEOUT_CYCLES
                        total_asleep += (gap - IDLE_TIMEOUT_CYCLES)
                    else:
                        total_awake += gap
                    last_cycle = curr_cycle

                on_m, off_m = ts_model.get_hardware_matrices()
                gesture_feature_vector.extend(on_m.flatten())
                gesture_feature_vector.extend(off_m.flatten())

                expected_feature_length = GRID_X * GRID_Y * 2 * 3
                if len(gesture_feature_vector) == expected_feature_length:
                    X_features.append(gesture_feature_vector)
                    y_labels.append(class_id)

    total_cycles = total_awake + total_asleep
    savings = (total_asleep / total_cycles * 100) if total_cycles > 0 else 0
    return np.array(X_features), np.array(y_labels), savings


# --- TRUE CONVOLUTIONAL NEURAL NETWORK ---
class HardwareCNN(nn.Module):
    def __init__(self, num_classes):
        super(HardwareCNN, self).__init__()
        # 6 Channels total: 3 temporal frames * 2 polarities (ON/OFF)
        # Using Conv2d to actually scan the 32x32 matrix as a 2D physical image
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Image shrinks to 16x16

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Image shrinks to 8x8

            nn.Dropout2d(0.3)
        )

        # 32 channels * 8 width * 8 height = 2048 parameters
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten ONLY after spatial features are found
        x = self.fc_layers(x)
        return x


def evaluate_gpu(X, y, savings):
    if X is None or len(X) == 0: return

    print("\n==========================================================")
    print("  DVS128 DATASET: CNN (COMPUTER VISION) PERFORMANCE REPORT")
    print("==========================================================")
    print(f"Gestures Processed : {len(X)}")
    print(f"Matrix Grid Size   : {GRID_X}x{GRID_Y}")
    print(f"Power Gating Est.  : {savings:.2f}% Sleep Time")
    print("----------------------------------------------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Processing Unit Detected: {device.type.upper()}")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    print(f"-> Detected {num_classes} unique gesture classes.")

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # --- CRITICAL FIX: Fold the flat arrays back into 3D (Channels, Height, Width) boxes ---
    X_train_cnn = X_train.reshape(-1, 6, GRID_X, GRID_Y)
    X_test_cnn = X_test.reshape(-1, 6, GRID_X, GRID_Y)

    X_train_t = torch.FloatTensor(X_train_cnn).to(device) / 255.0
    X_test_t = torch.FloatTensor(X_test_cnn).to(device) / 255.0
    y_train_t = torch.LongTensor(y_train).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)

    train_data = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Initialize the true CNN (no input_size needed, CNNs infer it from the data shape)
    model = HardwareCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"-> Training Convolutional Neural Network on GPU...")
    epochs = 150
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"   Epoch [{epoch + 1}/{epochs}] | Loss: {total_loss / len(train_loader):.4f}")

    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, predicted = torch.max(test_outputs.data, 1)
        correct = (predicted == y_test_t).sum().item()
        accuracy = (correct / len(y_test_t)) * 100

    print("----------------------------------------------------------")
    print(f"-> FINAL CNN ACCURACY : {accuracy:.2f}%")
    print(f"-> GPU Training Time  : {train_time:.2f} seconds")
    print("==========================================================")


if __name__ == "__main__":
    X, y, savings = process_dvs_directory(max_files=None)
    evaluate_gpu(X, y, savings)