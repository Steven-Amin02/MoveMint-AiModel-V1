import os
from pathlib import Path
import torch

# Base Directories
WORKSPACE_ROOT = Path(r"d:\Graduation-project-26\AI-2")
DATA_DIR = WORKSPACE_ROOT / "Data" / "28943915"
HTC_DIR = DATA_DIR / "HTC"
SHL_DIR = DATA_DIR / "SHL"

CHECKPOINT_DIR = WORKSPACE_ROOT / "checkpoints"
OUTPUT_DIR = WORKSPACE_ROOT / "outputs"

# Ensure directories exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Specs
HTC_X_PATH = HTC_DIR / "htc_x_small.npy"
HTC_Y_PATH = HTC_DIR / "htc_y_small.npy"

SHL_MOTION_PATH = SHL_DIR / "Motion.txt"
SHL_LABEL_PATH = SHL_DIR / "Label.txt"

# Time-Series Parameters
SAMPLING_RATE_HZ = 50.0  # 50 Hz target sampling rate (Android SENSOR_DELAY_GAME)
WINDOW_LENGTH = 450      # 450 time-steps = 9.0 seconds per window
NUM_CHANNELS = 12        # 3 Accel, 3 Linear Accel, 3 Gyro, 3 Mag

# Sensor Channel Index Groups
CHANNEL_NAMES = [
    "acc_x", "acc_y", "acc_z",
    "lacc_x", "lacc_y", "lacc_z",
    "gyr_x", "gyr_y", "gyr_z",
    "mag_x", "mag_y", "mag_z"
]

# Transportation Mode Classes (8 Classes)
LABEL_MAPPING = {
    0: "Still",
    1: "Walk",
    2: "Run",
    3: "Bike",
    4: "Car",
    5: "Bus",
    6: "Train",
    7: "Subway"
}
NUM_CLASSES = len(LABEL_MAPPING)

# MoveMint Domain Categories for Anti-Cheat & Gamification
STILL_CLASSES = [0]
MICRO_MOBILITY_CLASSES = [1, 2, 3]       # Walk, Run, Bike (Eco-friendly micro-mobility)
PRIVATE_VEHICLE_CLASSES = [4]           # Car (Anti-cheat target: Must NOT get transit points!)
PUBLIC_TRANSIT_CLASSES = [5, 6, 7]      # Bus, Train, Subway (Eco-friendly public transit)

# Training Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
