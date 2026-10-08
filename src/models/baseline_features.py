import numpy as np
from scipy.fft import fft
from typing import Tuple


def extract_14_features(window: np.ndarray) -> np.ndarray:
    """
    Extracts the 14 low-dimensional physical & time-frequency features
    defined in Fang et al. (2016, Sensors):
    
    1. Mean of accelerometer magnitude
    2. Std dev of accelerometer magnitude
    3. Peak FFT magnitude of accelerometer
    4. Mean of gyroscope magnitude
    5. Mean Z-axis acceleration
    6. Horizontal section (X-Z plane) magnitude
    7. Mean X-axis acceleration
    8. Mean Y-axis acceleration
    9. Mean Z-axis acceleration (redundancy guard)
    10. Max accelerometer magnitude
    11. Mean instant acceleration change (jerk)
    12. Std dev instant acceleration change
    13. Mean magnetometer magnitude
    14. Mean instant magnetic change
    
    Args:
        window: np.ndarray of shape (TimeSteps=450, Channels=12)
    Returns:
        feature_vector: np.ndarray of shape (14,)
    """
    # Channel indexing:
    # 0..2: Accel (X, Y, Z)
    # 3..5: Linear Accel (X, Y, Z)
    # 6..8: Gyro (X, Y, Z)
    # 9..11: Mag (X, Y, Z)
    acc = window[:, 0:3]
    gyr = window[:, 6:9]
    mag = window[:, 9:12]

    # Magnitudes
    acc_mag = np.linalg.norm(acc, axis=1)
    gyr_mag = np.linalg.norm(gyr, axis=1)
    mag_mag = np.linalg.norm(mag, axis=1)

    # 1. Mean of accelerometer magnitude
    f1 = np.mean(acc_mag)
    # 2. Std dev of accelerometer magnitude
    f2 = np.std(acc_mag)

    # 3. Peak FFT magnitude of accelerometer
    fft_vals = np.abs(fft(acc_mag))
    half_len = len(fft_vals) // 2
    f3 = np.max(fft_vals[1:half_len]) if half_len > 1 else 0.0

    # 4. Mean of gyroscope magnitude
    f4 = np.mean(gyr_mag)

    # 5. Z-axis acceleration mean
    f5 = np.mean(acc[:, 2])

    # 6. Horizontal section (X-Z plane) magnitude
    f6 = np.mean(np.sqrt(acc[:, 0]**2 + acc[:, 2]**2))

    # 7, 8, 9. Directional acceleration means
    f7 = np.mean(acc[:, 0])
    f8 = np.mean(acc[:, 1])
    f9 = np.mean(acc[:, 2])

    # 10. Maximum of accelerometer magnitude
    f10 = np.max(acc_mag)

    # 11, 12. Instantaneous acceleration changes (Jerk = d_acc/dt)
    acc_jerk = np.diff(acc_mag)
    f11 = np.mean(np.abs(acc_jerk))
    f12 = np.std(acc_jerk)

    # 13, 14. Magnetometer mean & instant change
    f13 = np.mean(mag_mag)
    mag_diff = np.diff(mag_mag)
    f14 = np.mean(np.abs(mag_diff))

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14], dtype=np.float32)


def extract_features_dataset(x_data: np.ndarray) -> np.ndarray:
    """Extracts 14 features for a dataset of windows of shape (N, 450, 12)."""
    n_samples = len(x_data)
    features = np.zeros((n_samples, 14), dtype=np.float32)
    for i in range(n_samples):
        features[i] = extract_14_features(x_data[i])
    return features
