"""Kalman filtering example for movement data."""

import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error

# Simulated position measurements (linear motion with noise)
np.random.seed(42)
time_steps = 200
true_positions = np.linspace(1, 100, time_steps)  # True smooth motion
measurements = true_positions + np.random.normal(
    0, 2, time_steps
)  # Adding noise

# Introduce artificial outliers (sudden jumps & drops)
measurements[30] += 50
measurements[60] -= 30

# --- Kalman Filter Setup ---
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([[0], [1]])  # Initial position & velocity
kf.F = np.array([[1, 1], [0, 1]])  # State transition matrix
kf.H = np.array([[1, 0]])  # Measurement function
kf.P *= 10  # Lower uncertainty for faster convergence
kf.R = 3  # **Reduced measurement noise**
kf.Q = np.array(
    [[0.05, 0], [0, 0.02]]
)  # **Allow slight flexibility in updates**


# --- Smart Outlier Handling ---
def reject_outliers(z, last_value, threshold=7):
    """Adaptive rejection: Allow slight movement but ignore extreme jumps."""
    diff = abs(z - last_value)
    if diff > threshold:
        return last_value + np.sign(z - last_value) * (
            threshold * 0.5
        )  # Reduce outlier impact
    return z


filtered_positions_kf = []
last_estimate = measurements[0]

for z in measurements:
    z = reject_outliers(z, last_estimate)  # Apply outlier rejection
    kf.predict()
    kf.update(z)

    last_estimate = kf.x[0, 0]  # Store last estimate
    filtered_positions_kf.append(last_estimate)

# --- Other Filters for Comparison ---
filtered_positions_savgol = savgol_filter(
    measurements, window_length=11, polyorder=2
)
filtered_positions_median = median_filter(measurements, size=5)

# --- Compute Error Metrics ---
mae_kf = mean_absolute_error(true_positions, filtered_positions_kf)
mae_savgol = mean_absolute_error(true_positions, filtered_positions_savgol)
mae_median = mean_absolute_error(true_positions, filtered_positions_median)

std_kf = np.std(filtered_positions_kf)
std_savgol = np.std(filtered_positions_savgol)
std_median = np.std(filtered_positions_median)

# --- Print Error Metrics ---
print(f"Kalman Filter - MAE: {mae_kf:.3f}, STD: {std_kf:.3f}")
print(f"Savitzky-Golay - MAE: {mae_savgol:.3f}, STD: {std_savgol:.3f}")
print(f"Median Filter - MAE: {mae_median:.3f}, STD: {std_median:.3f}")

# --- Plot Results ---
plt.figure(figsize=(12, 6))
plt.plot(
    true_positions[:100],
    label="True Positions (Ground Truth)",
    linestyle="solid",
    color="black",
    linewidth=2,
)
plt.plot(
    measurements[:100],
    label="Original Position (Noisy)",
    alpha=0.5,
    color="gray",
)
plt.plot(
    filtered_positions_kf[:100],
    label="Kalman Filter (Optimized)",
    linestyle="dashed",
    color="blue",
    linewidth=2,
)
plt.plot(
    filtered_positions_savgol[:100],
    label="Savitzky-Golay",
    linestyle="dotted",
    color="red",
)
plt.plot(
    filtered_positions_median[:100],
    label="Median Filter",
    linestyle="dashdot",
    color="green",
)

plt.xlabel("Time")
plt.ylabel("X Position")
plt.title("Final Optimization")
plt.legend()
plt.grid()
plt.show()
