"""Smooth pose tracks using Kalman filter
===========================================

Smooth pose tracks using a Kalman filter to estimate position, velocity,
and acceleration while accounting for measurement noise and process uncertainty.
"""

# %%
# Imports
# -------

import matplotlib.pyplot as plt
from scipy.signal import welch

from movement import sample_data
from movement.filtering import interpolate_over_time, kalman_filter

# %%
# Load a sample dataset
# ---------------------
# Let's load a sample dataset and print it to inspect its contents.

ds_wasp = sample_data.fetch_dataset("DLC_single-wasp.predictions.h5")
print(ds_wasp)

# %%
# We see that the dataset contains the 2D pose tracks and confidence scores
# for a single wasp, generated with DeepLabCut. The wasp is tracked at two
# keypoints: "head" and "stinger" in a video that was recorded at 40 fps and
# lasts for approximately 27 seconds.

# %%
# Apply Kalman filter to smooth position data
# --------------------------------------------
# The Kalman filter uses a constant acceleration model to smooth position
# measurements. It estimates position, velocity, and acceleration while
# accounting for measurement noise and process uncertainty.
#
# Let's apply the Kalman filter with default parameters to smooth the
# position data:

position_smooth = kalman_filter(
    ds_wasp.position,
    process_noise=0.01,
    measurement_noise=1.0,
    output="position",
    print_report=True,
)

# %%
# The Kalman filter has smoothed the position data. Let's compare the raw
# and smoothed trajectories for the stinger keypoint:

keypoint = "stinger"
time_range = slice(0, 5)  # First 5 seconds

fig, ax = plt.subplots(2, 1, figsize=(10, 6))

for color, label, position_data in zip(
    ["k", "r"],
    ["raw", "Kalman filtered"],
    [ds_wasp.position, position_smooth],
    strict=False,
):
    pos_x = position_data.sel(keypoints=keypoint, space="x", time=time_range)
    pos_y = position_data.sel(keypoints=keypoint, space="y", time=time_range)

    ax[0].plot(pos_x.time, pos_x, color=color, lw=2, alpha=0.7, label=f"{label} x")
    ax[1].plot(pos_y.time, pos_y, color=color, lw=2, alpha=0.7, label=f"{label} y")

ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("X position (pixels)")
ax[0].set_title(f"{keypoint} X position")
ax[0].legend()
ax[0].grid(True, alpha=0.3)

ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Y position (pixels)")
ax[1].set_title(f"{keypoint} Y position")
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Compute velocity and acceleration
# ----------------------------------
# The Kalman filter can also estimate velocity and acceleration directly
# from the position measurements. Let's get all three outputs:

results = kalman_filter(
    ds_wasp.position,
    process_noise=0.01,
    measurement_noise=1.0,
    output="all",
)

# %%
# Now we have position, velocity, and acceleration estimates. Let's plot
# them for the stinger keypoint:

keypoint = "stinger"
time_range = slice(0, 5)  # First 5 seconds

fig, ax = plt.subplots(3, 1, figsize=(10, 8))

for i, (var_name, var_data) in enumerate(
    [
        ("Position", results.position),
        ("Velocity", results.velocity),
        ("Acceleration", results.acceleration),
    ]
):
    for j, (coord, space) in enumerate([("X", "x"), ("Y", "y")]):
        data = var_data.sel(keypoints=keypoint, space=space, time=time_range)
        ax[i].plot(
            data.time,
            data,
            lw=2,
            alpha=0.7,
            label=f"{coord} {var_name.lower()}",
        )

    ax[i].set_xlabel("Time (s)")
    ax[i].set_ylabel(f"{var_name} (pixels/s^{i if i > 0 else ''})")
    ax[i].set_title(f"{keypoint} {var_name}")
    ax[i].legend()
    ax[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Compare with frequency domain
# ------------------------------
# Let's compare the power spectral density (PSD) of the raw and filtered
# position data to see how the Kalman filter affects different frequency
# components:

keypoint = "stinger"
space = "x"
time_range = slice(0, 10)  # First 10 seconds

fig, ax = plt.subplots(2, 1, figsize=(10, 6))

for color, label, position_data in zip(
    ["k", "r"],
    ["raw", "Kalman filtered"],
    [ds_wasp.position, position_smooth],
    strict=False,
):
    pos = position_data.sel(keypoints=keypoint, space=space, time=time_range)

    # Interpolate to remove NaNs for PSD calculation
    pos_interp = interpolate_over_time(pos, fill_value="extrapolate")

    # Compute and plot the PSD
    freq, psd = welch(pos_interp, fs=ds_wasp.fps, nperseg=256)
    ax[0].semilogy(
        freq,
        psd,
        color=color,
        lw=2,
        alpha=0.7,
        label=f"{label} {space}",
    )

    # Plot time series
    ax[1].plot(
        pos.time,
        pos,
        color=color,
        lw=2,
        alpha=0.7,
        label=f"{label} {space}",
    )

ax[0].set_xlabel("Frequency (Hz)")
ax[0].set_ylabel("Power Spectral Density")
ax[0].set_title(f"PSD of {keypoint} {space} position")
ax[0].legend()
ax[0].grid(True, alpha=0.3)

ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Position (pixels)")
ax[1].set_title(f"{keypoint} {space} position")
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Tuning filter parameters
# -------------------------
# The Kalman filter has two main parameters that control its behavior:
#
# - ``process_noise``: Controls how much we trust the dynamic model.
#   Larger values make the filter more responsive to changes but less smooth.
# - ``measurement_noise``: Controls how much we trust the measurements.
#   Larger values make the filter trust the model more than the measurements.
#
# Let's compare the effect of different parameter values:

keypoint = "stinger"
space = "x"
time_range = slice(0, 5)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Raw data
pos_raw = ds_wasp.position.sel(keypoints=keypoint, space=space, time=time_range)
ax.plot(pos_raw.time, pos_raw, "k-", lw=2, alpha=0.3, label="raw", zorder=1)

# Different parameter combinations
param_combinations = [
    (0.001, 1.0, "Low process noise (smooth)"),
    (0.01, 1.0, "Default"),
    (0.1, 1.0, "High process noise (responsive)"),
    (0.01, 0.1, "Low measurement noise (trust measurements)"),
    (0.01, 10.0, "High measurement noise (trust model)"),
]

colors = ["r", "b", "g", "orange", "purple"]

for (proc_noise, meas_noise, label), color in zip(
    param_combinations, colors, strict=False
):
    pos_smooth = kalman_filter(
        ds_wasp.position.sel(keypoints=keypoint, space=space, time=time_range),
        process_noise=proc_noise,
        measurement_noise=meas_noise,
        output="position",
    )
    ax.plot(
        pos_smooth.time,
        pos_smooth,
        color=color,
        lw=2,
        alpha=0.7,
        label=label,
        zorder=2,
    )

ax.set_xlabel("Time (s)")
ax.set_ylabel("Position (pixels)")
ax.set_title(f"{keypoint} {space} position - Parameter tuning")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Handling missing data
# ----------------------
# The Kalman filter can bridge gaps in the data by predicting forward based
# on the current velocity and acceleration estimates. Let's see how it handles
# missing data:

# Create a copy with some missing values
ds_wasp_with_gaps = ds_wasp.copy()
ds_wasp_with_gaps.position.loc[
    {"keypoints": "stinger", "time": slice(2.0, 2.5)}
] = float("nan")

# Apply Kalman filter
position_smooth_gaps = kalman_filter(
    ds_wasp_with_gaps.position,
    process_noise=0.01,
    measurement_noise=1.0,
    output="position",
    print_report=True,
)

# Compare raw (with gaps) and filtered
keypoint = "stinger"
space = "x"
time_range = slice(1.5, 3.0)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

pos_raw_gaps = ds_wasp_with_gaps.position.sel(
    keypoints=keypoint, space=space, time=time_range
)
pos_smooth_gaps_sel = position_smooth_gaps.sel(
    keypoints=keypoint, space=space, time=time_range
)

ax.plot(
    pos_raw_gaps.time,
    pos_raw_gaps,
    "ko",
    markersize=4,
    alpha=0.5,
    label="raw (with gaps)",
    zorder=1,
)
ax.plot(
    pos_smooth_gaps_sel.time,
    pos_smooth_gaps_sel,
    "r-",
    lw=2,
    alpha=0.7,
    label="Kalman filtered (bridged gaps)",
    zorder=2,
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Position (pixels)")
ax.set_title(f"{keypoint} {space} position - Handling missing data")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Update the dataset with smoothed data
# --------------------------------------
# You can update the dataset with the smoothed position data:

ds_wasp["position"] = position_smooth
ds_wasp["velocity"] = results.velocity
ds_wasp["acceleration"] = results.acceleration

print(ds_wasp)
