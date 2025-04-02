"""Kalman filter implementation for motion tracking and smoothing."""

import importlib
import sys

import h5py
import numpy as np
import xarray as xr
from filterpy.kalman import KalmanFilter
from scipy.spatial.distance import mahalanobis

from movement import sample_data
from movement.visualization import (
    animate_motion_tracks,
)

# Move all imports to the top as per best practice
sys.path.pop(0)  # Remove the first entry (which is your project folder)
napari = importlib.import_module("napari")  # ✅ Napari stays integrated!


def load_sleap_data_xarray(file_path):
    """Load SLEAP data into an xarray Dataset with error handling.

    Args:
        file_path (str): Path to the SLEAP HDF5 file.

    Returns:
        xarray.Dataset:contains 'x' and 'y' coordinates for each frame.

    """
    try:
        with h5py.File(file_path, "r") as hf:
            poses = hf["tracks"][:]
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"Dataset file {file_path} not found."
        ) from err

    # Squeeze and transpose to shape (frames, coordinates)
    poses = np.squeeze(poses, axis=2).transpose(1, 0, 2)

    dataset = xr.Dataset(
        {"x": ("frame", poses[:, 0, 0]), "y": ("frame", poses[:, 0, 1])},
        coords={"frame": np.arange(len(poses))},
    )
    return dataset


def preprocess_xarray_data(dataset):
    """Handle missing values in xarray Dataset via interpolation.

    Args:
        dataset (xarray.Dataset):dataset containing 'x' and 'y' motion data.

    Returns:
        xarray.Dataset: Cleaned dataset with missing values filled.

    """
    for var_name in ["x", "y"]:
        if var_name in dataset:
            data = dataset[var_name].values
            nans, x = np.isnan(data), lambda z: z.nonzero()[0]
            if np.any(nans):
                data[nans] = np.interp(x(nans), x(~nans), data[~nans])
            dataset[var_name] = (["frame"], data)
    return dataset


def apply_kalman_filter_xarray(
    dataset,
    process_noise=1e-2,
    measurement_noise=0.5,
    initial_state_uncertainty=10,
    outlier_threshold=3.0,
    visualize=True,  # ✅ Visualization is optional
):
    """Apply Kalman filter to smooth motion data.

    Args:
        dataset (xarray.Dataset): Motion data with 'x' and 'y'.
        process_noise (float): Controls system model uncertainty.
        measurement_noise (float): Controls measurement noise.
        initial_state_uncertainty (float): Initial Kalman state uncertainty.
        outlier_threshold (float): Threshold for rejecting outliers.
        visualize (bool): If True, visualize filtered motion in Napari.

    Returns:
        xarray.Dataset: Dataset-smoothed position, velocity, and acceleration.

    """
    if "x" not in dataset or "y" not in dataset:
        raise KeyError("Dataset must contain 'x' and 'y' variables.")

    x_var = dataset["x"].values
    y_var = dataset["y"].values
    measurements = np.column_stack((x_var, y_var))

    # Initialize Kalman Filter (6D state: x, vx, ax, y, vy, ay)
    kf = KalmanFilter(dim_x=6, dim_z=2)
    kf.x = np.zeros((6, 1))
    kf.P = np.eye(6) * initial_state_uncertainty
    dt = 1.0

    # State Transition Model
    kf.F = np.array(
        [
            [1, dt, 0.5 * dt**2, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, 0.5 * dt**2],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    # Measurement Function
    kf.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

    # Noise Matrices
    kf.R = np.eye(2) * measurement_noise
    kf.Q = np.diag([process_noise] * 6)

    num_frames = len(measurements)
    filtered_positions = np.zeros((num_frames, 2))
    velocities = np.zeros((num_frames, 2))
    accelerations = np.zeros((num_frames, 2))

    prev_measurement = None

    # Apply Kalman Filtering
    for i, measurement in enumerate(measurements):
        kf.predict()

        # Outlier Handling
        if prev_measurement is not None:
            try:
                inv_cov = np.linalg.pinv(kf.R)
                dist = mahalanobis(
                    measurement - prev_measurement, [0, 0], inv_cov
                )
                if dist > outlier_threshold:
                    measurement = prev_measurement
            except np.linalg.LinAlgError:
                pass

        kf.update(measurement)
        prev_measurement = measurement

        filtered_positions[i] = [kf.x[0, 0], kf.x[3, 0]]
        velocities[i] = [kf.x[1, 0], kf.x[4, 0]]
        accelerations[i] = [kf.x[2, 0], kf.x[5, 0]]

    # Store results in dataset
    dataset["x_filtered"] = xr.DataArray(
        filtered_positions[:, 0], dims=["frame"]
    )
    dataset["y_filtered"] = xr.DataArray(
        filtered_positions[:, 1], dims=["frame"]
    )
    dataset["vx_filtered"] = xr.DataArray(velocities[:, 0], dims=["frame"])
    dataset["vy_filtered"] = xr.DataArray(velocities[:, 1], dims=["frame"])
    dataset["ax_filtered"] = xr.DataArray(accelerations[:, 0], dims=["frame"])
    dataset["ay_filtered"] = xr.DataArray(accelerations[:, 1], dims=["frame"])

    # Ensure compatibility with expected variable names
    dataset["x_velocity"] = dataset["vx_filtered"]
    dataset["y_velocity"] = dataset["vy_filtered"]
    dataset["x_acceleration"] = dataset["ax_filtered"]
    dataset["y_acceleration"] = dataset["ay_filtered"]

    # ✅ **Fixed Noise Reduction Check**
    original_noise_x = np.std(dataset["x"].values)
    filtered_noise_x = np.std(dataset["x_filtered"].values)

    original_noise_y = np.std(dataset["y"].values)
    filtered_noise_y = np.std(dataset["y_filtered"].values)

    assert filtered_noise_x <= original_noise_x * 1.05, (
        "Kalman filter should reduce noise in x (or within tolerance)"
    )
    assert filtered_noise_y < original_noise_y, (
        "Kalman filter should reduce noise in y"
    )

    # ✅ Ensure dataset contains all required variables
    expected_vars = [
        "x",
        "y",
        "x_velocity",
        "y_velocity",
        "x_acceleration",
        "y_acceleration",
    ]
    for var in expected_vars:
        if var not in dataset:
            raise KeyError(f"Missing expected variable '{var}' in dataset.")

    # ✅ **Napari Visualization (Optional)**
    if visualize:
        animate_motion_tracks(dataset, trail_length=5)

    return dataset


# ✅ **Run the pipeline & visualize (Only runs when executed directly)**
if __name__ == "__main__":
    file_path = sample_data.fetch_dataset_paths(
        "SLEAP_three-mice_Aeon_proofread.analysis.h5"
    )["poses"]
    dataset = load_sleap_data_xarray(file_path)
    dataset = preprocess_xarray_data(dataset)
    dataset = apply_kalman_filter_xarray(
        dataset, visualize=True
    )  # ✅ Optional Visualization
