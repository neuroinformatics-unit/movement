"""Kalman filtering implementation for movement tracking.

This module applies Kalman filtering to smooth noisy position data.
"""

import h5py
import numpy as np
import xarray as xr
from filterpy.kalman import KalmanFilter
from scipy.spatial.distance import mahalanobis

from movement import sample_data


def load_sleap_data_xarray(file_path):
    """Load SLEAP data into an xarray Dataset with error handling."""
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
        {
            "x": ("frame", poses[:, 0, 0]),
            "y": ("frame", poses[:, 0, 1]),
        },
        coords={"frame": np.arange(len(poses))},
    )
    return dataset


def preprocess_xarray_data(dataset):
    """Handle missing values in xarray Dataset using interpolation."""
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
    process_noise=1e-3,
    measurement_noise=1.0,
    initial_state_uncertainty=10,
    outlier_threshold=3.0,
):
    """Apply Kalman filter to xarray data with adaptive noise rejection."""
    if "x" not in dataset or "y" not in dataset:
        raise KeyError("Dataset must contain 'x' and 'y' variables.")

    x_var = dataset["x"].values
    y_var = dataset["y"].values
    measurements = np.array(list(zip(x_var, y_var, strict=False)))

    kf = KalmanFilter(dim_x=6, dim_z=2)  # [x, vx, ax, y, vy, ay]
    kf.x = np.zeros((6, 1))
    kf.P = np.eye(6) * initial_state_uncertainty
    dt = 1.0  # Time step
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
    kf.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    kf.R = np.eye(2) * measurement_noise
    kf.Q = np.diag([process_noise] * 6)

    filtered_positions, velocities, accelerations = [], [], []
    prev_measurement = None

    for measurement in measurements:
        kf.predict()

        # Mahalanobis outlier rejection using pseudo-inverse
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

        filtered_positions.append([kf.x[0, 0], kf.x[3, 0]])
        velocities.append([kf.x[1, 0], kf.x[4, 0]])
        accelerations.append([kf.x[2, 0], kf.x[5, 0]])

    dataset["x_filtered"] = xr.DataArray(
        np.array(filtered_positions)[:, 0], dims=["frame"]
    )
    dataset["y_filtered"] = xr.DataArray(
        np.array(filtered_positions)[:, 1], dims=["frame"]
    )
    dataset["vx_filtered"] = xr.DataArray(
        np.array(velocities)[:, 0], dims=["frame"]
    )
    dataset["vy_filtered"] = xr.DataArray(
        np.array(velocities)[:, 1], dims=["frame"]
    )
    dataset["ax_filtered"] = xr.DataArray(
        np.array(accelerations)[:, 0], dims=["frame"]
    )
    dataset["ay_filtered"] = xr.DataArray(
        np.array(accelerations)[:, 1], dims=["frame"]
    )

    # Ensure velocity and acceleration names match test expectations
    dataset["x_velocity"] = dataset["vx_filtered"]
    dataset["y_velocity"] = dataset["vy_filtered"]
    dataset["x_acceleration"] = dataset["ax_filtered"]
    dataset["y_acceleration"] = dataset["ay_filtered"]

    return dataset


# Example usage
file_path = sample_data.fetch_dataset_paths(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5"
)["poses"]
dataset = load_sleap_data_xarray(file_path)
dataset = preprocess_xarray_data(dataset)
dataset = apply_kalman_filter_xarray(dataset)
