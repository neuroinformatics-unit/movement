from filterpy.kalman import KalmanFilter
import numpy as np
import xarray as xr


def create_kalman_filter(
    model_type: str = "pos_vel",
    process_variance: float = 1e-3,
    measurement_variance: float = 1e-2,
    covariance_matrix: int = 10,
    dt: float = 1 / 40,
) -> xr.DataArray:
    """
    General Kalman filter for different motion models.

    model_type:
        - "pos" (Position only)
        - "pos_vel" (Position + velocity) [Default]
        - "pos_vel_acc" (Position + velocity + acceleration)
        - "pos_vel_measured" (Position + velocity, but velocity is also measured)
    """

    if model_type == "pos":
        dim_x, dim_z = 2, 2
        F = np.eye(2)
        H = np.eye(2)

    elif model_type == "pos_vel":
        dim_x, dim_z = 4, 2
        F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    elif model_type == "pos_vel_acc":
        dim_x, dim_z = 6, 2
        F = np.array(
            [
                [1, 0, dt, 0, 0.5 * dt**2, 0],
                [0, 1, 0, dt, 0, 0.5 * dt**2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    elif model_type == "pos_vel_measured":
        dim_x, dim_z = 4, 4
        F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        H = np.eye(4)
    else:
        raise ValueError("Invalid model_type")

    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf.F = F
    kf.H = H
    kf.Q = process_variance * np.eye(dim_x)
    kf.R = measurement_variance * np.eye(dim_z)
    kf.P = covariance_matrix

    return kf


def fit_kalman(
    data: xr.DataArray, Kalman_filter: create_kalman_filter
) -> xr.DataArray:
    filtered_data = data.values
    num_time_steps, _, num_keypoints, num_individuals = filtered_data.shape

    filtered_positions = np.zeros_like(filtered_data)

    for i in range(num_individuals):
        for k in range(num_keypoints):
            kf = Kalman_filter

            kf.x = np.array(
                [filtered_data[0, 0, k, i], filtered_data[0, 1, k, i], 0, 0]
            )

            for t in range(num_time_steps):
                kf.predict()
                kf.update(filtered_data[t, :, k, i])
                filtered_positions[t, :, k, i] = kf.x[:2]

    return filtered_positions
