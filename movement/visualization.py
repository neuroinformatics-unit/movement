"""Visualization utilities for Kalman filter motion tracking results."""

import napari
import numpy as np


def animate_motion_tracks(dataset, trail_length=1):
    """Animate motion tracks frame-by-frame using Napari."""
    x_raw = dataset["x"].values
    y_raw = dataset["y"].values

    if "x_filtered" in dataset and "y_filtered" in dataset:
        x_filt = dataset["x_filtered"].values
        y_filt = dataset["y_filtered"].values
    else:
        x_filt, y_filt = x_raw, y_raw

    viewer = napari.Viewer(ndisplay=2)
    viewer.dims.ndim = 1  # ✅ Ensure frame slider appears

    raw_layer = viewer.add_points(
        np.array([[x_raw[0], y_raw[0]]]),
        size=5,
        face_color="red",
        name="Raw Data",
    )
    filt_layer = viewer.add_points(
        np.array([[x_filt[0], y_filt[0]]]),
        size=5,
        face_color="green",
        name="Filtered Data",
    )

    def update_frame(frame_idx):
        """Update the displayed points for each frame."""
        start_idx = max(0, frame_idx - trail_length + 1)
        raw_points = np.column_stack(
            (
                x_raw[start_idx : frame_idx + 1],
                y_raw[start_idx : frame_idx + 1],
            )
        )
        filt_points = np.column_stack(
            (
                x_filt[start_idx : frame_idx + 1],
                y_filt[start_idx : frame_idx + 1],
            )
        )
        raw_layer.data = raw_points
        filt_layer.data = filt_points

    viewer.dims.events.current_step.connect(
        lambda event: update_frame(viewer.dims.current_step[0])
    )
    napari.run()
