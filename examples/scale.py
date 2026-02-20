"""Scale pose tracks
=====================

Scale 2D pose tracks to real world units using known distances.
"""

# %%
# Imports
# -------

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from movement import sample_data
from movement.filtering import (
    filter_by_confidence,
    interpolate_over_time,
    rolling_filter,
)
from movement.kinematics import compute_pairwise_distances
from movement.transforms import scale

# %%
# Load sample dataset
# -------------------
# In this example, we will use the ``DLC_single-mouse_DBTravelator_2D``
# sample dataset, which contains DeepLabCut predictions of a single
# mouse running across a dual-belt travelator, where the second belt
# runs faster than the first.
#
# The back wall of the travelator has a visible 1 cm grid, which
# we can use as a scaling reference to convert pixel coordinates into
# real-world units.

ds = sample_data.fetch_dataset(
    "DLC_single-mouse_DBTravelator_2D.predictions.h5", with_video=True
)
print(ds)

# %%
# We can see the DeepLabCut dataset contains positions and confidence scores
# for 50 keypoints tracked on a single mouse, recorded at 247 fps over
# approximately 1.7 seconds.

# %%
# Prepare the dataset
# -------------------
# Before scaling, let's inspect the tracked keypoints and clean up the
# data.

# %%
# We can see that out of the 50 tracked keypoints, the majority track
# positions along the mouse's body (head, back, tail, and limbs).
print(ds.keypoints.values)

# %%
# However, there are also a number of keypoints which track physical
# landmarks on the apparatus rather than the mouse itself. We don't need
# these so we can filter them out of the dataset.

landmark_keypoints = [
    "Door",
    "StartPlatL",
    "StartPlatR",
    "StepL",
    "StepR",
    "TransitionL",
    "TransitionR",
]

# Select all keypoints excluding the landmarks, and the single individual.
ds_mouse = ds.sel(
    keypoints=~ds.keypoints.isin(landmark_keypoints),
    individuals="individual_0",
)

print(ds_mouse)
print("----------------------------------")
print(f"Keypoints:\n{ds_mouse.keypoints.values}")

# %%
# Next we remove low-confidence predictions, interpolate over gaps,
# and apply a rolling median filter to suppress any remaining tracking
# outliers.

ds_mouse.update(
    {
        "position": filter_by_confidence(
            ds_mouse.position,
            ds_mouse.confidence,
            threshold=0.9,
            print_report=True,
        )
    }
)
ds_mouse.update(
    {
        "position": interpolate_over_time(
            ds_mouse.position, max_gap=40, print_report=True
        )
    }
)
ds_mouse.update(
    {
        "position": rolling_filter(
            ds_mouse.position,
            window=6,
            min_periods=2,
            statistic="median",
            print_report=True,
        )
    }
)


# %%
# Visualise the skeleton in pixels
# --------------------------------
# To see what the pose data looks like before scaling, we define a helper
# function that draws the mouse skeleton in a single frame.


def plot_skeleton(
    position_data,
    skeleton,
    frame,
    ax,
    s=3,
    **plot_kwargs,
):
    """Plot the mouse skeleton for a single frame.

    Parameters
    ----------
    position_data : xarray.DataArray
        Position data with dimensions ``time``, ``keypoints``, and ``space``.
    skeleton : list of tuple
        List of (joint_1, joint_2) pairs defining skeletal connections,
        where each element is a keypoint name present in ``position_data``.
    frame : int
        Index of the time frame to plot.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw.
    s : float, optional
        Marker size for keypoint scatter points. Default is 3.
    **plot_kwargs
        Additional keyword arguments passed to the plot and scatter calls.
        Supports ``alpha`` (default 1) and ``linewidth`` (default 1.5).

    """
    defaults = {"alpha": 1, "linewidth": 1.5}
    defaults.update(plot_kwargs)

    pos_frame = position_data.squeeze().isel(time=frame)

    # Draw skeleton connections
    for joint_1, joint_2 in skeleton:
        x1, y1 = pos_frame.sel(keypoints=joint_1)
        x2, y2 = pos_frame.sel(keypoints=joint_2)
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="b",
            alpha=defaults["alpha"],
            linewidth=defaults["linewidth"],
        )

    # Draw keypoints
    for bodypart in pos_frame.keypoints:
        x, y = pos_frame.sel(keypoints=bodypart)
        ax.scatter(x, y, c="g", s=s, alpha=defaults["alpha"])


# %%
# We define the connections between keypoints that form the mouse skeleton,
# then plot it for a single frame using the above function.

skeleton = [
    ("Nose", "Back1"),
    ("EarL", "Back1"),
    ("EarR", "Back1"),
    ("Back1", "Back2"),
    ("Back2", "Back3"),
    ("Back3", "Back4"),
    ("Back4", "Back5"),
    ("Back5", "Back6"),
    ("Back6", "Back7"),
    ("Back7", "Back8"),
    ("Back8", "Back9"),
    ("Back9", "Back10"),
    ("Back10", "Back11"),
    ("Back11", "Back12"),
    ("Back12", "Tail1"),
    ("Tail1", "Tail2"),
    ("Tail2", "Tail3"),
    ("Tail3", "Tail4"),
    ("Tail4", "Tail5"),
    ("Tail5", "Tail6"),
    ("Tail6", "Tail7"),
    ("Tail7", "Tail8"),
    ("Tail8", "Tail9"),
    ("Tail9", "Tail10"),
    ("Tail10", "Tail11"),
    ("Tail11", "Tail12"),
    ("Back3", "ForepawKneeL"),
    ("Back3", "ForepawKneeR"),
    ("Back9", "HindpawKneeL"),
    ("Back9", "HindpawKneeR"),
    ("ForepawKneeL", "ForepawAnkleL"),
    ("ForepawKneeR", "ForepawAnkleR"),
    ("HindpawKneeL", "HindpawAnkleL"),
    ("HindpawKneeR", "HindpawAnkleR"),
    ("ForepawAnkleL", "ForepawKnuckleL"),
    ("ForepawAnkleR", "ForepawKnuckleR"),
    ("HindpawAnkleL", "HindpawKnuckleL"),
    ("HindpawAnkleR", "HindpawKnuckleR"),
    ("ForepawKnuckleL", "ForepawToeL"),
    ("ForepawKnuckleR", "ForepawToeR"),
    ("HindpawKnuckleL", "HindpawToeL"),
    ("HindpawKnuckleR", "HindpawToeR"),
]

example_frame = 275

fig, ax = plt.subplots()
plot_skeleton(ds_mouse.position, skeleton, frame=example_frame, ax=ax, s=10)

ax.invert_yaxis()  # image coordinates have y increasing downward
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_aspect("equal")
plt.show()

# %%
# The skeleton gives us the shape of the mouse's body, but the coordinate
# system is in pixels. To interpret sizes and distances in real-world units,
# we need to convert to physical units. We can do this by measuring a known
# distance in the video using the ``napari`` GUI and then applying
# :func:`movement.transforms.scale` to convert from pixels to centimetres.

# %%
# Measure a known distance from video footage
# -------------------------------------------
# First, open the video file in napari:
#
# .. code-block:: python
#
#    import napari
#    viewer = napari.Viewer()
#    viewer.open(ds_mouse.video_path)
#
# N.B. You can also load a single frame image instead of the full video.
#
# Next, measure a known distance in the napari viewer:
#
# 1. Add a new shapes layer by clicking 'New shapes layer' in the layer list.
# 2. Select the 'Add lines' tool (shortcut: L) from the layer controls.
# 3. Draw a line across a feature of known length. In this case, the grid
#    squares are 1x1 cm, so we can draw a line along one side of a grid square.
#
# .. image:: /_static/napari_scale_draw.png
#    :width: 600
#
# 4. To read the line length in pixels, go to Layers → Measure → Toggle
# shape dimensions measurement (napari builtins).
#
# .. image:: /_static/napari_scale_measure.png
#    :width: 600
#
# 5. Close the napari viewer.

# %%
# We can then retrieve the measured distance and line coordinates from the
# shapes layer.

# sphinx_gallery_start_ignore
import pandas as pd  # noqa: E402


class MockShapesLayer:
    """Mock napari shapes layer for testing."""

    def __init__(self):
        """Initialise with dummy perimeter, area, and data attributes."""
        self.features = pd.DataFrame(
            {"_perimeter": [29.063293], "_area": [0.0]}
        )
        self.data = [
            np.array(
                [[275.0, 48.455124, 930.55444], [275.0, 48.452595, 959.61774]],
                dtype=np.float32,
            )
        ]


class MockLayers:
    """Mock napari layers collection for testing."""

    def __getitem__(self, key):
        """Return a MockShapesLayer for any key."""
        return MockShapesLayer()


class MockViewer:
    """Mock napari viewer for testing."""

    layers = MockLayers()


viewer = MockViewer()
# sphinx_gallery_end_ignore

shapes_layer = viewer.layers["Shapes"]
measurements_px = shapes_layer.features
shape_coords = shapes_layer.data

print(f"Measurements:\n{measurements_px}\n")
print(f"Coordinates:\n{shape_coords}")

# Extract the perimeter of the drawn shape in pixels
distance_px = measurements_px["_perimeter"].values.squeeze()

# %%
# Scale poses to real units
# -------------------------
# The measured line spans one grid square, which we know to be 1 cm.
# Dividing this known length by the distance in pixels gives us the scaling
# factor.

scaling_factor = 1 / distance_px  # cm per pixel
print(f"Scaling factor: {scaling_factor:.6f} cm/pixel")

# %%
# Let's inspect the position values before scaling.

# Select a frame range in which the mouse is visible
sample_range = np.arange(300, 305)
print(ds_mouse.position.isel(time=sample_range).values)

# %%
# Now we apply :func:`movement.transforms.scale` to convert from pixels to
# centimetres. We can assign our space unit 'cm' here as an attribute in
# ``xarray.DataArray.attrs['space_unit']``.

ds_mouse["position"] = scale(
    ds_mouse["position"], factor=scaling_factor, space_unit="cm"
)

# %%
# Now we inspect our sample of values again. We can see the values have been
# adjusted.

print(ds_mouse.position.isel(time=sample_range).values)

# %%
# The scaled data array's attributes now contain the ``space_unit`` and a
# ``log`` entry recording the operation and its parameters, alongside the
# operations applied in earlier steps.

print(f"Unit:\n{ds_mouse['position'].space_unit}\n")
print(f"Log:\n{ds_mouse['position'].log}")

# %%
# Furthermore, since image coordinates have the y-axis pointing downward,
# we further transform the data array by flipping and shifting relative to
# the maximum so that positive y corresponds to upward in real world space.

y = ds_mouse["position"].sel(space="y")
ds_mouse["position"].loc[dict(space="y")] = y.max() - y

# %%
# We can now re-plot the same skeleton, this time in centimetres and with
# the y-axis pointing upward.

fig, ax = plt.subplots()
plot_skeleton(ds_mouse.position, skeleton, frame=example_frame, ax=ax, s=10)

ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_aspect("equal")
plt.show()

# %%
# Working with real-world distances
# ----------------------------------
# With the data now in centimetres, we can measure the mouse directly. Let's
# first compute its approximate body length and height in ``example_frame``.

frame_x = ds_mouse.position.sel(space="x").isel(time=example_frame)
frame_y = ds_mouse.position.sel(space="y").isel(time=example_frame)

mouse_length = frame_x.sel(keypoints="Nose") - frame_x.sel(keypoints="Tail12")
mouse_height = frame_y.max() - frame_y.min()

print(f"Mouse length: {mouse_length.values:.1f} cm")
print(f"Mouse height: {mouse_height.values:.1f} cm")

# %%
# Beyond simple static measurements, real-world units also make aspects of
# movement, such as gait-related distances interpretable. Let's visualise
# the paw trajectories in x and y over time.

# Select the toe keypoints for all four limbs
toe_keypoint_names = [
    "ForepawToeR",
    "ForepawToeL",
    "HindpawToeR",
    "HindpawToeL",
]
colours = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# Construct a new data array with only the toe keypoints
ds_limbs = ds_mouse.position.sel(keypoints=toe_keypoint_names)

fig, axs = plt.subplots(2)
for i, limb in enumerate(toe_keypoint_names):
    # Plot limb trajectories in x
    axs[0].plot(
        ds_limbs.time,
        ds_limbs.sel(keypoints=limb, space="x"),
        color=colours[i],
        label=limb,
    )
    # Plot limb trajectories in y
    axs[1].plot(
        ds_limbs.time,
        ds_limbs.sel(keypoints=limb, space="y"),
        color=colours[i],
    )

axs[0].legend()
axs[0].set_ylabel("x (cm)")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("y (cm)")
fig.tight_layout()
plt.show()

# %%
# Here we see the mouse traverses approximately 60 cm across the travelator,
# and step heights typically remain below around 0.5 cm.

# %%
# These trajectories give us a broad sense of limb movement, but we can go
# further and quantify dynamics between limbs, such as the inter-limb
# distances in real-world units using
# :func:`movement.kinematics.compute_pairwise_distances`. Here, we again use
# the toe keypoints to compare the anterior-posterior separation of the fore-
# and hindpaw pairs.

forepaw_dist = compute_pairwise_distances(
    ds_mouse.position,
    dim="keypoints",
    pairs={"ForepawToeL": "ForepawToeR"},
)
hindpaw_dist = compute_pairwise_distances(
    ds_mouse.position,
    dim="keypoints",
    pairs={"HindpawToeL": "HindpawToeR"},
)

# %%
# We plot these inter-limb distances for the fore- and hindpaws against the
# mean x-position of the body as the mouse traverses the travelator. Since
# we know the first belt is 47 cm long, we can also easily plot the
# transition point at which the mouse moves onto the faster second belt.

belt_transition = 47  # cm

# Compute a proxy for body centre in x
avg_body_x = ds_mouse.position.sel(space="x").mean(dim="keypoints")

fig, ax = plt.subplots()
ax.plot(avg_body_x, forepaw_dist, color="tab:blue", label="Forepaws")
ax.plot(avg_body_x, hindpaw_dist, color="tab:orange", label="Hindpaws")
ax.axvline(belt_transition, color="k", linestyle="--", label="Belt transition")

ax.set_xlabel("x (cm)")
ax.set_ylabel("Inter-limb distance (cm)")
plt.legend()
plt.show()

# %%
# The forepaw and hindpaw inter-limb distances oscillate largely in
# synchrony, consistent with a trotting gait, with peak separations of
# approximately 4 cm for the majority of strides.
#
# However, after the mouse's body centre passes the belt transition,
# the forepaw inter-limb distance drops noticeably whilst the hindpaw
# distance increases. This likely reflects the mouse accommodating the speed
# difference between belts - the forepaws shorten their stride to avoid
# over-reaching, while the hindpaws widen to keep the rear of the body from
# trailing behind.
#
# We can quantify these observations by comparing peak inter-limb distances
# across all strides with those during only the transitioning stride.
#
# First, let's compute the mean peak inter-limb distance across all strides.

# Find maximum interlimb distances in each stride by locating the peaks.
forepaw_peaks, _ = find_peaks(forepaw_dist.values.squeeze(), prominence=0.2)
hindpaw_peaks, _ = find_peaks(hindpaw_dist.values.squeeze(), prominence=0.2)

# Find inter-limb distances at these peak locations
forepaw_peak_vals = forepaw_dist.values.squeeze()[
    forepaw_peaks[2:]
]  # Exclude the first two peaks to match the available hindpaw strides
hindpaw_peak_vals = hindpaw_dist.values.squeeze()[hindpaw_peaks]

forepaw_mean = forepaw_peak_vals.mean()
hindpaw_mean = hindpaw_peak_vals.mean()

forepaw_std = forepaw_peak_vals.std()
hindpaw_std = hindpaw_peak_vals.std()

paw_diffs = hindpaw_peak_vals - forepaw_peak_vals
paw_diffs_mean = np.mean(paw_diffs)
paw_diffs_std = np.std(paw_diffs)

print(
    f"Mean peak inter-limb distance (forepaws): "
    f"{forepaw_mean:.2f} ± {forepaw_std:.2f} cm (± std)"
)
print(
    f"Mean peak inter-limb distance (hindpaws): "
    f"{hindpaw_mean:.2f} ± {hindpaw_std:.2f} cm (± std)"
)
print(
    f"Hindpaw - forepaw inter-limb difference: "
    f"{paw_diffs_mean:.2f} ± {paw_diffs_std:.2f} cm (± std)"
)

# %%
# Now let's compare these average distances with the maximum inter-limb
# distances during the transitioning stride where the mouse steps onto the
# faster second belt (the second-to-last peak in each trace).

post_forepaw_dist = forepaw_dist.values.squeeze()[forepaw_peaks[-2]]
post_hindpaw_dist = hindpaw_dist.values.squeeze()[hindpaw_peaks[-2]]

print(
    f"Transitioning forepaw inter-limb distance:  {post_forepaw_dist:.2f} cm"
)
print(
    f"Transitioning hindpaw inter-limb distance:  {post_hindpaw_dist:.2f} cm"
)
print(
    f"Hindpaw–forepaw difference at transition:     "
    f"{post_hindpaw_dist - post_forepaw_dist:.2f} cm"
)

# %%
# Here we can see that during typical locomotion, the fore- and hindpaw
# inter-limb distances are synchronised and closely matched, differing by
# only ~30 mm on average. During the transitioning stride, however,
# this difference grows more than tenfold, reflecting the gait adjustments
# the mouse makes to accommodate the speed differential between belts.

# %%
# Having the data in real-world units makes these quantitative comparisons
# possible.
