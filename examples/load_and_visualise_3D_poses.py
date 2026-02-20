"""Load and visualise 3D pose tracks
====================================

Visualise 3D pose tracks from multi-camera data and explore
kinematic features accessible with 3D tracking.
"""

# %%
# Imports
# -------
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from movement import sample_data
from movement.filtering import filter_by_confidence, interpolate_over_time
from movement.kinematics import compute_pairwise_distances, compute_speed
from movement.utils.vector import compute_norm

# %%
# Load sample dataset
# -------------------
# In this example, we will use the ``DLC_single-mouse_DBTravelator_3D`` example
# dataset. This dataset contains 3D pose tracks of a mouse locomoting on a
# dual-belt treadmill apparatus, where the second belt moves faster than
# the first.
#
# This dataset was derived via triangulation of DeepLabCut predictions from
# three synchronised camera views (side, front, and overhead). The pose tracks
# from the side view were used in the :ref:`sphx_glr_examples_scale.py`
# example.
# Since triangulation combines predictions from multiple camera views, there is
# no single confidence score for each 3D point. Here, confidence values were
# summarised as the median across the three camera views, and coordinates where
# triangulation could not be performed (NaN values) have been backfilled
# with zeros.
#
# The `.h5` file contains 4 x 50 columns, with 50 being the number of tracked
# keypoints, comprising both mouse and apparatus features, and 4 being the
# number
# of spatial dimensions + confidence values (x, y, z, confidence). Let's
# load it
# and inspect the structure.

ds = sample_data.fetch_dataset(
    "DLC_single-mouse_DBTravelator_3D.predictions.h5"
)

print(ds)
print("-----------------------------------------------------")
print(f"Individuals: {ds.individuals.values}")
print("-----------------------------------------------------")
print(f"Keypoints: {ds.keypoints.values}")

# %%
# The dataset contains 50 keypoints tracked in 3D for a single mouse,
# along with associated confidence scores. The recording was captured
# at 247 fps and spans 418 frames (~1.7 seconds).

# %%
# Visualise camera positions
# --------------------------
# To help us understand the recording geometry, let's visualise the
# camera positions relative to the travelator apparatus. The extrinsic
# parameters below were obtained from multi-camera calibration and describe
# the translation and rotation of each camera relative to a shared world
# coordinate system.

cameras_extrinsics = {
    "side": {
        "tvec": np.array([[-298.85353394], [65.67187339], [1071.78906513]]),
        "rotm": np.array(
            [
                [0.9999789, -0.00207372, 0.00615665],
                [0.00621094, 0.02727888, -0.99960857],
                [0.00190496, 0.99962571, 0.02729118],
            ]
        ),
    },
    "front": {
        "tvec": np.array([[-76.42235183], [18.56898049], [1243.26951668]]),
        "rotm": np.array(
            [
                [0.03650804, 0.99931535, -0.00600009],
                [0.00385228, -0.00614478, -0.9999737],
                [-0.99932593, 0.03648397, -0.00407397],
            ]
        ),
    },
    "overhead": {
        "tvec": np.array([[-201.40483901], [272.68542377], [2188.82953675]]),
        "rotm": np.array(
            [
                [0.9987034, 0.00421961, -0.05073173],
                [-0.00395296, -0.98712181, -0.15992155],
                [-0.0507532, 0.15991474, -0.98582523],
            ]
        ),
    },
}

# %%
# We plot these camera positions relative to a mockup of the
# dual-belt travelator.

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(projection="3d")
ax.set_proj_type("ortho")

# --- Draw travelator apparatus ---

# Travelator dimensions (mm)
w, h, d = 940.0, 100.0, 50.0  # belt width, box height, belt depth
box_dx = 100.0  # end‐platforms X extension

# Plot wireframe box
xs = [-box_dx, w + box_dx]
ys = [0, d]
zs = [0, h]
corners = np.array([[x, y, z] for z in zs for y in ys for x in xs])
edges = [
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),  # bottom
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),  # top
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]  # verticals
for i, j in edges:
    ax.plot(*corners[[i, j]].T, color="black", lw=0.5, zorder=110)


# Plot belt and end platform surfaces
def add_floor(ax, x0, x1, y_max, **kwargs):
    """Add floor surface to mock travelator"""
    verts = np.array([[x0, 0, 0], [x1, 0, 0], [x1, y_max, 0], [x0, y_max, 0]])
    ax.add_collection3d(Poly3DCollection([verts], **kwargs))


for x0 in (-box_dx, w):  # black end-platforms
    add_floor(
        ax, x0, x0 + box_dx, d, facecolors="black", edgecolors="none", zorder=1
    )
add_floor(
    ax,
    0,
    w,
    d,  # white belt
    facecolors="white",
    edgecolors="black",
    linewidths=0.5,
    zorder=2,
)

# Plot belt midline
ax.plot([w / 2] * 2, [0, d], [0, 0], color="black", lw=0.5, zorder=100)

# --- Plot camera positions and orientation axes ---
# For each camera, invert the extrinsic parameters to recover the
# camera's position in world coordinates, then draw its local axes.

shaft_length = 100.0  # length of camera orientation arrows (mm)
axis_colours = ["r", "g", "b"]

for cam, ext in cameras_extrinsics.items():
    R = ext["rotm"]  # rotation matrix (world → camera)
    t = ext["tvec"]  # translation vector (in camera coordinates)
    pos = (-R.T @ t).flatten()  # invert extrinsics to get camera
    # position in world coordinates

    # Plot camera axes
    for vec, col in zip(R, axis_colours, strict=True):
        direction = vec / np.linalg.norm(vec) * shaft_length
        ax.quiver(
            *pos, *direction, color=col, arrow_length_ratio=0.5, length=1.7
        )

    # Label camera
    ax.text(
        pos[0] - 50,
        pos[1] - 50,
        pos[2] + 100,
        s=cam,
        c="b",
        zorder=4,
        fontsize=7,
    )

#  --- Format axes ---
ax.set(
    xlim=(-200, 1400),
    ylim=(-1000, 200),
    zlim=(-500, 2000),
    xlabel="X (mm)",
    ylabel="Y (mm)",
    zlabel="Z (mm)",
)
ax.set_aspect("equal")
ax.tick_params(labelsize=7)
ax.grid(False)
ax.view_init(elev=15, azim=-40, roll=0)
fig.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.99)

plt.show()

# %%
# Select mouse keypoints
# ----------------------
# Let's inspect the 50 tracked keypoints.

print(ds.keypoints.values)

# %%
# The loaded dataset contains 43 keypoints spanning the mouse's head, back,
# tail,
# and limbs. An additional 7 keypoints mark structural landmarks on
# the travelator itself. Let's filter the dataset to keep only the mouse
# keypoints.

mouse_keywords = ["Nose", "Ear", "Back", "Tail", "paw"]
# Find the keypoints which contain the mouse keywords
mask = ds.keypoints.str.contains("|".join(mouse_keywords))
# Filter the dataset by the mouse keypoints
ds_mouse = ds.sel(keypoints=ds.keypoints[mask])

# %%
# Since the travelator landmarks are static, we can average their positions
# over time to obtain stable reference coordinates for the belt corners.
# We'll use these later to add spatial context to our plots.

belt_corner_names = ["StartPlatL", "StartPlatR", "TransitionL", "TransitionR"]
belt_corners_avg_position = ds.position.sel(
    keypoints=belt_corner_names, individuals="individual_0"
).mean(dim="time")
print(f"Belt corner positions: {belt_corners_avg_position}")

# %%
# Visualise a subset of the data
# ------------------------------
# Let's plot the 3D trajectory of a single keypoint to get a sense of the data.
# We'll use the nose of our single mouse, which shows the overall path along
# the
# travelator.

nose_position = ds_mouse.position.sel(
    keypoints="Nose", individuals="individual_0"
)
print(nose_position)


# %%
# To visualise 3D trajectories coloured by a scalar variable, we define
# a helper function adapted from `matplotlib's multicoloured line example
# <https://matplotlib.org/stable/gallery/lines_bars_and_markers
# /multicolored_line.html>`_.
# It creates a 3D figure, draws a scatter-line coloured by a scalar,
# overlays belt corner landmarks, and adds a horizontal colourbar.


def plot_coloured_trajectory_3d(
    position,
    colour_values,
    belt_corners,
    colour_label="Frame index",
    cmap="turbo",
    elev=1,
    azim=330,
    box_aspect=(6, 2, 2),
):
    """Plot a 3D keypoint trajectory coloured by a scalar.

    Overlays belt corner landmarks and adds a horizontal colourbar.

    Parameters
    ----------
    position : xarray.DataArray
        Position data for a single keypoint with dimensions (time, space).
    colour_values : array-like
        Scalar values to colour the trajectory by.
    belt_corners : xarray.DataArray
        Averaged belt corner positions.
    colour_label : str
        Label for the colourbar.
    cmap : str
        Matplotlib colourmap name.
    elev, azim : float
        Viewing angles for the 3D axes.
    box_aspect : tuple
        Aspect ratio for the 3D axes.

    Returns
    -------
    fig, ax
        Figure and axes objects.

    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Extract spatial coordinates
    x = position.sel(space="x")
    y = position.sel(space="y")
    z = position.sel(space="z")

    # Scatter points coloured by the scalar variable
    ax.scatter(x, y, z, c=colour_values, s=1, cmap=cmap)

    # Build per-segment line collection for continuous colour mapping
    x, y, z = (np.asarray(arr).ravel() for arr in (x, y, z))
    lc_kwargs = {"capstyle": "butt", "cmap": cmap}

    x_mid = np.hstack((x[:1], 0.5 * (x[1:] + x[:-1]), x[-1:]))
    y_mid = np.hstack((y[:1], 0.5 * (y[1:] + y[:-1]), y[-1:]))
    z_mid = np.hstack((z[:1], 0.5 * (z[1:] + z[:-1]), z[-1:]))

    start = np.column_stack((x_mid[:-1], y_mid[:-1], z_mid[:-1]))[:, None, :]
    mid = np.column_stack((x, y, z))[:, None, :]
    end = np.column_stack((x_mid[1:], y_mid[1:], z_mid[1:]))[:, None, :]

    segments = np.concatenate((start, mid, end), axis=1)

    lc = Line3DCollection(segments, **lc_kwargs)
    lc.set_array(colour_values)
    ax.add_collection3d(lc)

    # Overlay belt corner landmarks
    ax.scatter(
        belt_corners.sel(space="x"),
        belt_corners.sel(space="y"),
        belt_corners.sel(space="z"),
        c="k",
        s=5,
        alpha=1,
    )

    # Format axes and add colourbar
    ax.view_init(elev=elev, azim=azim, roll=0)
    ax.set_box_aspect(box_aspect)
    ax.set_xlabel("x (mm)", labelpad=20)
    ax.set_ylabel("y (mm)", labelpad=15)
    ax.set_zlabel("z (mm)", labelpad=5)
    fig.colorbar(
        ax.collections[0],
        ax=ax,
        label=colour_label,
        orientation="horizontal",
        pad=-0.05,
        fraction=0.06,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.1)

    return fig, ax


# %%
# We can use ``plot_trajectory_3d`` to plot the trajectory of the
# nose as the mouse runs along the dual-belt travelator. We also plot each of
# our fixed landmark coordinates to add further spatial information. We colour
# each point by the frame index across all recorded frames.

plot_coloured_trajectory_3d(
    nose_position,
    colour_values=nose_position.time,
    belt_corners=belt_corners_avg_position,
    colour_label="Frame index",
)
plt.show()

# %%
# Filter out points with low confidence and interpolate
# -----------------------------------------------------
# This trajectory reveals some artefacts from coordinates where
# triangulation failed, which were backfilled with zeros in this
# dataset. Let's filter out any points in our ``position`` data with
# confidence scores below ``threshold`` and interpolate to fill in
# the resulting gaps.

# Replace position values with NaN where confidence < threshold
threshold = 0.9
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

# Linearly interpolate over NaN gaps of up to 10 consecutive frames
ds_mouse.update(
    {
        "position": interpolate_over_time(
            ds_mouse.position, method="linear", max_gap=10, print_report=True
        )
    }
)

# %%
# We recalculate our (now cleaned up) nose dataset.
clean_nose_position = ds_mouse.position.sel(
    keypoints="Nose", individuals="individual_0"
)

# %%
# And plot again.
plot_coloured_trajectory_3d(
    clean_nose_position,
    colour_values=clean_nose_position.time,
    belt_corners=belt_corners_avg_position,
    colour_label="Frame index",
)
plt.show()

# %%
# The nose appears to dip vertically (in ``z``) as it progresses along the
# dual-belt
# travelator and towards the transition area (marked by keypoints
# ``TransitionL``
# and ``TransitionR``). Let's visualise how the running speed changes along the
# trajectory using the `plot_coloured_trajectory_3d` plotting function and
# :func:`movement.kinematics.compute_speed`. Since the second belt moves faster
# than the first, we might expect to see changes in speed around the
# transition.

# Compute the Euclidean speed of the nose across all three spatial dimensions
speed_nose = compute_speed(clean_nose_position)

plot_coloured_trajectory_3d(
    clean_nose_position,
    colour_values=speed_nose,
    belt_corners=belt_corners_avg_position,
    colour_label="Speed (mm/s)",
    cmap="inferno",
    azim=300,
)
plt.show()

# %%
# The colour gradient shows that the nose decelerates as the mouse
# approaches the belt transition area, marked by the ``TransitionL`` and
# ``TransitionR`` landmarks. Combined with the downward dip observed
# in the previous plot, this suggests the mouse slows and lowers its
# head in anticipation of stepping onto the faster-moving second belt.

# %%
# Visualise the skeleton
# ----------------------
# Next, we define the skeleton of our mouse body and the belt area.

mouse_skeleton = [
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
belt_skeleton = [
    ("StartPlatL", "StartPlatR"),
    ("StartPlatR", "TransitionR"),
    ("TransitionR", "TransitionL"),
    ("TransitionL", "StartPlatL"),
]

# %%
# We plot the mouse skeleton in 3D space.

mouse_frame = ds_mouse.position.isel(time=275)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for bodypart in mouse_frame.keypoints:
    x, y, z = mouse_frame.sel(keypoints=bodypart)
    ax.scatter(x, y, z, s=5, color="g")

for connection in mouse_skeleton:
    joint_1, joint_2 = connection
    x1, y1, z1 = mouse_frame.sel(keypoints=joint_1)
    x2, y2, z2 = mouse_frame.sel(keypoints=joint_2)
    ax.plot([x1, x2], [y1, y2], [z1, z2], "b-")

ax.view_init(elev=10, azim=310, roll=0)
ax.set_aspect("equal")
ax.set_xlabel("x (mm)", labelpad=25)
ax.set_ylabel("y (mm)", labelpad=5)
ax.set_zlabel("z (mm)", labelpad=2)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

plt.show()

# %%
# We can add in the belt landmark positions to visualise the full tracked
# field.
# We can see the mouse is positioned within the area of the first belt and
# is approaching the transition area between the first and second belt.

mouse_frame = ds_mouse.position.isel(time=275)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    belt_corners_avg_position.sel(space="x"),
    belt_corners_avg_position.sel(space="y"),
    belt_corners_avg_position.sel(space="z"),
    s=1,
    c="r",
)

for bodypart in mouse_frame.keypoints:
    x, y, z = mouse_frame.sel(keypoints=bodypart)
    ax.scatter(x, y, z, s=1, color="g")

for connection in belt_skeleton:
    joint_1, joint_2 = connection
    x1, y1, z1 = belt_corners_avg_position.sel(keypoints=joint_1)
    x2, y2, z2 = belt_corners_avg_position.sel(keypoints=joint_2)
    ax.plot([x1, x2], [y1, y2], [z1, z2], "y-")

for connection in mouse_skeleton:
    joint_1, joint_2 = connection
    x1, y1, z1 = mouse_frame.sel(keypoints=joint_1)
    x2, y2, z2 = mouse_frame.sel(keypoints=joint_2)
    ax.plot([x1, x2], [y1, y2], [z1, z2], "b-", linewidth=1.1)

ax.view_init(elev=40, azim=290, roll=0)
ax.set_aspect("equal")
ax.set_xlabel("x (mm)", labelpad=50)
ax.set_ylabel("y (mm)", labelpad=5)
ax.set_zlabel("z (mm)", labelpad=2)

fig.subplots_adjust(left=0.01, right=0.95, top=0.99, bottom=0.01)

plt.show()


# %%
# Now let's visualise how the skeleton evolves over time.
# We create a helper function to plot this skeleton over time.


def plot_skeleton_over_time(
    position_data,
    skeleton_edges,
    frame_range,
    c,
    ax=None,
    s=3,
    **plot_kwargs,
):
    """Plot skeleton over multiple frames.

    Parameters
    ----------
    position_data : xarray.DataArray
        Position data with dimensions (time, space, keypoints).
    skeleton_edges : list of tuples
        Skeleton connections, e.g. [('Nose', 'Back1'), ('Back1', 'Back2')].
    frame_range : range or list
        Frames to plot, e.g. range(100, 200, 5).
    c : array-like
        Colour values for each frame (must match length of frame_range).
    ax : matplotlib Axes3D, optional
        Axes to plot on. Creates new figure if None.
    s : float, optional
        Marker size. Set to 0 to hide markers. Default is 3.
    elev, azim, roll : float, optional
        Viewing angles. Defaults are 20, 300, 0.
    box_aspect : str or tuple, optional
        'equal' or (x, y, z) ratio. Default is 'equal'.
    **plot_kwargs
        Additional options: cmap, alpha, linewidth.

    Returns
    -------
    fig, ax
        Figure and axes objects.

    """
    # Set defaults
    defaults = {"cmap": "viridis", "alpha": 0.6, "linewidth": 1.5}
    defaults.update(plot_kwargs)

    pos = position_data.squeeze()

    # Map colour values to colours
    norm = plt.Normalize(np.min(c), np.max(c))
    cmap_obj = plt.get_cmap(defaults["cmap"])
    colours = cmap_obj(norm(c))

    # Plot each frame
    for i, frame_idx in enumerate(frame_range):
        pos_frame = pos.isel(time=frame_idx)

        # Draw skeleton edges
        for node1, node2 in skeleton_edges:
            try:
                xyz1 = pos_frame.sel(keypoints=node1).values
                xyz2 = pos_frame.sel(keypoints=node2).values
                if not (np.isnan(xyz1).any() or np.isnan(xyz2).any()):
                    ax.plot(
                        [xyz1[0], xyz2[0]],
                        [xyz1[1], xyz2[1]],
                        [xyz1[2], xyz2[2]],
                        color=colours[i],
                        alpha=defaults["alpha"],
                        linewidth=defaults["linewidth"],
                    )
            except (KeyError, IndexError):
                pass

        # Draw keypoints
        if s > 0:
            for kpt in pos_frame.keypoints.values:
                try:
                    xyz = pos_frame.sel(keypoints=kpt).values
                    if not np.isnan(xyz).any():
                        ax.scatter(
                            xyz[0],
                            xyz[1],
                            xyz[2],
                            c=[colours[i]],
                            s=s,
                            alpha=defaults["alpha"],
                        )
                except (KeyError, IndexError):
                    pass

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array(c)

    return sm


# %%
# We also define a simple helper function to plot the
# transition area between the two belts.


def plot_belt_transition_line(ax, belt_corners, **kwargs):
    """Plot the transition line between the two travelator belts.

    Parameters
    ----------
    ax : matplotlib Axes3D
        Axes to plot on.
    belt_corners : xarray.DataArray
        Averaged belt corner positions (must contain ``TransitionL``
        and ``TransitionR`` keypoints).
    **kwargs
        Keyword arguments passed to ``ax.plot`` (default: black solid
        line, linewidth 2).

    """
    defaults = {"color": "k", "linewidth": 2, "linestyle": "-"}
    defaults.update(kwargs)
    ax.plot(
        [
            belt_corners.sel(keypoints="TransitionL", space="x"),
            belt_corners.sel(keypoints="TransitionR", space="x"),
        ],
        [
            belt_corners.sel(keypoints="TransitionL", space="y"),
            belt_corners.sel(keypoints="TransitionR", space="y"),
        ],
        [
            belt_corners.sel(keypoints="TransitionL", space="z"),
            belt_corners.sel(keypoints="TransitionR", space="z"),
        ],
        color=defaults["color"],
        linewidth=defaults["linewidth"],
        linestyle=defaults["linestyle"],
    )


# %%
# Let's use these to plot the full mouse skeleton over a subset of
# frames from the run (``frame_range``), coloured by frame index.

frame_range = range(280, 340, 1)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

sm = plot_skeleton_over_time(
    ds_mouse.position,
    mouse_skeleton,
    frame_range=frame_range,
    c=np.array(frame_range),
    ax=ax,
    alpha=0.4,
    linewidth=1,
    s=0,
    cmap="viridis_r",
)

plot_belt_transition_line(
    ax, belt_corners_avg_position, color="k", linewidth=2, linestyle="-"
)

ax.view_init(elev=10, azim=290, roll=0)
ax.set_aspect("equal")
ax.set_xlabel("x (mm)", labelpad=40)
ax.set_ylabel("y (mm)", labelpad=5)
ax.set_zlabel("z (mm)", labelpad=5)
cb = fig.colorbar(
    sm,
    ax=ax,
    label="Frame number",
    orientation="horizontal",
    pad=-0.05,
    fraction=0.06,
    shrink=0.5,
)
ax.grid(False)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.1, top=0.99)
plt.show()

# %%
# Compute limb kinematics in 3D
# -----------------------------
# The full-body skeleton plot above is visually dense. To better understand
# how the mouse negotiates the belt transition, we can isolate a single limb
# and examine how its speed and spatial trajectory change across the gait
# cycle. Because the second belt moves faster than the first, we expect
# to see kinematic differences - particularly in stance-phase speed - before
# and after the transition.
#
# We focus on the right forelimb, defined by the chain of keypoints from
# toe to knee.

# Define the keypoints and skeleton edges for the right forelimb
limb_keypoint_names = [
    "ForepawToeR",
    "ForepawKnuckleR",
    "ForepawAnkleR",
    "ForepawKneeR",
]
ds_limb = ds_mouse.position.sel(
    keypoints=limb_keypoint_names, individuals="individual_0"
)
limb_skeleton = list(
    zip(limb_keypoint_names, limb_keypoint_names[1:], strict=False)
)

# %%
# We next compute the Euclidean speed of each limb keypoint using
# :func:`movement.kinematics.compute_speed`, which returns the frame-to-frame
# displacement magnitude across all three spatial dimensions (x, y, z).
# Taking the median across keypoints gives a single representative speed
# for the limb at each frame.

limb_speed = compute_speed(ds_limb)
avg_limb_velocity = limb_speed.median(dim="keypoints")
print(avg_limb_velocity)

# %%
# Now we plot the right forelimb skeleton over time, coloured by its
# speed at each frame. The frame range covers the last two strides on
# the first (slower) belt and a single stride on the second (faster)
# belt. A vertical black line marks the belt transition.

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
cmap = "inferno"

frame_range = range(260, 365, 1)

# Plot the right forelimb skeleton at every frame in the range
sm = plot_skeleton_over_time(
    ds_limb,
    limb_skeleton,
    frame_range=frame_range,  # 400,460,2
    c=avg_limb_velocity.isel(time=np.array(frame_range)),
    ax=ax,
    alpha=0.5,
    linewidth=1,
    s=3,
    cmap=cmap,
)

# Overlay the belt transition line for spatial reference
plot_belt_transition_line(
    ax, belt_corners_avg_position, color="k", linewidth=2, linestyle="-"
)

ax.view_init(elev=20, azim=310, roll=0)
ax.set_box_aspect((6, 1, 1))
ax.set_xlabel("x (mm)", labelpad=30)
ax.set_ylabel("y (mm)", labelpad=5)
ax.set_zlabel("z (mm)", labelpad=5)
cbR = fig.colorbar(
    sm,
    ax=ax,
    label="Speed mm/s",
    orientation="horizontal",
    pad=-0.05,
    fraction=0.06,
    shrink=0.5,
)
ax.grid(False)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.1)
plt.show()

# %%
# The limb moves fastest during early swing, when it is rapidly
# protracted to reposition for the next footfall, and slowest
# during stance, when the paw is in contact with the belt surface.
# After the transition line, stance-phase speed is noticeably higher
# than on the first belt, consistent with the faster surface speed
# of the second belt (30 cm/s vs 6 cm/s).

# %%
# We can also explore the limb kinematics by measuring the inter-limb
# distance - the Euclidean distance between the left and right forepaw
# toes - using :func:`movement.kinematics.compute_pairwise_distances`.
# In 3D, this gives us a good measure of the base of support across
# the gait cycle.

# Compute the frame-by-frame distance between the two forepaw toes
forepaw_dist = compute_pairwise_distances(
    ds_mouse.position.sel(individuals="individual_0"),
    dim="keypoints",
    pairs={"ForepawToeL": "ForepawToeR"},
)

# %%
# Let's plot the right forelimb over time again, now coloured by the
# forepaw inter-limb distance. This lets us see how the base of support
# varies spatially across the stride cycle.

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
cmap = "viridis"

frame_range = range(260, 365, 1)

# Plot the right forelimb skeleton, coloured by inter-paw distance
sm = plot_skeleton_over_time(
    ds_limb,
    limb_skeleton,
    frame_range=frame_range,
    c=forepaw_dist.isel(time=np.array(frame_range)),
    ax=ax,
    alpha=0.5,
    linewidth=1,
    s=3,
    cmap=cmap,
)

plot_belt_transition_line(
    ax, belt_corners_avg_position, color="k", linewidth=2, linestyle="-"
)

ax.view_init(elev=10, azim=270, roll=0)
ax.set_box_aspect((6, 1, 1))
ax.set_xlabel("x (mm)", labelpad=30)
ax.set_ylabel("y (mm)", labelpad=5)
ax.set_zlabel("z (mm)", labelpad=5)
cbR = fig.colorbar(
    sm,
    ax=ax,
    label="Forepaw L–R distance",
    orientation="horizontal",
    pad=-0.05,
    fraction=0.06,
    shrink=0.5,
)
ax.grid(False)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.1)
plt.show()

# %%
# The inter-limb distance peaks at the start and end of each swing
# phase, when the limbs are maximally apart, and dips mid-swing as
# the limbs pass each other moving in opposite directions. This
# pattern is consistent with a trotting gait, where diagonal limb
# pairs move in synchrony. We can verify this with a simple
# stance phase diagram, using the z-coordinate of each toe to
# classify it as in stance (on the belt surface, z < 1 mm) or in
# swing (elevated above the belt).

frame_range = range(260, 365, 1)

fig, ax = plt.subplots(figsize=(8, 2))

# Select toe keypoints in diagonal-pair order for visual clarity
toe_names = ["HindpawToeL", "ForepawToeL", "ForepawToeR", "HindpawToeR"]
ds_toes = ds_mouse.position.sel(
    keypoints=toe_names, individuals="individual_0"
).isel(time=np.array(frame_range))

# Get time values in seconds
time_vals = ds_toes.time.values

# Classify each frame as stance (z < 1 mm) or swing
stance_mask = ds_toes.sel(space="z") < 1

# For each toe, find contiguous stance regions and draw them as bars
for i, kp in enumerate(toe_names):
    stance = stance_mask.sel(keypoints=kp).values
    # Detect stance onset/offset via diff of the binary mask
    diff = np.diff(stance.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends, strict=True):
        e = min(e, len(time_vals) - 1)
        ax.barh(
            i,
            time_vals[e] - time_vals[s],
            left=time_vals[s],
            height=0.6,
            color="k",
        )

# Compute a proxy for body centre x-position
avg_body_x = ds_mouse.position.sel(space="x").mean(dim="keypoints").squeeze()

# Find the time at which the body centre crosses the belt transition
transition_x = 470
transition_time = avg_body_x.time.where(
    avg_body_x > transition_x, drop=True
).values[0]

ax.axvline(transition_time, color="k", linestyle="--", label="Transition")

ax.set_title("Stance phases per limb")
ax.set_yticks(range(len(toe_names)))
ax.set_yticklabels(toe_names)
ax.set_xlabel("Time (s)")
ax.invert_yaxis()
ax.legend(bbox_to_anchor=(0.8, 1.2), loc="upper left", borderaxespad=0)
plt.subplots_adjust(left=0.15, right=0.99, top=0.85, bottom=0.25)
plt.show()

# %%
# Indeed, we see here that the mouse is exhibiting a trotting gait,
# where the diagonal limb pairs (e.g. left hindpaw and right forepaw)
# move in synchrony. This pattern holds for the first few strides on
# the slower belt, but is disrupted around the transition point
# (approximately 1.325 s) as the mouse adjusts to the faster second
# belt.
#
# Note that this simple z-threshold approach to stance detection is not
# fully robust. Some stance periods are incorrectly fragmented where
# toe tracking dips briefly above the 1 mm threshold, perhaps due to
# tracking inaccuracies. More reliable stance detection would require a
# dedicated gait classification method.

# %%
# Compute 3D head orientation
# ---------------------------
# With 3D tracking of the nose and both ears, we can also derive all three
# rotational degrees of freedom of the head: yaw (lateral rotation),
# pitch (vertical rotation), and roll (head tilt).
#
# We define the head origin as the midpoint between the ears, with its
# z-coordinate shifted by the median vertical offset between the ear
# midpoint and nose to represent a level head position.

# Select the three head keypoints for a single individual
ear_l = ds_mouse.position.sel(keypoints="EarL", individuals="individual_0")
ear_r = ds_mouse.position.sel(keypoints="EarR", individuals="individual_0")
nose = ds_mouse.position.sel(keypoints="Nose", individuals="individual_0")

# Head origin: midpoint between the two ears
ear_mid = (ear_l + ear_r) / 2

# Shift the ear midpoint's z-coordinate by the median vertical offset
# between nose and ear midpoint, so the origin sits at "level head" height
avg_z_offset = (nose - ear_mid).sel(space="z").median()
ear_mid.loc[dict(space="z")] = ear_mid.sel(space="z") + avg_z_offset

# %%
# Now let's compute yaw, pitch and roll from the head vector
# (``ear_mid`` → ``nose``) and the ear-to-ear vector (``ear_r`` → ``ear_l``).
#
# We use :func:`numpy.arctan2` throughout to compute angles from pairs
# of vector components. For pitch and roll, we take the angle between
# the vector's z-component and its horizontal (XY) magnitude, obtained
# via :func:`movement.utils.vector.compute_norm`, to get the elevation
# relative to the horizontal plane.

# Yaw: horizontal rotation of the head, computed as the angle of the
# head vector projected onto the XY plane.
head_vec = nose - ear_mid
yaw = np.degrees(
    np.arctan2(
        head_vec.sel(space="y"),
        head_vec.sel(space="x"),
    )
)

# Pitch: vertical tilt, computed as the angle between the full 3D head
# vector and its horizontal (XY) projection.
horiz_vec = head_vec.sel(space=["x", "y"])
horiz_dist = compute_norm(horiz_vec)
pitch = np.degrees(
    np.arctan2(
        head_vec.sel(space="z"),
        horiz_dist,
    )
)

# Roll: lateral tilt, computed as the angle between the ear-to-ear
# vector and the horizontal plane.
ear_vec = ear_l - ear_r
ear_horiz_vec = ear_vec.sel(space=["x", "y"])
ear_horiz_dist = compute_norm(ear_horiz_vec)
roll = np.degrees(
    np.arctan2(
        ear_vec.sel(space="z"),
        ear_horiz_dist,
    )
)

# %%
# Let's plot all three angles over time. We display the median for each
# angle and the ``x`` position of the transition area between belts.

fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

ax = axes[0]
ax.plot(nose.sel(space="x"), yaw, color="r", linewidth=1.5)
ax.set_ylabel("Yaw (°)")
ax.axhline(
    float(yaw.median()),
    color="grey",
    linewidth=1,
    linestyle="--",
    label="Median",
)
ax.axvline(
    belt_corners_avg_position.sel(keypoints="TransitionR", space="x"),
    color="k",
    linewidth=1,
    linestyle="--",
    label="Belt transition",
)

ax = axes[1]
ax.plot(nose.sel(space="x"), pitch, color="g", linewidth=1.5)
ax.set_ylabel("Pitch (°)")
ax.axhline(float(pitch.median()), color="grey", linewidth=1, linestyle="--")
ax.axvline(
    belt_corners_avg_position.sel(keypoints="TransitionR", space="x"),
    color="k",
    linewidth=1,
    linestyle="--",
)

ax = axes[2]
ax.plot(nose.sel(space="x"), roll, color="b", linewidth=1.5)
ax.set_ylabel("Roll (°)")
ax.axhline(float(roll.median()), color="grey", linewidth=1, linestyle="--")
ax.axvline(
    belt_corners_avg_position.sel(keypoints="TransitionR", space="x"),
    color="k",
    linewidth=1,
    linestyle="--",
)
ax.set_xlabel("X (mm)")
plt.subplots_adjust(left=0.01, right=0.99, top=0.5, bottom=0.01)

fig.legend(loc="center")
fig.tight_layout()
plt.show()


# %%
# A positive yaw shows the head is angled leftward, towards the back
# wall of the travelator. A positive pitch means the head is angled
# more upward. A Positive roll shows the head is tilted leftward.
#
# Overall, we can see the roll of the head appears to tilt more
# towards the back wall but is not modulated by the belt transition.
# Comparatively, the yaw and the pitch of the head show some
# modulation by the position along the travelator. Let's visualise
# this better in 3D space.

# %%
# We can visualise the head orientation on the 3D trajectory using
# a quiver plot. We define a helper function to plot arrows
# coloured by any of the three head angles.


def plot_head_direction_quiver(
    nose_position,
    head_vector,
    colour_values,
    belt_corners,
    colour_label="Angle (°)",
    step=5,
    arrow_scale=1.0,
    cmap="coolwarm",
    elev=5,
    azim=290,
    box_aspect=(20, 2, 2),
):
    """Plot 3D nose trajectory with horizontal head direction arrows.

    Parameters
    ----------
    nose_position : xarray.DataArray
        Nose position with dimensions (time, space).
    head_vector : xarray.DataArray
        Ear-midpoint to nose vector with dimensions (time, space).
    colour_values : xarray.DataArray
        Scalar values to colour arrows by (e.g. yaw, pitch, or roll).
    belt_corners : xarray.DataArray
        Averaged belt corner positions for the transition line.
    colour_label : str
        Label for the colourbar.
    step : int
        Subsample every n-th frame for visual clarity.
    arrow_scale : float
        Length of the normalised arrows in mm.
    cmap : str
        Matplotlib colourmap name.
    elev, azim : float
        Viewing angles.
    box_aspect : tuple
        Aspect ratio for the 3D axes.

    Returns
    -------
    fig, ax
        Figure and axes objects.

    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Subsample frames for visual clarity
    t_slice = slice(None, None, step)

    nose_sub = nose_position.isel(time=t_slice)
    head_sub = head_vector.isel(time=t_slice)

    # Normalise head vectors to unit length, then scale to arrow_scale mm
    arrow_norm = compute_norm(head_sub)
    normalised = head_sub / arrow_norm * arrow_scale

    c_sub = colour_values.isel(time=t_slice)

    # Plot trajectory as a faint background line
    ax.plot(
        nose_position.sel(space="x"),
        nose_position.sel(space="y"),
        nose_position.sel(space="z"),
        color="grey",
        alpha=0.3,
        linewidth=0.5,
    )

    # Map scalar colour values to a symmetric diverging colourmap
    vmax = max(abs(float(c_sub.min())), abs(float(c_sub.max())))
    norm_c = plt.Normalize(-vmax, vmax)
    cmap_obj = plt.get_cmap(cmap)
    colours = cmap_obj(norm_c(c_sub.values))

    # Draw one quiver arrow per subsampled frame
    for i in range(len(c_sub.time)):
        ax.quiver(
            float(nose_sub.sel(space="x").isel(time=i)),
            float(nose_sub.sel(space="y").isel(time=i)),
            float(nose_sub.sel(space="z").isel(time=i)),
            float(normalised.sel(space="x").isel(time=i).values),
            float(normalised.sel(space="y").isel(time=i).values),
            float(normalised.sel(space="z").isel(time=i).values),
            color=colours[i],
            arrow_length_ratio=0.3,
            linewidth=1.0,
        )

    plot_belt_transition_line(
        ax, belt_corners, color="k", linewidth=2, linestyle="-"
    )

    ax.view_init(elev=elev, azim=azim, roll=0)
    ax.set_box_aspect(box_aspect)
    ax.set_xlabel("x (mm)", labelpad=40)
    ax.set_ylabel("y (mm)", labelpad=10)
    ax.set_zlabel("z (mm)", labelpad=5)
    ax.grid(False)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm_c)
    sm.set_array(c_sub)
    fig.colorbar(
        sm,
        ax=ax,
        label=colour_label,
        orientation="horizontal",
        pad=-0.05,
        fraction=0.06,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.1)

    return fig, ax


# %%
# Now we can visualise the nose trajectory coloured by each angle in turn.
# We downsample every 20th frame to aid interpretability. Let's start with
# plotting pitch.

plot_head_direction_quiver(
    nose,
    head_vec,
    pitch,
    belt_corners_avg_position,
    step=20,
    arrow_scale=25,
    colour_label="Pitch (°, negative = more dipped)",
    elev=10,
    azim=270,
)
plt.show()

# %%
# Here we see that the head is angled toward the belt
# surface at the start of the run, pitches upward on approach to the
# belt transition, and then dips again after stepping onto the second belt.
# This may reflect the mouse lowering its centre of mass as it adjusts
# to each moving belt surface.

# %%
# Next, let's plot the yaw of the head along the nose trajectory.

plot_head_direction_quiver(
    nose,
    head_vec,
    yaw,
    belt_corners_avg_position,
    step=20,
    arrow_scale=25,
    colour_label="Yaw (°, positive = more left-ward)",
    elev=80,
    azim=270,
    box_aspect=(6, 2, 2),
)
plt.show()

# %%
# The head turns rightward at the start of the run but is oriented
# leftward (toward the back wall) for the remainder, with a brief
# increase in leftward yaw during the belt transition. This may
# reflect a whole-body twist as the mouse steps onto the faster belt.
