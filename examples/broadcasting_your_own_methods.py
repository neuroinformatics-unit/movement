"""Extend your analysis methods along data dimensions
=====================================================

Learn how to use the `make_broadcastable` decorator, to easily write functions
that can then be cast across an entire ``DataArray``.
"""

# %%
# Imports
# -------

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following lines in your notebook
# %matplotlib widget

# We will need numpy and xarray to make our custom data for this example,
# and matplotlib to show what it contains.
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# We will be using the the movement.utils.broadcasting module to
# turn our one-dimensional functions into functions that work across
# entire ``DataArray`` objects.
from movement.utils.broadcasting import (
    broadcastable_method,
    make_broadcastable,
)

# %%
# Create Sample Dataset
# ---------------------
# We will need some example data to work with. To make it clear what we are
# building, we will create a custom ``DataArray`` that contains the positions
# of three individuals, at 5 different times.

# First, define some keypoints
centroid_keypoint = np.array(
    [
        [[0.0, 0.0, 7.5], [0.0, 10.0, 8.5]],
        [[2.0, 2.0, 6.25], [0.5, 7.5, 7.25]],
        [[4.0, 4.0, 5.0], [1.25, 5.0, 5.0]],
        [[6.0, 2.0, 3.75], [2.25, 7.5, 4.25]],
        [[8.0, 1.5, 2.5], [4.0, 5.0, 2.0]],
        [[10.0, 0.5, 1.25], [6.5, 7.5, 1.0]],
    ],
    dtype=float,
)
left_keypoint = centroid_keypoint.copy()
left_keypoint[:, 0, :] -= 0.5
right_keypoint = centroid_keypoint.copy()
right_keypoint[:, 0, :] += 0.5

# Create a sample DataArray just for the centroid keypoint
da_centroid = xr.DataArray(
    data=centroid_keypoint,
    dims=["time", "space", "individuals"],
    coords={"space": ["x", "y"], "individuals": ["Alfie", "Bravo", "Charlie"]},
)
# And create another that holds all 3 keypoints
da = xr.DataArray(
    data=np.stack([left_keypoint, centroid_keypoint, right_keypoint], axis=-1),
    dims=["time", "space", "individuals", "keypoints"],
    coords={
        "space": ["x", "y"],
        "individuals": ["Alfie", "Bravo", "Charlie"],
        "keypoints": ["left", "centre", "right"],
    },
)

# We can visualise the paths that our individuals are taking in the plot
# below.
fig, ax = plt.subplots()
colours = ["red", "green", "blue"]
markers = ["o", "x", "."]
for name_index, name in enumerate(da_centroid.individuals.values):
    name_colour = colours[name_index]
    name_marker = markers[name_index]
    x_data, y_data = (
        da_centroid.sel(space="x", individuals=name),
        da_centroid.sel(space="y", individuals=name),
    )
    sc = ax.scatter(
        x_data, y_data, label=name, c=name_colour, marker=name_marker
    )
    for i in range(x_data.shape[0] - 1):
        ax.annotate(
            "",
            xy=(x_data[i + 1], y_data[i + 1]),
            xytext=(x_data[i], y_data[i]),
            arrowprops={"arrowstyle": "->", "color": name_colour},
            size=15,
        )
ax.legend(loc="upper right")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Centroid keypoint paths")
fig.show()

# %%
# Motivation
# ----------
# Suppose that, during our experiment, we had a camera with a narrow field of
# view positioned at (0,0), that the individuals were aware of. We want to know
# at which time points the individuals were in the field of view of this
# camera.
# Given a single pair of (x, y) coordinates, and the camera parameters, we
# have written a function to determine whether (x, y) was within the
# camera's field of view as follows:


def in_camera_fov(xy_position, fov_angle, direction_camera_faces):
    """Determine if a position is in view of a camera."""
    if fov_angle >= 2 * np.pi:
        return True

    # Find the angle in [0, 2\pi] (or [0, 360] degrees) between the point
    # and the camera
    angle_to_point = np.arctan2(xy_position[1], xy_position[0])
    if angle_to_point < 0:
        angle_to_point += 2 * np.pi
    # The angle the camera is facing in [0, 2\pi] (or [0, 360] degrees)
    camera_facing_angle = np.arctan2(
        direction_camera_faces[1], direction_camera_faces[0]
    )
    if camera_facing_angle < 0:
        camera_facing_angle += 2 * np.pi

    # What are the limits of the camera angle?
    max_fov_angle = camera_facing_angle + np.abs(fov_angle) / 2.0
    min_fov_angle = camera_facing_angle - np.abs(fov_angle) / 2.0
    # The camera can see the point, if it is at an angle between its max and
    # min field of view angles.
    while max_fov_angle > 2 * np.pi:
        # The max camera angle includes a full rotation.
        # Subtract one full rotation from it.
        max_fov_angle -= 2 * np.pi
    while min_fov_angle < 0.0:
        # The min camera angle includes a full (anti-)rotation.
        # Add a rotation to it.
        min_fov_angle += 2 * np.pi
    # Both fov angles are now in the correct range, however the max and min
    # might have swapped due to previous adjustments.
    if max_fov_angle < min_fov_angle:
        # We are "crossing the 0-angle" line
        return (
            angle_to_point <= max_fov_angle or angle_to_point >= min_fov_angle
        )
    else:
        return min_fov_angle <= angle_to_point <= max_fov_angle


# If our camera faces along the (0,0) -> (1,1) line, with a 45-degree
# (np.pi/4 radian) field of view, it should see everything in the
# positive x, positive y quadrant and nothing else.
degrees_45 = np.pi / 4.0
camera_direction = (1.0, 1.0)
print(in_camera_fov((1, 1), degrees_45, camera_direction))
print(in_camera_fov((-1, 1), degrees_45, camera_direction))
print(in_camera_fov((1, -1), degrees_45, camera_direction))
print(in_camera_fov((-1, -1), degrees_45, camera_direction))

# %%
# We can visualise the relation between our camera's field of view and our
# custom dataset in a figure.

illustration_x = np.linspace(0.0, 10.0, num=21, endpoint=True)
forward_ray = illustration_x * np.tan(degrees_45)
min_fov_ray = illustration_x * np.tan(degrees_45 / 2.0)
max_fov_ray = illustration_x * np.tan(3.0 * degrees_45 / 2.0)

camera_fig, camera_ax = plt.subplots()
camera_ax.plot(
    illustration_x,
    forward_ray,
    "--",
    color="black",
    marker=None,
    alpha=0.75,
    label="Camera faces this direction",
)
camera_ax.plot(
    illustration_x,
    max_fov_ray,
    "--",
    color="blue",
    alpha=0.5,
    label="Max FoV",
)
camera_ax.plot(
    illustration_x,
    min_fov_ray,
    "--",
    color="cyan",
    label="Min FoV",
)

camera_ax.fill_between(
    illustration_x, min_fov_ray, max_fov_ray, color="black", alpha=0.25
)
camera_ax.set_title("The camera can see everything in the shaded region")


# Overlay our data from above
for name_index, name in enumerate(da_centroid.individuals.values):
    name_colour = colours[name_index]
    name_marker = markers[name_index]
    x_data, y_data = (
        da_centroid.sel(space="x", individuals=name),
        da_centroid.sel(space="y", individuals=name),
    )
    sc = camera_ax.scatter(
        x_data, y_data, label=name, c=name_colour, marker=name_marker
    )

# Retain aspect ratio of visual effect
camera_ax.set_ylim(illustration_x.min(), illustration_x.max())
camera_ax.set_xlim(illustration_x.min(), illustration_x.max())
camera_ax.legend(loc="upper right")
camera_ax.set_xlabel("x")
camera_ax.set_ylabel("y")

camera_fig.show()

# %%
# Determine if each position can be seen
# --------------------------------------
# Given our data, we could extract whether each position (for each timepoint,
# and each individual) was in view of the camera by looping over the values.

data_shape = da_centroid.shape
in_view = np.zeros(
    shape=(len(da_centroid["time"]), len(da_centroid["individuals"])),
    dtype=bool,
)  # We would save one result per timepoint, per individual

for time_index in range(len(da_centroid["time"])):
    print(f"At timepoint {time_index}:")
    for individual_index in range(len(da_centroid["individuals"])):
        was_in_view = in_camera_fov(
            da_centroid.isel(time=time_index, individuals=individual_index),
            degrees_45,
            camera_direction,
        )
        was_seen = "was in view" if was_in_view else "was not in view"
        print(
            "\tIndividual "
            f"{da_centroid['individuals'].values[individual_index]} "
            f"{was_seen}"
        )
        in_view[time_index, individual_index] = was_in_view

# %%
# We could then build a new DataArray to store our results, so that we can
# access the results in the same way that we did our original data
was_in_view_da = xr.DataArray(
    in_view,
    dims=["time", "individuals"],
    coords={"individuals": ["Alfie", "Bravo", "Charlie"]},
)

print(
    "Boolean DataArray indicating if at a given time, "
    "a given individual was in view of the camera:"
)
print(was_in_view_da)

# %%
# Data Generalisation Issues
# --------------------------
# The shape our the resulting DataArray is the same as our original DataArray,
# but without the space dimension. Indeed, we have essentially collapsed the
# space dimension, since our ``in_camera_fov`` function takes in a 1D data
# slice (the position) and returns a scalar value.
# However, the fact that we have to construct a new DataArray after running our
# function over all space slices in our DataArray is not scalable - our ``for``
# loop approach relied on knowing how many dimensions our data had (and the
# size of those dimensions). We don't have a guarantee that future DataArrays
# we are given will have the same structure.

# %%
# Making our Function Broadcastable
# ---------------------------------
# To combat this problem, we can make the observation that given any DataArray,
# we always want to broadcast our ``in_camera_fov`` function along the
# ``"space"`` dimension. As such, we can decorate our function with the
# ``make_broadcastable`` decorator:


@make_broadcastable()
def in_camera_fov_broadcastable(
    xy_position, fov_angle, direction_camera_faces
) -> float:
    return in_camera_fov(
        xy_position=xy_position,
        fov_angle=fov_angle,
        direction_camera_faces=direction_camera_faces,
    )


# %%
# Note that when writing your own methods, there is no need to have both
# ``in_camera_fov`` and ``_in_camera_fov_broadcastable``, simply apply the
# ``make_broadcastable`` decorator to ``in_camera_fov`` directly. We've made
# two separate functions here to illustrate what's going on.
#
# ``in_camera_fov_broadcastable`` is usable in exactly the same ways as
# ``in_camera_fov`` was:

print(in_camera_fov_broadcastable((1, 1), degrees_45, camera_direction))
print(in_camera_fov_broadcastable((-1, 1), degrees_45, camera_direction))
print(in_camera_fov_broadcastable((1, -1), degrees_45, camera_direction))
print(in_camera_fov_broadcastable((-1, -1), degrees_45, camera_direction))

# %%
# However, ``in_camera_fov_broadcastable`` also takes a DataArray as the first
# (``xy_position``) argument, and an extra keyword argument
# "``broadcast_dimension``"". These arguments let us broadcast across the given
# dimension of the input DataArray, treating each 1D-slice as a separate input
# to ``in_camera_fov``.

was_in_view_da_broadcasting = in_camera_fov_broadcastable(
    da_centroid,  # Now a DataArray input
    fov_angle=degrees_45,
    direction_camera_faces=camera_direction,
    broadcast_dimension="space",
)

print("DataArray output using broadcasting:")
print(was_in_view_da_broadcasting)

# %%
# Calling ``in_camera_fov_broadcastable`` in this way gives us a DataArray
# output - and one that retains any information that was in our original
# DataArray to boot!
# The result is exactly the same as what we got from using our ``for`` loop,
# and then adding the extra information to the result.

xr.testing.assert_equal(was_in_view_da, was_in_view_da_broadcasting)

# %%
# But importantly, ``in_camera_fov_broadcastable`` also works on DataArrays
# with different dimensions:

keypoints_in_camera_view = in_camera_fov_broadcastable(
    da,
    fov_angle=degrees_45,
    direction_camera_faces=camera_direction,
    broadcast_dimension="space",
)

print(
    "We get a 3D output from our 4D data, "
    "again with the dimension that we broadcast along collapsed:"
)
print(keypoints_in_camera_view)

# %%
# Only Broadcast Along Select Dimensions
# --------------------------------------
# The ``make_broadcastable`` decorator has some flexibility with its input
# arguments, to help you avoid unintentional behaviour. You may have noticed,
# for example, that there is nothing stopping someone who wants to use your
# analysis code from trying to broadcast along the wrong dimension.

silly_broadcast = in_camera_fov_broadcastable(
    da_centroid, degrees_45, camera_direction, broadcast_dimension="time"
)

print("The output has collapsed the time dimension:")
print(silly_broadcast)

# %%
# There is no error thrown because functionally, this is a valid operation.
# The time slices of our data were 1D, so we can run ``in_camera_fov`` on them.
# But each slice isn't a position, it's an array of one spatial coordinate
# (EG x) for each individual, at every time! So from an analysis standpoint,
# doing this doesn't make sense and isn't how we intend our function to be
# used.

# We can pass the ``only_broadcastable_along`` keyword argument to
# ``make_broadcastable`` to prevent these kinds of mistakes, and make our
# intentions clearer.


@make_broadcastable(only_broadcastable_along="space")
def in_camera_fov_space_only(xy_position, fov_angle, direction_camera_faces):
    return in_camera_fov(xy_position, fov_angle, direction_camera_faces)


# %%
# Now, ``in_camera_fov_space_only`` no longer takes the
# ``broadcast_dimension`` argument.

try:
    in_camera_fov_space_only(
        da_centroid,
        fov_angle=degrees_45,
        direction_camera_faces=camera_direction,
        broadcast_dimension="space",
    )
except TypeError as e:
    print(f"Got a TypeError when trying to run, here's the message:\n{e}")

# %%
# The error we get seems to be telling us that we've tried to set the value of
# ``broadcast_dimension`` twice. Specifying
# ``only_broadcastable_along = "space"`` forces ``broadcast_dimension`` to be
# set to ``"space"``, so trying to set it again (even to to the same value)
# results in an error.
# However, ``in_camera_fov_space_only`` knows to only use the ``"space"``
# dimension of the input by default.

was_in_view_space_only = in_camera_fov_space_only(
    da_centroid,
    fov_angle=degrees_45,
    direction_camera_faces=camera_direction,
)

xr.testing.assert_equal(was_in_view_da_broadcasting, was_in_view_space_only)

# %%
# Extending to Classmethods
# -------------------------
# ``make_broadcastable`` can also be applied to class methods, though it needs
# to be told that you are doing so via the ``is_classmethod`` parameter.


class Camera:
    """Represents an observing camera in the experiment."""

    fov: float
    facing_direction: tuple[float, float]

    def __init__(self, fov=np.pi / 4.0, facing_direction=(1.0, 1.0)):
        """Create a new instance."""
        self.fov = fov
        self.facing_direction = facing_direction

    @make_broadcastable(is_classmethod=True, only_broadcastable_along="space")
    def is_in_view(self, xy_position) -> bool:
        """Whether the camera can see the position."""
        # For the sake of brevity, we won't redefine the entire method here,
        # and will just call our existing function.
        return in_camera_fov(
            xy_position,
            fov_angle=self.fov,
            direction_camera_faces=self.facing_direction,
        )


camera = Camera(degrees_45, camera_direction)
was_in_view_clsmethod = camera.is_in_view(da_centroid)

xr.testing.assert_equal(was_in_view_clsmethod, was_in_view_da_broadcasting)

# %%
# The ``broadcastable_method`` decorator is provided as a helpful alias for
# ``make_broadcastable(is_classmethod=True)``, and otherwise works in the same
# way (and accepts the same parameters).


class CameraAlternative:
    """Represents an observing camera in the experiment."""

    fov: float
    facing_direction: tuple[float, float]

    def __init__(self, fov=np.pi / 4.0, facing_direction=(1.0, 1.0)):
        """Create a new instance."""
        self.fov = fov
        self.facing_direction = facing_direction

    @broadcastable_method(only_broadcastable_along="space")
    def is_in_view(self, xy_position) -> bool:
        """Whether the camera can see the position."""
        # For the sake of brevity, we won't redefine the entire method here,
        # and will just call our existing function.
        return in_camera_fov(
            xy_position,
            fov_angle=self.fov,
            direction_camera_faces=self.facing_direction,
        )


alternative_camera = Camera(degrees_45, camera_direction)
was_in_view_clsmethod_alternative = camera.is_in_view(da_centroid)

xr.testing.assert_equal(
    was_in_view_clsmethod, was_in_view_clsmethod_alternative
)

# %%
