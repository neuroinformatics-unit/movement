"""
Compute kinematics
============================

Compute velocity and acceleration on an example dataset of pose tracks.
"""

# %%
# Imports
# -------
# Install circle_fit in your virtual environment with `pip install circle-fit`
from matplotlib import pyplot as plt
import numpy as np

from movement import sample_data
from movement.io import load_poses
import movement.analysis.kinematics as kin
%matplotlib widget

from circle_fit import taubinSVD, plot_data_circle

# %%
# Fetch an example dataset
# ------------------------
# Print a list of available datasets:

for file_name in sample_data.list_sample_data():
    print(file_name)

# %%
# Fetch the path to an example dataset.
# In this case, we select the SLEAP_three-mice_Aeon_proofread sample data.
# Feel free to replace this with the path to your own dataset.
# e.g., ``file_path = "/path/to/my/data.h5"``)
file_path = sample_data.fetch_sample_data_path(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5"
)

# %%
# Load and inspect the dataset
# ---------------------------------
# The data was acquired at 50fps.
# - If we pass this info to the data loading method, the time dimension will be expressed in seconds.
# - If we don't, the time dimension will be in units of frames (starting at frame 0)
ds = load_poses.from_sleap_file(file_path, fps=50)
print(ds)

# We can see in the printed dataset description that there are three individuals in this dataset,
# of which only one keypoint called 'centroid' is tracked in x and y.

# %%
# The loaded dataset `ds` contains two data variables:
# ``pose_tracks`` and ``confidence``.
# To compute velocity and acceleration, we will need the pose tracks:
pose_tracks = ds.pose_tracks


# %%
# Visualise the trajectories
# -----------------------
# First, let's visualise the mice trajectories in the XY plane.
# We can colour the data by individual:

fig, ax = plt.subplots(1, 1)
for mouse_name, col in zip(pose_tracks.individuals.values, ['r','g','b']):
    ax.plot(
        pose_tracks.sel(individuals=mouse_name, space="x"),
        pose_tracks.sel(individuals=mouse_name, space="y"),
        linestyle='-',
        marker='.',
        markersize=2,
        linewidth=.5,
        c=col,
        label=mouse_name,
    )
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.axis("equal")
    ax.legend()

# We can see that the trajectory of the three mice approximates the arc of a circle.
# Notice that the x and y axes are set to equal scales.


# %%
# We can compute the centre and the radius of a circle their best approximates 
# their trajectories.

xy_coords_all_mice = np.vstack(
    [
        pose_tracks.sel(space=['x','y']).squeeze()[:,i,:]
        for i, _ in enumerate(pose_tracks.individuals.values)
    ]
)

xc, yc, rc, rmse = taubinSVD(xy_coords_all_mice)
plot_data_circle(xy_coords_all_mice, xc, yc, rc)

# The aggregated data fits well a circle of radius 528.6 pixels centred at (711.11, 540.53) pixels.
# The root mean square distance between the data points and the circle is 2.71 pixels.


# %%
# If we now colour by time:
fig, axes = plt.subplots(3, 1)
for mouse_name, ax in zip(pose_tracks.individuals.values, axes):
    sc=ax.scatter(
        pose_tracks.sel(individuals=mouse_name, space="x"),
        pose_tracks.sel(individuals=mouse_name, space="y"),
        s=2,
        c=pose_tracks.time,
        cmap="viridis",
    )

    ax.set_title(mouse_name)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.axis("equal")
    fig.colorbar(sc, ax=ax, label='time (s)')
fig.tight_layout()

# We can see that two of the mice (AEON3B_NTP and AEON3B_TP1) started moving from the 
# quadrant of positive x and y coordinates, towards the quadrant
# with negative x and y coordinates. The third mouse (AEON3B_TP2) moved in the opposite direction.

# %%
# Compute displacement
# ---------------------
# We can start of by computing the distance travelled by the mice along their trajectories.
# For this, we can use the `compute_displacement` method. This will give us for each timestep t,
# the vector from the position at time t-1 to the corresponding position at time t.
pose_tracks_displacement = kin.compute_displacement(pose_tracks)

# Notice that the shape is changed! 

# %%
# We can verify this is the case with a quiver plot:
mouse_name = 'AEON3B_TP2'
fig = plt.figure()
ax = fig.add_subplot()
sc=ax.scatter(
    pose_tracks.sel(individuals=mouse_name, space="x"),
    pose_tracks.sel(individuals=mouse_name, space="y"),
    s=15,
    c=pose_tracks.time,
    cmap="viridis",
)
ax.quiver(
    pose_tracks.sel(individuals=mouse_name, space="x"), # origin of each vector
    pose_tracks.sel(individuals=mouse_name, space="y"),
    -pose_tracks_displacement.sel(individuals=mouse_name, space="x"), # tip of each vector
    -pose_tracks_displacement.sel(individuals=mouse_name, space="y"),
    angles='xy',
    scale=1,
    scale_units='xy'
)
ax.axis('equal')
ax.set_xlim(200,600)
ax.set_ylim(700,1100)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_title(f'Zoomed in trajectory of {mouse_name}')
fig.colorbar(sc, ax=ax, label='time (s)')

# Notice that we invert the sign of the displacement vector in the plot for easier visual check: 
# the displacement vector is the vector defined at time t, that goes from the 
# previous position t-1 to the current position at t. Therefore the opposite vector will point from the position at t, to the position at t-1
# This is what we see represented in the plot
# %%
# With the displacement data we can compute the distance covered by the mouse along the curve:
displacement_vectors_lengths = np.linalg.norm(
    pose_tracks_displacement.sel(individuals=mouse_name, space=["x","y"]).squeeze(),
    axis=1
)

total_displacement = np.sum(displacement_vectors_lengths, axis=0) # pixels

print(f"The mouse {mouse_name}'s trajectory is {total_displacement:.2f} pixels long")

# %%
# We can verify that this result makes sense using our circle fit.

# We first compute the vectors from the estimated centre of the circle, to the initial and final position of the mouse
ini_pos_rel_to_centre = pose_tracks.sel(individuals=mouse_name, space=["x","y"]).values[0,:] - [xc,yc]
end_pos_rel_to_centre = pose_tracks.sel(individuals=mouse_name, space=["x","y"]).values[-1,:] - [xc,yc]

# %%
# We divide this vectors by their norm (length) to make them unit vectors
ini_pos_rel_to_centre_unit = ini_pos_rel_to_centre/np.linalg.norm(ini_pos_rel_to_centre)
end_pos_rel_to_centre_unit = end_pos_rel_to_centre/np.linalg.norm(end_pos_rel_to_centre)

# %%
# The angle between these two vectors in radians times the radius of the circle is the length of the circle arc
theta_rad = np.arccos(np.dot(ini_pos_rel_to_centre_unit,end_pos_rel_to_centre_unit.T)).item()
arc_circle_length = rc*theta_rad

print(
    f"The mouse {mouse_name}'s trajectory is {total_displacement:.2f} pixels long. "
    f"It moved approximately {theta_rad*180/np.pi:.2f} degrees around a circle. The length of the best-fit "
    f"arc circle is {arc_circle_length:.2f} pixels."
)

# Notice that the mouse doesn't move in a straight line, and sometimes back tracks, so the measured displacement
# is larger than the reference one.

# %%
# Compute velocity
# ----------------
# We can easily compute the velocity vectors for all individuals in our data array:
pose_tracks_velocity = kin.compute_velocity(pose_tracks)

# We can plot the components of the velocity vector against time
# using ``xarray``'s built-in plotting methods:

# da.plot.line(x="time", row="space", aspect=2, size=2.5)
# da = pose_tracks.sel(keypoints="centroid")
# da.plot.line(x="time", row="individuals", aspect=2, size=2.5)



# As well as the norm of the velocity vector (speed):


# Quiver plot

# %%
# Compute displacement between consecutive positions
# ----------------------------------------------------
pose_tracks_displacement = kin.compute_displacement(pose_tracks)

# %%
# Compute acceleration
pose_tracks_accel = kin.compute_acceleration(pose_tracks)



