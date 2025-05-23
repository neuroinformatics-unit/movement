# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation as R

from movement import sample_data
from movement.io import load_poses
from movement.utils.vector import cart2pol, convert_to_unit, pol2cart

# For interactive images; requires `pip install ipympl`
# %matplotlib widget


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Import sample data
# one individual, 6 keypoints

ds_path = sample_data.fetch_dataset_paths(
    "SLEAP_single-mouse_EPM.analysis.h5"
)["poses"]
ds = load_poses.from_sleap_file(ds_path, fps=None)  # force time_unit = frames


print(ds)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define anterior and posterior keypoints

anterior_keypoints = ["snout", "left_ear", "right_ear"]
posterior_keypoints = ["centre", "tail_base"]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute centroids

# get position data array
position = ds.position

# Compute centroid per individual
centroid = position.mean(dim="keypoints")  # v

# Compute centroid for anterior and posterior keypoints
centroid_anterior = position.sel(keypoints=anterior_keypoints).mean(
    dim="keypoints", skipna=True
)
centroid_posterior = position.sel(keypoints=posterior_keypoints).mean(
    dim="keypoints", skipna=True
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute posterior2anterior vector per individual

# Compute vector from posterior to anterior centroid
posterior2anterior = centroid_anterior - centroid_posterior

# Compute polar angle of posterior2anterior vector
# the angle theta is positive going from the positive x-axis to the positive y-axis
posterior2anterior_pol = cart2pol(posterior2anterior)
# theta = posterior2anterior_pol.sel(space_pol="phi")  # h


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute coordinates in ego-centric coordinate system

# Compute position in image coord system translated
position_translated = position - centroid  # Y_centered
position_translated_pol = cart2pol(position_translated)  # Y_centered_pol


# Compute position in ego-centric coordinate system
# for every frame, x-axis == anterior-posterior vector
position_egocentric_pol = position_translated_pol.copy()

# rho is the same as in the translated image coordinate system
# phi angle is measured relative to the phi angle of the posterior2anterior vector
position_egocentric_pol.loc[{"space_pol": "phi"}] = (
    position_translated_pol.sel(space_pol="phi")
    - posterior2anterior_pol.sel(
        space_pol="phi"
    )  # angle_relative_to_posterior2anterior
)

# Convert rotated position coordinates to cartesian
position_egocentric = pol2cart(position_egocentric_pol)

# Create a dataset with the `position` data array holding the egocentric coordinates
# of the keypoints
ds_egocentric = ds.copy()
ds_egocentric["position"] = (
    position_egocentric  # keypoint positions in egocentric coord system
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check by plotting the keypoint trajectories in the egocentric coordinate system

fig, ax = plt.subplots(1, 1)
for kpt in ds_egocentric.coords["keypoints"].data:
    ax.scatter(
        x=ds_egocentric.position.sel(keypoints=kpt, space="x"),
        y=ds_egocentric.position.sel(keypoints=kpt, space="y"),
        label=kpt,
        alpha=0.5,
    )
    # add axis of egocentric coordinate system
    ax.quiver(
        [0],  # x
        [0],  # y
        [100],  # u
        [0],  # v
        color="r",
        angles="xy",
        scale=1,
        scale_units="xy",
    )
    ax.quiver(
        [0],  # x
        [0],  # y
        [0],  # u
        [100],  # v
        color="g",
        angles="xy",
        scale=1,
        scale_units="xy",
    )


ax.legend()
ax.invert_yaxis()
ax.axis("equal")
ax.set_xlim(-200, 200)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check that posterior2anterior vector in the egocentric coordinate system
# is parallel to the x-axis

# Compute centroid for anterior and posterior keypoints
# in egocentric coordinate system
centroid_anterior_rotated = position_egocentric.sel(
    keypoints=anterior_keypoints
).mean(dim="keypoints", skipna=True)
centroid_posterior_rotated = position_egocentric.sel(
    keypoints=posterior_keypoints
).mean(dim="keypoints", skipna=True)

# Compute posterior2anterior vector in egocentric coordinate system
posterior2anterior_rotated = (
    centroid_anterior_rotated - centroid_posterior_rotated
)

# Check that the y-component of the vector is close to zero
print(np.nanmax(posterior2anterior_rotated.sel(space="y").data))
print(np.nanmin(posterior2anterior_rotated.sel(space="y").data))

# Check that the unit posterior2anterior vector is parallel to the x-axis
posterior2anterior_rotated_unit = convert_to_unit(posterior2anterior_rotated)
posterior2anterior_rotated_unit.plot.line(x="time", row="individuals")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compare to alternative approach using scipy Rotation objects

# expand posterior2anterior data array to 3d space
# ideally, reference vector defined everywhere -- interpolate with slerp?
posterior2anterior_3d = posterior2anterior.pad(
    {"space": (0, 1)}, constant_values={"space": (0)}
)
posterior2anterior_3d.coords["space"] = ["x", "y", "z"]


# compute array of rotation matrices from ICS to ECS
def compute_rotation_to_align_x_axis(vec):
    if np.isnan(vec).any():
        return R.from_matrix(np.eye(3))  # ---> identity, maybe not the best?
    else:
        rrot, rssd = R.align_vectors(
            np.array([[1, 0, 0]]), vec, return_sensitivity=False
        )
        return rrot


rotation2egocentric = xr.apply_ufunc(
    lambda v: compute_rotation_to_align_x_axis(v),
    posterior2anterior_3d,
    input_core_dims=[["space"]],
    vectorize=True,
)

# add to dataset
ds["rotation2egocentric"] = rotation2egocentric

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute rotated keypoints

# expand position data array to 3d space
position_3d = position.pad({"space": (0, 1)}, constant_values={"space": (0)})
position_3d.coords["space"] = ["x", "y", "z"]

# compute translation from ICS to ECS
centroid_3d = position_3d.mean(dim="keypoints")

# compute keypoints in ECS (translated and rotated)
position_ego_3d = xr.apply_ufunc(
    lambda rot, trans, vec: rot.apply(vec - trans),
    rotation2egocentric,  # rot
    centroid_3d,  # trans
    position_3d,  # vec
    input_core_dims=[[], ["space"], ["space"]],
    output_core_dims=[["space"]],
    vectorize=True,
)

# compare to other approach
print(position_3d.sel(keypoints="snout").data[-3:, :, :])  # ICS
print("-----")
print(position_ego_3d.sel(keypoints="snout").data[-3:, :, :])  # ECS
print("-----")
print(ds_egocentric.position.sel(keypoints="snout").data[-3:, :, :])  # ECS-2D

# %%
