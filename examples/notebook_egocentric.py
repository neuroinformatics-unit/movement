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
# Import data

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
# (we dont use any of the other data arrays in the dataset)
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
# add centroid and heading angle of the posterior2anterior vector in the
# image coordinate system (ICS)
ds_egocentric["centroid_ics"] = centroid
ds_egocentric["heading_angle_ics"] = posterior2anterior_pol.sel(
    space_pol="phi"
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check by plotting keypoint trajectories in egocentric coordinate system

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
# Check if position rotated matches the result by rotations

# extend posterior2anterior vector to 3D
posterior2anterior_3d = np.pad(
    posterior2anterior.data.squeeze(), ((0, 0), (0, 1))
)

# compute rotations to align posterior2anterior vector to x-axis of ECS
# ideally, if nan return nan?
list_rotations = []
list_rssd = []
for vec in posterior2anterior_3d:
    # add nan to list if no vector defined
    if np.isnan(vec).any():
        # list_rotations.append(np.nan)
        # list_rssd.append(np.nan)
        list_rotations.append(R.from_matrix(np.zeros((3, 3))))
        continue

    # else compute rotation to x-axis
    rrot, rssd = R.align_vectors(
        np.array([[1, 0, 0]]), vec, return_sensitivity=False
    )
    list_rotations.append(rrot)
    list_rssd.append(rssd)

#

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Add rotation matrices to dataset

# expand posterior2anterior data array to 3d space
# ideally, reference vector defined everywhere --- slerp?
posterior2anterior_3d = posterior2anterior.pad(
    {"space": (0, 1)}, constant_values={"space": (0)}
)
posterior2anterior_3d.coords["space"] = ["x", "y", "z"]


# compute array of rotation matrices from ICS to ECS
# ideally, reference vector defined everywhere
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
    lambda rot, trans, vec: rot.apply(vec - trans, inverse=False),
    rotation2egocentric,  # rot
    centroid_3d,  # trans
    position_3d,  # vec
    input_core_dims=[[], ["space"], ["space"]],
    output_core_dims=[["space"]],
    vectorize=True,
)

# compare to other approach
print(position_3d.sel(keypoints="snout").data)
print("-----")
print(position_ego_3d.sel(keypoints="snout").data)
print("-----")
print(ds_egocentric.position.sel(keypoints="snout").data)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%
# add rotation matrices from ICS to ECS to dataset
req_shape = tuple(ds.sizes[d] for d in ["time", "individuals"])
ds["rotation_matrices"] = xr.DataArray(
    np.array(list_rotations).reshape(req_shape),
    dims=("time", "individuals"),
)

# pad position
ds["position"] = position.pad({"space": (0, 1)})
ds["position"].coords["space"] = ["x", "y", "z"]

# %%
# compute rotated coordinates
ds_ego = ds.copy()

# ds_ego['position'] = ds['rotation_matrices'].apply(ds['position'])

# %%
xr.apply_ufunc(
    lambda r, p: r.apply(p),
    ds_ego["rotation_matrices"],
    ds_ego["position"],
    input_core_dims=[[], ["keypoints"]],
    output_core_dims=[["keypoints"]],
    vectorize=True,
)


# %%%%%
import xarray as xr

# expand data array to 3d space
posterior2anterior_3d = posterior2anterior.pad({"space": (0, 1)})
posterior2anterior_3d.coords["space"] = ["x", "y", "z"]


# %%


def align_vectors_modif(u, v):
    if np.isnan(v).any():
        return R.from_quat([0, 0, 0, 1])
    else:
        return R.align_vectors(u, v, return_sensitivity=False)


xr.apply_ufunc(
    align_vectors_modif,  # lambda u,v: R.align_vectors(u, v),
    np.broadcast_to(
        np.array([[1, 0, 0]]), posterior2anterior_3d.squeeze().shape
    ),
    posterior2anterior_3d.squeeze(),
    input_core_dims=[[], ["individuals"]],
)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# a = np.array([[1, 0, 0]])
# b = np.array([[0, 1, 0]])

# rrot, rssd = R.align_vectors(
#     a,
#     b,
#     return_sensitivity=False
# )

# print(rrot.as_matrix())

# print(np.testing.assert_allclose(a, rrot.apply(b), atol=1e-10))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot for a small time window

# time_window = range(1650, 1700)  # frames

# fig, ax = plt.subplots(1, 1)
# for mouse_name, col in zip(
#     position.individuals.values, ["r", "g"], strict=False
# ):
#     # plot centroid
#     ax.plot(
#         centroid.sel(individuals=mouse_name, time=time_window, space="x"),
#         centroid.sel(individuals=mouse_name, time=time_window, space="y"),
#         label=mouse_name,
#         color=col,
#         linestyle="-",
#         marker=".",
#         markersize=10,
#         linewidth=0.5,
#     )
#     # plot centroid anterior
#     ax.plot(
#         centroid_anterior.sel(
#             individuals=mouse_name, time=time_window, space="x"
#         ),
#         centroid_anterior.sel(
#             individuals=mouse_name, time=time_window, space="y"
#         ),
#         label=mouse_name,
#         color=col,
#         linestyle="-",
#         marker="x",
#         markersize=10,
#         linewidth=0.5,
#     )
#     # plot centroid posterior
#     ax.plot(
#         centroid_posterior.sel(
#             individuals=mouse_name, time=time_window, space="x"
#         ),
#         centroid_posterior.sel(
#             individuals=mouse_name, time=time_window, space="y"
#         ),
#         label=mouse_name,
#         color=col,
#         linestyle="-",
#         marker="*",
#         markersize=10,
#         linewidth=0.5,
#     )
#     # plot keypoints
#     ax.scatter(
#         x=position.sel(individuals=mouse_name, time=time_window, space="x"),
#         y=position.sel(individuals=mouse_name, time=time_window, space="y"),
#         s=1,
#     )

#     # plot vector
#     ax.quiver(
#         centroid_posterior.sel(
#             individuals=mouse_name, time=time_window, space="x"
#         ),
#         centroid_posterior.sel(
#             individuals=mouse_name, time=time_window, space="y"
#         ),
#         posterior2anterior.sel(
#             individuals=mouse_name, time=time_window, space="x"
#         ),
#         posterior2anterior.sel(
#             individuals=mouse_name, time=time_window, space="y"
#         ),
#         angles="xy",
#         scale=1,
#         scale_units="xy",
#         headwidth=7,
#         headlength=9,
#         headaxislength=9,
#         color="gray",
#     )
#     # # add text
#     # for kpt in position.keypoints.values:
#     #     ax.text(
#     #         position.sel(
#     #             individuals=mouse_name, time=time_window, space='x', keypoints=kpt
#     #         ).data,
#     #         position.sel(
#     #             individuals=mouse_name, time=time_window, space='y', keypoints=kpt
#     #         ).data,
#     #         str(kpt),
#     #     )
# ax.legend()
# ax.axis("equal")
# ax.set_xlabel("x (pixels)")
# ax.set_ylabel("y (pixels)")
# ax.invert_yaxis()
