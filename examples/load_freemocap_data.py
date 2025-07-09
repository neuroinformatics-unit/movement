"""Load FreeMoCap Data
==========================================

Load the ..
"""

# %%
# %matplotlib widget
import matplotlib.pyplot as plt
import numpy as np

from movement.io import load_poses

path_hello = (
    "C:/Users/Max/Downloads/session_2025-07-08_15_35_19/"
    "session_2025-07-08_15_35_19/recording_15_37_37_gmt+1/output_data/"
    "mediapipe_right_hand_3d_xyz.npy"
)
path_world = (
    "C:/Users/Max/Downloads/session_2025-07-08_15_35_19/"
    "session_2025-07-08_15_35_19/recording_15_45_49_gmt+1/output_data/"
    "mediapipe_right_hand_3d_xyz.npy"
)
# %%
data_hello = np.load(path_hello)
data_hello = np.array([data_hello])
data_world = np.load(path_world)
data_world = np.array([data_world])
print(data_hello.shape)
# %%
data_transposed_hello = np.transpose(data_hello, [1, 3, 2, 0])
data_transposed_world = np.transpose(data_world, [1, 3, 2, 0])
print(data_transposed_hello.shape)
# %%
ds_hello = load_poses.from_numpy(
    position_array=data_transposed_hello,
    confidence_array=np.ones(np.delete(data_transposed_hello.shape, 1)),
    individual_names=["person_0"],
)
ds_world = load_poses.from_numpy(
    position_array=data_transposed_world,
    confidence_array=np.ones(np.delete(data_transposed_world.shape, 1)),
    individual_names=["person_0"],
)
# %%
position_hello = ds_hello.position
position_world = ds_world.position
# %%
# fig3d = plt.figure()
# ax3d = fig3d.add_subplot(projection='3d')
# for i in range(21):
#     x=position_hello.sel(keypoints=f"keypoint_{i}", space="x")
#     z=position_hello.sel(keypoints=f"keypoint_{i}", space="z")
#     y=position_hello.sel(keypoints=f"keypoint_{i}", space="y")
#     t=position_hello.time
#     ax3d.plot(x,y,z,".-")
fig3d = plt.figure()
ax3d = fig3d.add_subplot(projection="3d")
for i in range(2, 12):
    x = position_world.sel(keypoints=f"keypoint_{i}", space="x")
    z = position_world.sel(keypoints=f"keypoint_{i}", space="z")
    y = position_world.sel(keypoints=f"keypoint_{i}", space="y")
    t = position_world.time
    ax3d.plot(x, y, z, ".-")
# %%
