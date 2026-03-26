"""Load BVH motion capture data
================================

Load a `BVH (Biovision Hierarchy)
<https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html>`_
motion capture file into ``movement`` and
visualise the 3D skeleton.
"""

# %%
# Imports
# -------

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from movement.io import load_poses

# %%
# About the BVH format
# ----------------------
# BVH is a widely used text-based motion capture format
# originally developed by Biovision. It stores:
#
# - A **HIERARCHY** section defining the skeleton structure
#   (joints, offsets, and channel types).
# - A **MOTION** section with per-frame channel values
#   (translations and Euler angle rotations).
#
# ``movement`` parses the hierarchy and computes absolute
# 3D joint positions via forward kinematics, making it easy
# to analyse the data with the same tools used for
# animal pose estimation.

# %%
# Create a sample BVH file
# --------------------------
# For this example, we create a simple 5-joint skeleton
# with 20 frames of motion.

n_frames = 20
frame_time = 0.033333  # ~30 fps

# Build the hierarchy string
hierarchy = """\
HIERARCHY
ROOT Hips
{
  OFFSET 0.00 0.00 0.00
  CHANNELS 6 Xposition Yposition Zposition\
 Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0.00 5.00 0.00
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT Head
    {
      OFFSET 0.00 4.00 0.00
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0.00 2.00 0.00
      }
    }
  }
  JOINT LeftHand
  {
    OFFSET 3.00 4.50 0.00
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 4.00 0.00 0.00
    }
  }
  JOINT RightHand
  {
    OFFSET -3.00 4.50 0.00
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET -4.00 0.00 0.00
    }
  }
}
"""

# Generate motion data: walking with arm swing
rng = np.random.default_rng(42)
motion_lines = []
for frame in range(n_frames):
    t = frame / n_frames * 2 * np.pi
    # Root translation: walk forward along X
    xpos = frame * 2.0
    ypos = 0.0
    zpos = 0.5 * np.sin(2 * t)  # slight bounce
    # Root rotation
    zrot, xrot, yrot = 0.0, 0.0, 0.0
    # Spine: slight lean
    spine_z = 3.0 * np.sin(t)
    spine_x, spine_y = 0.0, 0.0
    # Head: look around
    head_z = 5.0 * np.sin(0.5 * t)
    head_x, head_y = 0.0, 0.0
    # Left hand: swing
    lh_z = 20.0 * np.sin(t)
    lh_x, lh_y = 0.0, 0.0
    # Right hand: opposite swing
    rh_z = -20.0 * np.sin(t)
    rh_x, rh_y = 0.0, 0.0

    vals = [
        xpos,
        ypos,
        zpos,
        zrot,
        xrot,
        yrot,
        spine_z,
        spine_x,
        spine_y,
        head_z,
        head_x,
        head_y,
        lh_z,
        lh_x,
        lh_y,
        rh_z,
        rh_x,
        rh_y,
    ]
    motion_lines.append(" ".join(f"{v:.4f}" for v in vals))

motion_section = (
    "MOTION\n"
    f"Frames: {n_frames}\n"
    f"Frame Time: {frame_time}\n" + "\n".join(motion_lines) + "\n"
)
bvh_content = hierarchy + motion_section

# Save to temp file (using NamedTemporaryFile for security)
with tempfile.NamedTemporaryFile(
    mode="w", suffix=".bvh", delete=False
) as f:
    f.write(bvh_content)
    bvh_path = Path(f.name)
print(f"Created sample BVH file: {bvh_path}")

# %%
# Load the BVH file into movement
# ---------------------------------
# :func:`movement.io.load_poses.from_bvh_file` parses the
# BVH hierarchy and computes 3D positions via forward
# kinematics. If ``fps`` is not specified, it is derived
# from the BVH ``Frame Time`` field.

ds = load_poses.from_bvh_file(bvh_path)
print(ds)

# %%
# Explore the dataset
# --------------------

print("Shape:", ds.position.shape)
print(
    "Keypoints (joints):",
    ds.coords["keypoints"].values,
)
print("FPS:", ds.fps)
print("Space dims:", ds.coords["space"].values)

# %%
# Visualise the 3D skeleton at a single frame
# ---------------------------------------------
# Let's plot the skeleton at the first frame.

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

frame_idx = 0
t = ds.coords["time"].values[frame_idx]
pos = ds.position.sel(time=t, individuals="id_0")

# Plot joints
for kp in ds.coords["keypoints"].values:
    p = pos.sel(keypoints=kp).values
    ax.scatter(*p, s=50, zorder=5)
    ax.text(p[0], p[1], p[2] + 0.5, kp, fontsize=8)

# Draw skeleton connections
bones = [
    ("Hips", "Spine"),
    ("Spine", "Head"),
    ("Hips", "LeftHand"),
    ("Hips", "RightHand"),
]
for j1, j2 in bones:
    p1 = pos.sel(keypoints=j1).values
    p2 = pos.sel(keypoints=j2).values
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [p1[2], p2[2]],
        "k-",
        linewidth=2,
    )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("BVH skeleton (frame 0)")
plt.tight_layout()
plt.show()

# %%
# Visualise joint trajectories over time
# ----------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i, coord in enumerate(["x", "y", "z"]):
    ax = axes[i]
    for kp in ds.coords["keypoints"].values:
        vals = ds.position.sel(keypoints=kp, individuals="id_0", space=coord)
        ax.plot(ds.coords["time"], vals, label=kp)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{coord} position")
    ax.set_title(f"{coord.upper()} over time")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

# %%
# BVH data is now in the standard ``movement`` format,
# so you can use all available analysis tools:
# filtering, kinematics computation, distance metrics,
# and more — just as you would with DeepLabCut or SLEAP
# data.

# %%
# Clean up
# --------
bvh_path.unlink()
