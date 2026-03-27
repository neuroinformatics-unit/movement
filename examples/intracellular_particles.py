"""Demonstrate trajectory analysis for intracellular particle tracking.

This example simulates noisy and irregularly sampled intracellular particle
trajectories, analyses them with movement.kinematics, and computes
trajectory-level summary metrics such as straightness index and tortuosity.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import movement.kinematics as mk

"""  Synthetic intracellular trajectory generation

# We simulate particle motion similar to intracellular transport
# observed in microscopy
#
# Key properties we model:
#  Directed motion (active transport)
#  Diffusive motion (Brownian-like)
#  Mixed motion (biologically realistic)
#  Noise (localisation error)
#  Missing observations (tracking failures)

"""

rng = np.random.default_rng(42)

n_time = 300  # number of time steps
n_particles = 3  # number of tracked particles


"""Model irregular sampling.

In microscopy, frames are not always evenly spaced due to dropped
frames or hardware constraints.

Simulate this by sampling variable time intervals.
"""

dt = np.clip(rng.normal(loc=0.2, scale=0.03, size=n_time), 0.05, None)
time = np.cumsum(dt)
time = time - time[0]


# Motion models


def directed_motion(n, drift=(1.0, 0.3), noise=0.20):
    """Simulate directed transport along microtubules."""
    steps = rng.normal(scale=noise, size=(n, 2)) + np.array(drift)
    return np.cumsum(steps, axis=0)


def diffusive_motion(n, noise=0.80):
    """Simulate diffusive random-walk motion."""
    steps = rng.normal(scale=noise, size=(n, 2))
    return np.cumsum(steps, axis=0)


def mixed_motion(n):
    """Simulate mixed biological motion.

    The trajectory contains:
    - a directed phase
    - a random exploration phase
    - a final change in direction
    """
    pos = []
    current = np.zeros(2)

    for i in range(n):
        if i < n // 3:
            step = rng.normal(scale=0.20, size=2) + np.array([0.9, 0.2])
        elif i < 2 * n // 3:
            step = rng.normal(scale=0.70, size=2)
        else:
            step = rng.normal(scale=0.20, size=2) + np.array([0.2, -0.8])

        current = current + step
        pos.append(current.copy())

    return np.array(pos)


# Generate trajectories
tracks_clean = np.stack(
    [
        directed_motion(n_time),
        diffusive_motion(n_time),
        mixed_motion(n_time),
    ],
    axis=-1,  # shape: (time, space, individuals)
)


# Add localisation noise
# Represents measurement error in microscopy
tracks_noisy = tracks_clean + rng.normal(scale=0.5, size=tracks_clean.shape)


# Introduce missing observations
""" Simulates:
- occlusion
- tracking failure
- segmentation errors
"""
mask = rng.random((n_time, n_particles)) < 0.10

for i in range(n_particles):
    tracks_noisy[mask[:, i], :, i] = np.nan


""" Convert to movement-compatible dataset
# movement expects data in xarray format with labelled dimensions.
#
Dimensions:
   time
   space (x, y)
   individuals (tracked particles)
"""

ds = xr.Dataset(
    data_vars={
        "position": (("time", "space", "individuals"), tracks_noisy),
    },
    coords={
        "time": time,
        "space": ["x", "y"],
        "individuals": [
            "particle_A_directed",
            "particle_B_diffusive",
            "particle_C_mixed",
        ],
    },
)

# print(ds)


# Core kinematics using movement
# These are LOW-LEVEL quantities already provided by movement.
#
# They describe motion locally (step-by-step)

velocity = mk.compute_velocity(ds.position)  # derivative of position
speed = mk.compute_speed(ds.position)  # magnitude of velocity
path_length = mk.compute_path_length(ds.position, nan_policy="ffill")

print("\nVelocity:")
print(velocity)

print("\nSpeed:")
print(speed)

print("\nPath length:")
print(path_length)


# Higher-level trajectory metrics
# They summarise GLOBAL trajectory structure.


def net_displacement_da(position: xr.DataArray) -> xr.DataArray:
    """Compute straight-line displacement from first to last valid point.

    This represents the "as-the-crow-flies" distance.
    """
    results = []

    for ind in position.individuals.values:
        xy = position.sel(individuals=ind).values
        valid = ~np.isnan(xy).any(axis=1)

        if valid.sum() < 2:
            results.append(np.nan)
            continue

        first = xy[np.argmax(valid)]
        last = xy[len(valid) - 1 - np.argmax(valid[::-1])]

        results.append(np.linalg.norm(last - first))

    return xr.DataArray(
        results,
        coords={"individuals": position.individuals},
        dims=["individuals"],
        name="net_displacement",
    )


def straightness_index(
    position: xr.DataArray, nan_policy: str = "ffill"
) -> xr.DataArray:
    """Straightness index:

        SI = net displacement / path length

    Interpretation:
    - SI -> 1 means straight trajectory (directed motion)
    - SI -> 0 means highly convoluted trajectory
    """
    path = mk.compute_path_length(position, nan_policy=nan_policy)
    net = net_displacement_da(position)

    out = net / path
    out.name = "straightness_index"
    return out


def tortuosity(
    position: xr.DataArray, nan_policy: str = "ffill"
) -> xr.DataArray:
    """Tortuosity:

        t = path length / net displacement

    Interpretation:
    - high means very winding trajectory
    - low means straight trajectory
    """
    path = mk.compute_path_length(position, nan_policy=nan_policy)
    net = net_displacement_da(position)

    out = path / net
    out.name = "tortuosity"
    return out


# Compute metrics
straightness = straightness_index(ds.position, nan_policy="ffill")
tort = tortuosity(ds.position, nan_policy="ffill")

print("\nStraightness index:")
print(straightness)

print("\nTortuosity:")
print(tort)


# Summary statistics
# Combine local (speed) and global (trajectory) metrics

mean_speed = speed.mean(dim="time", skipna=True)
max_speed = speed.max(dim="time", skipna=True)

summary = xr.Dataset(
    {
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "path_length": path_length,
        "net_displacement": net_displacement_da(ds.position),
        "straightness_index": straightness,
        "tortuosity": tort,
    }
)

print("\nSummary:")
print(summary)


# Visualisation: trajectories
# Shows spatial structure of motion

fig, ax = plt.subplots(figsize=(7, 7))

for particle in ds.individuals.values:
    xy = ds.position.sel(individuals=particle).values
    ax.plot(xy[:, 0], xy[:, 1], label=str(particle), alpha=0.85)

ax.set_title("Synthetic intracellular particle trajectories")
ax.set_xlabel("x position")
ax.set_ylabel("y position")
ax.legend()
ax.axis("equal")
plt.show()

# Visualisation: speed over time
# Shows temporal dynamics

fig, ax = plt.subplots(figsize=(10, 5))

for particle in ds.individuals.values:
    s = speed.sel(individuals=particle)
    ax.plot(s.time, s, label=str(particle), alpha=0.85)

ax.set_title("Instantaneous speed over time")
ax.set_xlabel("time")
ax.set_ylabel("speed")
ax.legend()
plt.show()
