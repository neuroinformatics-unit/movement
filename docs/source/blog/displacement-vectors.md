---
blogpost: true
date: October 27, 2025
author: Carlo Castoldi
location: Milan, Italy
category: update
language: English
---

# Replacing displacement vectors for greater clarity

This post introduces recent improvements to how movement computes displacement vectors, making the definitions more explicit, flexible, and intuitive for users.

Computing kinematic properties is a core functionality since early versions of `movement`, when they where first introduced by [Chang Huan Lo](https://github.com/lochhh) in [#106](https://github.com/neuroinformatics-unit/movement/pull/106).
For a library dedicated to analysing motion-tracking data, quantifying how far a tracked point moves between consecutive frames is fundamental. This measure underpins subsequent computations, such as the total distance travelled along a path. That's why we introduced the `compute_displacement` function early on, and why it features in our {ref}`compute and visualise kinematics <sphx_glr_examples_compute_kinematics.py>` example<!--#compute-displacement-vectors heading-->.

Its original implementation, however, produced results that were difficult to interpret. For a given individual and keypoint at timestep `t`, displacement was defined as the vector pointing from the previous position at `t-1` to the current position at `t`. This definition is somewhat counter-intuitive: it identifies the last spatial translation used by the keypoint in order to reach its current position. It informed of where the point _came from_ rather than where it is _going_.

For this reason, during the Hackday at [Open Software Week 2025](https://neuroinformatics.dev/open-software-summer-school/2025/index.html)—and as my first contribution to `movement`—I [volunteered](https://github.com/neuroinformatics-unit/osw25-hackday/issues/16) to develop a more intuitive interface for displacement vectors, under the supervision of [Sofía Miñano](https://github.com/sfmig).
These improvements were introduced in [#657](https://github.com/neuroinformatics-unit/movement/pull/657) through a collaborative effort. The update provides a simpler, well-tested, and better-documented implementation that makes displacement computations easier to understand and use.

![A diagram that shows the comparison between the previous implementation and the updated one.](resources/displacement_old_vs_new.png)

__API changes__

{mod}`kinematics <movement.kinematics>` has two new sister functions:

- {func}`compute_forward_displacement <movement.kinematics.compute_forward_displacement>`, computing the vector defined at time `t` that goes from the position in the current frame to the position in the next frame, at `t+1`.
- {func}`compute_backward_displacement <movement.kinematics.compute_backward_displacement>`, computing the vector defined at time `t` that goes from the position in the current frame to the position in the previous frame, at `t-1`.

In turn, we deprecated the old implementation {func}`compute_displacement <movement.kinematics.compute_displacement>`. We highly suggest you to re-think your use case in relation to _forward_ and _backward_ displacement vectors. For a drop-in replacement, however, see the example below:

  ```python
  import movement.kinematics as kin

  # Instead of:
  displacement = kin.compute_displacement(ds.position)

  # Use:
  displacement = -kin.compute_backward_displacement(ds.position)
  ```

__Breaking changes__

We slightly modified the behaviour of vector conversion from Cartesian to polar coordinates. For simplicity and interpretability, {func}`cart2pol <movement.utils.vector.cart2pol>` now always sets the angle `phi` to 0 when the vector's norm `rho` is 0, rather than following the [C standard](https://www.iso.org/standard/29237.html) for [`arctan2`](https://en.wikipedia.org/wiki/Atan2). This change should not affect existing workflows, as a zero-length vector has an undefined direction—meaning it could point in any direction, and assigning `phi = 0` is a safe, neutral choice.

## Conclusions

I would like to extend my sincere gratitude to the [Neuroinformatics Unit](https://neuroinformatics.dev/) for fostering an exceptional open environment that has even inspired me to enhance my own projects. Their efforts have motivated me to make [BraiAn](https://silvalab.codeberg.page/BraiAn/) more accessible to inexperienced researchers, to improve its interoperability and to develop automated pipelines for software verification.

I am firmly convinced that bridging the gap between experimental laboratories is crucial for enabling reproducible and comparable results across research groups. I have long _believed_ that the development and adoption of shared standards and widely accepted platforms can facilitate this goal. The Neuroinformatics Unit has not only reinforced my conviction but also _demonstrated_, through their remarkable work, that this vision can be turned into a practical reality.
