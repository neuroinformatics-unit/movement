---
blogpost: true
date: October 27, 2025
author: Carlo Castoldi
location: Milan, Italy
category: update
language: English
---

# Improving displacement vectors

Computing kinematic properties is a core functionality since early versions of `movement`, when they where first introduced by [Chang Huan Lo](https://github.com/lochhh) in [#106](https://github.com/neuroinformatics-unit/movement/pull/106).
For a library dedicated to analysing motion-tracking data, quantifying how far a tracked point moves between consecutive frames is fundamental. This measure underpins subsequent computations, such as the total distance travelled along a path. That's why we introduced the `compute_displacement` function early on, and why it features in our {ref}`compute and visualise kinematics <sphx_glr_examples_compute_kinematics.py>` example<!--#compute-displacement-vectors heading-->.

Its implementation, however, produces results of complex interpretability. It defines, for a given individual and keypoint at timestep `t`, a vector that points to the opposite direction of its previous position at time `t-1`.

For this reason, during the Hackday for [OSW 2025](https://neuroinformatics.dev/open-software-summer-school/2025/index.html) and as my first contribution to `movement`, I [offered myself](https://github.com/neuroinformatics-unit/osw25-hackday/issues/16) to work on a more intuitive interface for displacement vectors, supervisioned by [@sfmig](https://github.com/sfmig).
In an inspiring collaborative effort to simplify the user experience, we are pleased to present the key changes introduced in [#657](https://github.com/neuroinformatics-unit/movement/pull/657). This update provides an elegant, well-tested, and thoroughly documented solution that reduces the mental load on users.

__API changes__

{mod}`kinematics <movement.kinematics>` has two new sister functions:

- {func}`compute_forward_displacement <movement.kinematics.compute_forward_displacement>`, computing the vector defined at time `t` that goes from the position in the current frame to the position in the next frame, at `t+1`.
- {func}`compute_backwards_displacement <movement.kinematics.compute_backwards_displacement>`, computing the vector defined at time `t` that goes from the position in the current frame to the position in the previous frame, at `t-1`.

In turn, we deprecated the old implementation {func}`compute_displacement <movement.kinematics.compute_displacement>`. We highly suggest you to re-think your use case in relation to _forward_ and _backward_ displacement vectors. For a drop-in replacement, however, see the example below:

  ```python
  import movement.kinematics as kin

  # Instead of:
  displacement = kin.compute_displacement(ds.position)

  # Use:
  displacement = -kin.compute_backward_displacement(ds.position)
  ```

__Breaking changes__

- We have slightly modified the behaviour when converting vectors from Cartesian to polar coordinates. For simplicity and interpretability, {func}`cart2pol <movement.utils.vector.cart2pol>` now always sets the angle `phi` to 0 when the norm `rho` of the vector is 0, instead of following [C standard](https://www.iso.org/standard/29237.html) for [`arctan2`](https://en.wikipedia.org/wiki/Atan2). This should not be a breaking change because when a vector's length is zero, it is correct to assume that the vector's direction is undefined, meaning that it could point to virtually _any_ direction.

## Conclusions

I would like to extend my sincere gratitude to the [Neuroinformatics Unit](https://neuroinformatics.dev/) for fostering an exceptional open environment that has even inspired me to enhance my own projects. Their efforts have motivated me to make [BraiAn](https://silvalab.codeberg.page/BraiAn/) more accessible to inexperienced researchers, to improve its interoperability and to develop automated pipelines for software verification.

I am firmly convinced that bridging the gap between experimental laboratories is crucial for enabling reproducible and comparable results across research groups. I have long _believed_ that the development and adoption of shared standards and widely accepted platforms can facilitate this goal. The Neuroinformatics Unit has not only reinforced my conviction but also _demonstrated_, through their remarkable work, that this vision can be turned into a practical reality.
