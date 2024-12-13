# Glossary

:::{glossary}

pose estimation
: The process of estimating the positions of body parts (keypoints) in an image
  or video frame.

pose tracks
: A sequence of pose estimates for an individual or multiple individuals over time.
  Each pose estimate consists of the positions of body parts and associated confidence values.

keypoint
: A body part or feature of interest that is tracked by a
  {term}`pose estimation` algorithm. Examples include the nose, ears etc.

individual
: An entity that is tracked by a {term}`pose estimation` algorithm.
  This could be an animal, a person, or an object.

confidence
: A value that indicates the confidence of the machine learning model in the
  estimated position of a keypoint. In `movement`, this refers to the
  point-wise confidence values output by the {term}`pose estimation` software.
: The homonymous data variable in a [movement dataset](target-poses-and-bboxes-dataset) holds
  the confidence values for all individuals and keypoints at each timestep.

position
: The estimated location of a {term}`keypoint` in an image or video frame.
  This is typically represented in 2D or 3D Cartesian coordinates.
: The position vector {math}`\vec{r_i}(t)` for keypoint {math}`i` at time {math}`t`
  points from the coordinate system origin to the keypoint's location.
  ```{math} \vec{r_i}(t) = (x_i(t), y_i(t))
  :label: position
  ```
  where {math}`x_i(t)`, {math}`y_i(t)` are the coordinates of
  keypoint {math}`i` at time {math}`t` (for 3D add {math}`z_i(t)`).
: The homonymous data variable in a [movement dataset](target-poses-and-bboxes-dataset)
  holds the positions of all individuals and keypoints at each timestep.
displacement:
: The change in {term}`position` of a keypoint between two consecutive frames.
  The displacement vector {math}`\vec{d_i}(t)` for keypoint {math}`i` at time
  {math}`t` points from the keypoint's position at time {math}`t-1` to its
  location at time {math}`t`.
  ```{math} \vec{d_i}(t) = \vec{r_i}(t) - \vec{r_i}(t-1)
  :label: displacement
  ```
  where {math}`\vec{r_i}(t)` is as defined in {math:numref}`position`.
: The homonymous data variable in a [movement dataset](target-poses-and-bboxes-dataset)
  holds the displacements of all individuals and keypoints at each timestep.
  At {math}`t=0`, the displacement is {math}`\vec{0}`.

velocity
: The rate of change of {term}`position` of a keypoint with respect to time.
  The velocity vector {math}`\vec{v_i}(t)` for keypoint {math}`i` at time
  {math}`t` is the derivative of the position vector {math}`\vec{r_i}(t)`
  with respect to time.
  ```{math} \vec{v_i}(t) = \frac{d\vec{r_i}(t)}{dt}
  :label: velocity
  ```
  where {math}`\vec{r_i}(t)` is as defined in {math:numref}`position`.
: The homonymous data variable in a [movement dataset](target-poses-and-bboxes-dataset)
  holds the velocities of all individuals and keypoints at each timestep.
acceleration
: The rate of change of {term}`velocity` of a keypoint with respect to time.
  The acceleration vector {math}`\vec{a_i}(t)` for keypoint {math}`i` at time
  {math}`t` is the derivative of the velocity vector {math}`\vec{v_i}(t)`
  with respect to time.
  ```{math} \vec{a_i}(t) = \frac{d\vec{v_i}(t)}{dt}
  :label: acceleration
  ```
  where {math}`\vec{v_i}(t)` is as defined in {math:numref}`velocity`.
: The homonymous data variable in a [movement dataset](target-poses-and-bboxes-dataset)
  holds the accelerations of all individuals and keypoints at each timestep.

:::
