## Release [v0.18.0](https://github.com/neuroinformatics-unit/movement/releases/tag/v0.18.0) is out

This is a big release, jam-packed with lots of new features in our GUI, a more streamlined saving interface, and many new kinematic measures.

:warning: **This release contains minor breaking changes to the saving functions, to the `nan_policy` options of path metrics, and to the default behaviour of `filter_by_confidence()`. Please read the notes below before updating.**

To update movement to the latest version, see the [update guide](https://movement.neuroinformatics.dev/latest/user_guide/installation.html#update-the-package).

### :high_voltage: Highlight: Edit pose tracks in the GUI

Pose estimation models often produce errors, such as mis-localised keypoints or false-positive detections in the background. [`movement.filtering`](https://movement.neuroinformatics.dev/latest/api/movement.filtering) offers several post-processing options, but not all errors can be dealt with automatically. You can now correct these stubborn errors by hand, directly in the `movement` GUI.

This feature is the result of @anna-teruel's [Google Summer of Code 2026 project](https://neuroinformatics.dev/blog/gsoc_2026_movement_edit_widget.html). A huge thank you to Anna for designing and building it, and congratulations on joining the `movement` team!

Once you [load tracked data](https://movement.neuroinformatics.dev/latest/user_guide/gui.html#load-tracked-data) into the GUI, you can:
- **Move** keypoints by dragging them to the correct position, or **remove** them altogether.
- See which frames you have edited on the **edit timeline**, optionally split by individual, and click on it to jump to an edited frame.
- **Reset** edits in the current frame or across all frames.
- **Save** the corrected dataset in `movement`'s native netCDF format, and load it again later to pick up where you left off.

The tracks layer stays in sync with the edited keypoints, and edited keypoints are shown as rings. The saved dataset has an additional boolean `edited` data variable, which records which keypoints were edited by hand. You can use it in your analysis, for example to report how many keypoints were corrected.

See the [Edit tracked data](https://movement.neuroinformatics.dev/latest/user_guide/gui.html#edit-tracked-data) section of the GUI guide for details.

**What's next?** Editing currently works one frame at a time and supports moving and removing keypoints in poses datasets, with saving to netCDF only. Planned extensions include:

- editing across a range of frames,
- addition of missing keypoints,
- correction of identity swaps between individuals or keypoints,
- support for editing bounding boxes datasets,
- and saving to third-party formats.

:warning: **The confidence of edited keypoints is set to `NaN`**, since the model's score no longer applies. To make sure your corrections survive filtering, [`filter_by_confidence()`](https://movement.neuroinformatics.dev/latest/api/movement.filtering.filter_by_confidence.html) now **keeps** points with `NaN` confidence by default. This also applies to data proofread in other tools (e.g. SLEAP). Previously, such points were dropped. To restore the old behaviour, set `keep_points_with_nan_confidence=False`.

### :floppy_disk: Unified `save_dataset` entry point and other I/O improvements

Following the unified loading interface introduced in v0.14.0, @lochhh has worked the same magic for saving. [`movement.io.save_dataset()`](https://movement.neuroinformatics.dev/latest/api/movement.io.save.save_dataset.html) saves a `movement` dataset to a file in any supported format. If `target_software` is not specified, the dataset is saved in `movement`'s native netCDF format.

```Python
from movement.io import save_dataset

# movement native netCDF (the default)
save_dataset(ds, "/path/to/file.nc")

# DeepLabCut (one of the supported third-party formats)
save_dataset(ds, "/path/to/file.h5", target_software="DeepLabCut")
```

Format-specific options are passed through as keyword arguments, e.g. `split_individuals` for DeepLabCut or `config` for NWB. Before writing, the dataset is validated, and the file path is checked for write permission and the right suffix for the chosen format. See the [saving section](https://movement.neuroinformatics.dev/latest/user_guide/input_output.html#saving-with-save-dataset) of the Input/Output guide for details.

The software-specific saving functions (e.g. `save_poses.to_dlc_file`) are still available. However, to make the saving and loading interfaces consistent, their `file_path` argument has been renamed to `file`.

:warning: **This is a breaking change** if you pass `file_path` as a keyword argument to `save_poses.to_dlc_file`, `save_poses.to_lp_file`, `save_poses.to_sleap_analysis_file`, or `save_bboxes.to_via_tracks_file`. Code that passes the path as a positional argument is not affected.

```Python
# Before
save_poses.to_dlc_file(ds, file_path="/path/to/file.h5")

# After
save_poses.to_dlc_file(ds, file="/path/to/file.h5")

# Or use the unified entry point
save_dataset(ds, "/path/to/file.h5", target_software="DeepLabCut")
```

:warning: **`save_poses.to_nwb_file` is deprecated** and has been renamed to [`save_poses.to_nwb_file_object`](https://movement.neuroinformatics.dev/latest/api/movement.io.save_poses.to_nwb_file_object.html). It creates `NWBFile` objects without writing them to disk. A future release will repurpose `to_nwb_file` as a function that writes NWB files directly to disk.

```Python
# Before
nwb_files = save_poses.to_nwb_file(ds)

# After: create NWBFile object(s) without writing to disk
nwb_files = save_poses.to_nwb_file_object(ds)

# Or write NWB file(s) directly to disk
save_dataset(ds, "/path/to/file.nwb", target_software="NWB")
```

**Other I/O improvements:**

- Thanks to @ishan372or, `movement` can now load pose tracks from [COCO keypoint results](https://cocodataset.org/#format-results) files, using [`load_poses.from_coco_file()`](https://movement.neuroinformatics.dev/latest/api/movement.io.load_poses.from_coco_file.html) or `load_dataset(..., source_software="COCO")`. Since COCO results do not contain track identities, please read the docstring notes on how individuals are assigned. Along the way, [`load_poses.from_numpy()`](https://movement.neuroinformatics.dev/latest/api/movement.io.load_poses.from_numpy.html) gained a `frame_array` argument for datasets whose frame numbers don't start at 0 or have gaps.
- The GUI can now load data from all 2D formats supported by `load_dataset()` (i.e. all except Anipose and NWB), including COCO.
- If you pass an unsupported `source_software` to `load_dataset()`, the error message now lists the supported options.
- Loaders are now kept in a single central registry, mirroring the new one for writers.
- Several bug fixes, including better support for datasets with individual-wise confidence scores (when creating them and when exporting to DeepLabCut, LightningPose and NWB).

### :triangular_ruler: New kinematic measures: path metrics and angular derivatives

Thanks to @isha822 and @vybhav72954, two more path metrics join those introduced in v0.17.0:

1. [`compute_path_sinuosity()`](https://movement.neuroinformatics.dev/latest/api/movement.kinematics.compute_path_sinuosity.html) quantifies the sinuosity of a trajectory. Higher values indicate more tortuous movement; a perfectly straight path has a sinuosity of 0.

2. [`compute_path_emax()`](https://movement.neuroinformatics.dev/latest/api/movement.kinematics.compute_path_emax.html) computes the maximum expected displacement (E_max), a straightness measure that captures the directional persistence of a path. Larger values indicate straighter paths; values close to zero indicate sinuous paths.

```Python
from movement.kinematics import compute_path_emax, compute_path_sinuosity

# Compute the path traced by each individual's centroid
centroid = ds.position.mean(dim="keypoint")

sinuosity = compute_path_sinuosity(centroid)
emax = compute_path_emax(centroid)
```

We have also added [`compute_angular_velocity()`](https://movement.neuroinformatics.dev/latest/api/movement.kinematics.compute_angular_velocity.html) and [`compute_angular_acceleration()`](https://movement.neuroinformatics.dev/latest/api/movement.kinematics.compute_angular_acceleration.html), which compute the rate of change of a 2D orientation, given as vectors. You can optionally estimate the derivatives over a rolling window, which reduces noise.

```Python
from movement.kinematics import (
    compute_angular_velocity,
    compute_forward_vector,
)

head_vector = compute_forward_vector(ds.position, "left_ear", "right_ear")

# Angular head velocity, in radians per unit of time
ahv = compute_angular_velocity(head_vector)

# A smoother estimate using a rolling window of 7 frames,
# expressed in degrees per unit of time
ahv = compute_angular_velocity(head_vector, window=7, in_degrees=True)
```

See the new [example](https://movement.neuroinformatics.dev/latest/examples/compute_angular_derivatives.html) for a complete walkthrough!

Thanks to @aliviahossain, the handling of missing values in [`compute_path_length()`](https://movement.neuroinformatics.dev/latest/api/movement.kinematics.compute_path_length.html) has been simplified. The `nan_policy` argument now accepts `"interpolate"` (the default), which assumes a straight line is travelled across gaps, or `"omit"`, which skips segments with missing positions entirely. [`compute_path_straightness()`](https://movement.neuroinformatics.dev/latest/api/movement.kinematics.compute_path_straightness.html) no longer has a `nan_policy` argument: it always computes the path length with the `"interpolate"` policy.

:warning: **In `compute_path_length()`, `nan_policy="ffill"` has been renamed to `nan_policy="interpolate"`**, with identical behaviour. Since it is the default, code that doesn't pass `nan_policy` is unaffected. `nan_policy="scale"` has been removed. Passing `"ffill"` or `"scale"` now raises a `ValueError`.

`compute_path_straightness()` returns the same results as with its previous default (`nan_policy="ffill"`), but passing `nan_policy` to it now raises a `TypeError`. For other ways of filling gaps, use [`interpolate_over_time()`](https://movement.neuroinformatics.dev/latest/api/movement.filtering.interpolate_over_time.html) first.

```Python
# Before
length = compute_path_length(centroid, nan_policy="ffill")

# After
length = compute_path_length(centroid, nan_policy="interpolate")

# Or skip segments with missing positions instead
length = compute_path_length(centroid, nan_policy="omit")

# Before
si = compute_path_straightness(centroid, nan_policy="ffill")

# After
si = compute_path_straightness(centroid)
```

### :beaver: New example: annotating behavioural events with BORIS

Thanks to @HollyMorley, a new [example](https://movement.neuroinformatics.dev/latest/examples/advanced/annotate_boris_events.html) shows how to manually annotate events in [BORIS](https://www.boris.unito.it/) (Behavioural Observation Research Interactive Software) and load them into a `movement` dataset as per-frame labels. The example annotates gait phases (whether each limb is in stance or swing) of a mouse locomoting on a dual-belt travelator, and then uses these labels to select and segment the pose data. The same workflow applies to any events of interest that you want to combine with tracking data.

See the [release notes](https://github.com/neuroinformatics-unit/movement/releases/tag/v0.18.0) for more info and a full list of changes.
