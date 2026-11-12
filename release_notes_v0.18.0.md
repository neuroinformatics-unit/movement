This is a big release, jam-packed with lots of new features in our GUI, a more streamlined saving interface, and many new kinematic measures.

> [!important]
> This release contains minor breaking changes to the saving functions, to the `nan_policy` options of path metrics, and to the default behaviour of `filter_by_confidence()`. Please read the notes below before updating.

To update movement to the latest version, see the [update guide](https://movement.neuroinformatics.dev/latest/user_guide/installation.html#update-the-package).

## What's Changed

### ⚡️ Highlight: Edit pose tracks in the GUI

Pose estimation models often produce errors, such as mis-localised keypoints or false-positive detections in the background. [`movement.filtering`](https://movement.neuroinformatics.dev/latest/api/movement.filtering) offers several post-processing options, but not all errors can be dealt with automatically. You can now correct these stubborn errors by hand, directly in the `movement` GUI.

This feature is the result of @anna-teruel's [Google Summer of Code 2026 project](https://neuroinformatics.dev/blog/gsoc_2026_movement_edit_widget.html). A huge thank you to Anna for designing and building it, and congratulations on joining the `movement` team! Thanks also to @SahilKumar75 and @devanshc777 for contributing the reset controls and a re-export fix.

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

> [!WARNING]
> The confidence of edited keypoints is set to `NaN`, since the model's score no longer applies. To make sure your corrections survive filtering, [`filter_by_confidence()`](https://movement.neuroinformatics.dev/latest/api/movement.filtering.filter_by_confidence.html) now **keeps** points with `NaN` confidence by default. This also applies to data proofread in other tools (e.g. SLEAP). Previously, such points were dropped. To restore the old behaviour, set `keep_points_with_nan_confidence=False`.

**Pull requests:**

* Conversion from napari Points layers to movement pose datasets by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1011
* Set confidence of edited points in napari to NaN by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1024
* Reconstruction of removed napari pose predictions back to movement ds by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1025
* Adding edited properties to napari layers when a keypoint is dragged.  by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1041
* Save widget for napari plugin by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1044
* Changing the point symbol to ring for edited points in napari by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1053
* Synchronize tracks layer to points layer by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1054
* Set property edited=True on removed predictions by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1057
* Add napari timeline widget to flag edited points by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1063
* Restore sync-tracks functionality dropped during PR #1054 rebase by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1083
* Add frame and full pose reset controls by @SahilKumar75 in https://github.com/neuroinformatics-unit/movement/pull/1137
* Keep edited flag of removed points on re-export by @devanshc777 in https://github.com/neuroinformatics-unit/movement/pull/1190
* Fix removed points missing from the edit timeline by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1199
* Fix lifetime of napari layer callbacks by @sfmig in https://github.com/neuroinformatics-unit/movement/pull/1090
* Widen frame slider range only if movement layers by @sfmig in https://github.com/neuroinformatics-unit/movement/pull/1107
* Document edit widget and how to generate GUI videos by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1126
* Add `keep_points_with_nan_confidence` kwarg to `filter_by_confidence` by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1037

### 💾 Unified `save_dataset` entry point and other I/O improvements

Following the unified loading interface introduced in v0.14.0, @lochhh has worked the same magic for saving. [`movement.io.save_dataset()`](https://movement.neuroinformatics.dev/latest/api/movement.io.save.save_dataset.html) saves a `movement` dataset to a file in any supported format. If `target_software` is not specified, the dataset is saved in `movement`'s native netCDF format.

```python
from movement.io import save_dataset

# movement native netCDF (the default)
save_dataset(ds, "/path/to/file.nc")

# DeepLabCut (one of the supported third-party formats)
save_dataset(ds, "/path/to/file.h5", target_software="DeepLabCut")
```

Format-specific options are passed through as keyword arguments, e.g. `split_individuals` for DeepLabCut or `config` for NWB. Before writing, the dataset is validated, and the file path is checked for write permission and the right suffix for the chosen format. See the [saving section](https://movement.neuroinformatics.dev/latest/user_guide/input_output.html#saving-with-save-dataset) of the Input/Output guide for details.

Downstream packages can use [`movement.io.save.get_supported_target_software()`](https://movement.neuroinformatics.dev/latest/api/movement.io.save.get_supported_target_software.html) to find out which formats `movement` can save to.

The software-specific saving functions (e.g. `save_poses.to_dlc_file`) are still available. However, to make the saving and loading interfaces consistent, their `file_path` argument has been renamed to `file`.

> [!WARNING]
> This is a **breaking change** if you pass `file_path` as a keyword argument to `save_poses.to_dlc_file`, `save_poses.to_lp_file`, `save_poses.to_sleap_analysis_file`, or `save_bboxes.to_via_tracks_file`. Code that passes the path as a positional argument is not affected.
>
> ```python
> # Before
> save_poses.to_dlc_file(ds, file_path="/path/to/file.h5")
>
> # After
> save_poses.to_dlc_file(ds, file="/path/to/file.h5")
>
> # Or use the unified entry point
> save_dataset(ds, "/path/to/file.h5", target_software="DeepLabCut")
> ```

> [!WARNING]
> `save_poses.to_nwb_file` is deprecated and has been renamed to [`save_poses.to_nwb_file_object`](https://movement.neuroinformatics.dev/latest/api/movement.io.save_poses.to_nwb_file_object.html). It creates `NWBFile` objects without writing them to disk. A future release will repurpose `to_nwb_file` as a function that writes NWB files directly to disk.
>
> ```python
> # Before
> nwb_files = save_poses.to_nwb_file(ds)
>
> # After: create NWBFile object(s) without writing to disk
> nwb_files = save_poses.to_nwb_file_object(ds)
>
> # Or write NWB file(s) directly to disk
> save_dataset(ds, "/path/to/file.nwb", target_software="NWB")
> ```

**Other I/O improvements:**

- Thanks to @ishan372or, `movement` can now load pose tracks from [COCO keypoint results](https://cocodataset.org/#format-results) files, using [`load_poses.from_coco_file()`](https://movement.neuroinformatics.dev/latest/api/movement.io.load_poses.from_coco_file.html) or `load_dataset(..., source_software="COCO")`. Since COCO results do not contain track identities, please read the docstring notes on how individuals are assigned. Along the way, [`load_poses.from_numpy()`](https://movement.neuroinformatics.dev/latest/api/movement.io.load_poses.from_numpy.html) gained a `frame_array` argument for datasets whose frame numbers don't start at 0 or have gaps.
- The GUI can now load data from all 2D formats supported by `load_dataset()` (i.e. all except Anipose and NWB), including COCO.
- If you pass an unsupported `source_software` to `load_dataset()`, the error message now lists the supported options.
- Loaders are now kept in a single central registry, mirroring the new one for writers.
- Several bug fixes, including better support for datasets with individual-wise confidence scores (when creating them and when exporting to DeepLabCut, LightningPose and NWB).

**Pull requests:**

* Add unified `save_dataset` entry point by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1056
* Add COCO keypoint results loader by @ishan372or in https://github.com/neuroinformatics-unit/movement/pull/1092
* Add frame_array support to ValidPosesInputs by @ishan372or in https://github.com/neuroinformatics-unit/movement/pull/1051
* Support all registered loaders in GUI unless excluded by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1191
* List supported software in `load_dataset` error by @dhruv1955 in https://github.com/neuroinformatics-unit/movement/pull/858
* Centralise load registry and refactor `infer_source_software` by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1076
* Handle individual-wise confidence during DLC/LP export by @ishan372or in https://github.com/neuroinformatics-unit/movement/pull/1017
* Fix `ValidPosesInputs.to_dataset` for datasets with individual-wise confidence arrays by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1039
* Log warning when saving dataset with individual-wise conf to NWB format by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1045
* Save single-individual datasets to the given path in `to_dlc_file` by @NoiceHax in https://github.com/neuroinformatics-unit/movement/pull/1078
* Round frame numbers when exporting VIA tracks with time in seconds by @SajalDevX in https://github.com/neuroinformatics-unit/movement/pull/1122

### 📐 New kinematic measures: path metrics and angular derivatives

Thanks to @isha822 and @vybhav72954, two more path metrics join those introduced in v0.17.0:

1. [`compute_path_sinuosity()`](https://movement.neuroinformatics.dev/latest/api/movement.kinematics.compute_path_sinuosity.html) quantifies the sinuosity of a trajectory. Higher values indicate more tortuous movement; a perfectly straight path has a sinuosity of 0.

2. [`compute_path_emax()`](https://movement.neuroinformatics.dev/latest/api/movement.kinematics.compute_path_emax.html) computes the maximum expected displacement (E<sub>max</sub>), a straightness measure that captures the directional persistence of a path. Larger values indicate straighter paths; values close to zero indicate sinuous paths.

```python
from movement.kinematics import compute_path_emax, compute_path_sinuosity

# Compute the path traced by each individual's centroid
centroid = ds.position.mean(dim="keypoint")

sinuosity = compute_path_sinuosity(centroid)
emax = compute_path_emax(centroid)
```

We have also added [`compute_angular_velocity()`](https://movement.neuroinformatics.dev/latest/api/movement.kinematics.compute_angular_velocity.html) and [`compute_angular_acceleration()`](https://movement.neuroinformatics.dev/latest/api/movement.kinematics.compute_angular_acceleration.html), which compute the rate of change of a 2D orientation, given as vectors. You can optionally estimate the derivatives over a rolling window, which reduces noise.

```python
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

> [!WARNING]
> In `compute_path_length()`, `nan_policy="ffill"` has been renamed to `nan_policy="interpolate"`, with identical behaviour. Since it is the default, code that doesn't pass `nan_policy` is unaffected. `nan_policy="scale"` has been removed. Passing `"ffill"` or `"scale"` now raises a `ValueError`.
>
> `compute_path_straightness()` returns the same results as with its previous default (`nan_policy="ffill"`), but passing `nan_policy` to it now raises a `TypeError`. For other ways of filling gaps, use [`interpolate_over_time()`](https://movement.neuroinformatics.dev/latest/api/movement.filtering.interpolate_over_time.html) first.
>
> ```python
> # Before
> length = compute_path_length(centroid, nan_policy="ffill")
>
> # After
> length = compute_path_length(centroid, nan_policy="interpolate")
>
> # Or skip segments with missing positions instead
> length = compute_path_length(centroid, nan_policy="omit")
>
> # Before
> si = compute_path_straightness(centroid, nan_policy="ffill")
>
> # After
> si = compute_path_straightness(centroid)
> ```

**Pull requests:**

* Add compute_path_sinuosity to path metrics by @isha822 in https://github.com/neuroinformatics-unit/movement/pull/981
* Add maximum expected displacement (Emax) as a path metric by @vybhav72954 in https://github.com/neuroinformatics-unit/movement/pull/1032
* Add functions to compute angular velocity and acceleration by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1146
* Simplifying NaN handling in compute_path_length by @aliviahossain in https://github.com/neuroinformatics-unit/movement/pull/1042
* Remove `nan_policy` option in `compute_path_straightness` by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1201
* Fix NaN warning for single-track data in path metrics by @Sarthak-Shreshtha01 in https://github.com/neuroinformatics-unit/movement/pull/1157
* Validate camera_view in orientation functions by @reyyishreyas in https://github.com/neuroinformatics-unit/movement/pull/1140

### 🦫 New example: annotating behavioural events with BORIS

Thanks to @HollyMorley, a new [example](https://movement.neuroinformatics.dev/latest/examples/advanced/annotate_boris_events.html) shows how to manually annotate events in [BORIS](https://www.boris.unito.it/) (Behavioural Observation Research Interactive Software) and load them into a `movement` dataset as per-frame labels. The example annotates gait phases (whether each limb is in stance or swing) of a mouse locomoting on a dual-belt travelator, and then uses these labels to select and segment the pose data. The same workflow applies to any events of interest that you want to combine with tracking data.

* Add example for annotating behavioural events with BORIS by @HollyMorley in https://github.com/neuroinformatics-unit/movement/pull/865

### 🤝 Community and contributor experience

We have written down how the `movement` project is run. The new [governance page](https://movement.neuroinformatics.dev/latest/community/governance.html) describes the roles of contributors, team members and maintainers, and explains how decisions are made. We have also introduced pull request review limits: new contributors can have one PR marked as "Ready for review" at a time, until two of their PRs have been merged.

* Define governance and contributor roles by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1099
* Add Anna Teruel as a team member by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1143
* Add PR review limits to contributing guide by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1189
* Convert code-of-conduct.rst to .md by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1119
* Improve look of contributor cards by @PP1703 in https://github.com/neuroinformatics-unit/movement/pull/1023
* Trusted pypi publishing and release guide by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1055
* Fix git commands in contributing guide by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1070
* Add IO guide update step to implementing new loaders guide by @reyyishreyas in https://github.com/neuroinformatics-unit/movement/pull/1115
* Run benchmark tests on CI without benchmarking by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1071
* Migrate tox INI to TOML by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1073
* Check numpydoc section order in package docstrings by @SahilKumar75 in https://github.com/neuroinformatics-unit/movement/pull/1149
* Use headless viewers in napari layer style, save widget and regions widget tests by @SahilKumar75 in https://github.com/neuroinformatics-unit/movement/pull/1128
* Test that single-frame datasets keep a 1D time coordinate by @SajalDevX in https://github.com/neuroinformatics-unit/movement/pull/1130
* Migrate data repository to SWC GIN by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1080
* Fall back to HTTP when sample data cannot be fetched over HTTPS by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1152
* Generate GUI guide screenshots with a script by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1162

### 🧹 Housekeeping

* Pin netCDF4>1.7.3 by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1046
* Update linkcheck configuration and permanently redirected URLs by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1068
* Ignore all g-node subdomains during linkcheck by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1082
* Pin GitHub Actions to commit SHAs by @Aniketsy in https://github.com/neuroinformatics-unit/movement/pull/1145
* Fix tests broken by pytest 9.2 and PyQt6 6.12 by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1176
* Notify on slack when scheduled conda workflow fails by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1193
* Fix `rolling_filter` returning an empty array when `window=1` by @NoiceHax in https://github.com/neuroinformatics-unit/movement/pull/1077
* Fix broken DeepLabCut URLs by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1031
* A single Zenodo link leading to multiple movement posters by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1035
* Fix some typos in blogposts by @PolarBean in https://github.com/neuroinformatics-unit/movement/pull/1043
* Fix redirected UCL ARC URLs by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1085
* Update FOSDEM links to archive version by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1087
* Fix broken setuptools-scm links by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1095
* Fix trailing slash in html_baseurl configuration by @IgorTatarnikov in https://github.com/neuroinformatics-unit/movement/pull/1151
* Fix broken docstring examples in load_bboxes, sample_data and broadcasting by @aliviahossain in https://github.com/neuroinformatics-unit/movement/pull/1168
* Fix FreeMoCap URL in example by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1186
* Fix docs build failing with sphinx-design 0.8 by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1200
* Bump actions/checkout from 6 to 7 by @dependabot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1030
* Bump actions/cache from 5 to 6 by @dependabot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1029
* Bump neuroinformatics-unit/actions from 2 to 3 by @dependabot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1062
* Bump pyvista/setup-headless-display-action from 4 to 5 by @dependabot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1089
* Bump conda-incubator/setup-miniconda from 4.0.1 to 4.1.0 by @dependabot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1132
* Bump pypa/gh-action-pypi-publish from 1.14.2 to 1.14.3 by @dependabot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1183
* Bump actions/download-artifact from 8.0.1 to 8.0.2 by @dependabot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1182
* Bump pyvista/setup-headless-display-action from 5.1.0 to 5.2.0 by @dependabot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1181
* Bump actions/create-github-app-token from 3.2.0 to 3.3.0 by @dependabot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1180
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/neuroinformatics-unit/movement/pull/1033
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/neuroinformatics-unit/movement/pull/1064
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/neuroinformatics-unit/movement/pull/1096
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/neuroinformatics-unit/movement/pull/1141
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/neuroinformatics-unit/movement/pull/1188
* Contributors-Readme-Action: Update contributors list by @neuroinformatics-unit-bot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1028
* Contributors-Readme-Action: Update contributors list by @neuroinformatics-unit-bot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1061
* Contributors-Readme-Action: Update contributors list by @neuroinformatics-unit-bot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1086
* Contributors-Readme-Action: Update contributors list by @neuroinformatics-unit-bot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1131
* Contributors-Readme-Action: Update contributors list by @neuroinformatics-unit-bot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1178

## New Contributors
* @PP1703 made their first contribution in https://github.com/neuroinformatics-unit/movement/pull/1023
* @PolarBean made their first contribution in https://github.com/neuroinformatics-unit/movement/pull/1043
* @NoiceHax made their first contribution in https://github.com/neuroinformatics-unit/movement/pull/1078
* @aliviahossain made their first contribution in https://github.com/neuroinformatics-unit/movement/pull/1042
* @reyyishreyas made their first contribution in https://github.com/neuroinformatics-unit/movement/pull/1115
* @SajalDevX made their first contribution in https://github.com/neuroinformatics-unit/movement/pull/1122
* @Aniketsy made their first contribution in https://github.com/neuroinformatics-unit/movement/pull/1145
* @SahilKumar75 made their first contribution in https://github.com/neuroinformatics-unit/movement/pull/1137
* @Sarthak-Shreshtha01 made their first contribution in https://github.com/neuroinformatics-unit/movement/pull/1157
* @devanshc777 made their first contribution in https://github.com/neuroinformatics-unit/movement/pull/1190

**Full Changelog**: https://github.com/neuroinformatics-unit/movement/compare/v0.17.0...v0.18.0
