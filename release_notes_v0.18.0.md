To update movement to the latest version, see the [update guide](https://movement.neuroinformatics.dev/latest/user_guide/installation.html#update-the-package).

## What's Changed

### ⚡️ Highlight 1: unified `save_dataset` entry point

* Add unified `save_dataset` entry point by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1056

Thanks to @lochhh, saving `movement` datasets now mirrors loading them: a single entry point, [`movement.io.save.save_dataset()`](https://movement.neuroinformatics.dev/latest/api/movement.io.save.save_dataset.html), writes a dataset to any supported third-party format or to `movement`'s native netCDF.

By default, `save_dataset` writes netCDF; pass `target_software` to export to a specific tool's format:

```python
from movement.io import save_dataset

# movement native netCDF (the default)
save_dataset(ds, "output.nc")

# DeepLabCut
save_dataset(ds, "output.h5", target_software="DeepLabCut")

# VGG Image Annotator tracks
save_dataset(ds, "output.csv", target_software="VIA-tracks")
```

Under the hood, a new `register_writer` decorator (analogous to `register_loader`) registers each software-specific writer and decouples dataset and file-path validation from the export logic. The software-specific save functions (e.g. `save_poses.to_dlc_file`, `save_bboxes.to_via_tracks_file`) remain available for full control.

To unify the interface with the loaders, a few small **breaking changes** were needed:

> [!WARNING]
> - Save functions now take a `file` argument instead of `file_path` (matching the load functions).
> - `save_bboxes.to_via_tracks_file` no longer returns a `Path` (it now returns `None`, like every other save function).
> - `save_poses.to_nwb_file` is deprecated and will be renamed to `to_nwb_file_object` in a future release. It currently returns an `NWBFile` object rather than writing to disk; use `save_dataset(..., target_software="NWB")` to write `.nwb` files to disk.

**Migrating to the new argument name:**

```python
# Before
from movement.io import save_poses

save_poses.to_dlc_file(ds, file_path="output.h5")

# After
from movement.io import save_poses

save_poses.to_dlc_file(ds, file="output.h5")

# Or use the unified entry point
from movement.io import save_dataset

save_dataset(ds, "output.h5", target_software="DeepLabCut")
```

### ⚡️ Highlight 2: interactive pose editing and saving in the napari plugin

* Conversion from napari Points layers to movement pose datasets by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1011
* Set confidence of edited points in napari to NaN by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1024
* Reconstruction of removed napari pose predictions back to movement ds by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1025
* Adding edited properties to napari layers when a keypoint is dragged.  by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1041
* Save widget for napari plugin by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1044
* Changing the point symbol to ring for edited points in napari by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1053
* Synchronize tracks layer to points layer by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1054
* Set property edited=True on removed predictions by @anna-teruel in https://github.com/neuroinformatics-unit/movement/pull/1057
* Restore sync-tracks functionality dropped during PR #1054 rebase by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1083

Huge thanks to @anna-teruel, who drove a whole new workflow in the napari plugin across this release. You can now **manually correct pose predictions directly in napari and save the result back to `movement`**.

Load your poses into napari, then use napari's native Points tools to drag keypoints to the right place or delete inaccurate detections. Edited points are tracked as you go: they are marked with `edited=True`, drawn with a distinct ring symbol, and their confidence is set to `NaN` (since a hand-placed point has no model confidence). The Tracks layer stays in sync with your edits to the Points layer.

When you are happy with your corrections, the new **Save tracked data** widget reconstructs a `movement` dataset from the edited Points layer (including points you removed) and writes it to `movement`'s native netCDF format — ready to be reloaded for further analysis.

### ⚡️ Highlight 3: two new path metrics — sinuosity and maximum expected displacement

* Add compute_path_sinuosity to path metrics by @isha822 in https://github.com/neuroinformatics-unit/movement/pull/981
* Add maximum expected displacement (Emax) as a path metric by @vybhav72954 in https://github.com/neuroinformatics-unit/movement/pull/1032

Two new ways to characterise how tortuous or how straight a trajectory is, contributed by @isha822 and @vybhav72954.

[`compute_path_sinuosity()`](https://movement.neuroinformatics.dev/latest/api/movement.kinematics.compute_path_sinuosity.html) quantifies the tortuosity of a path by combining turning-angle statistics with step-length variability (the corrected sinuosity index of Benhamou, 2004). Higher values indicate more tortuous movement; a perfectly straight path has `S = 0`.

[`compute_path_emax()`](https://movement.neuroinformatics.dev/latest/api/movement.kinematics.compute_path_emax.html) computes the maximum expected displacement (E_max), a straightness measure capturing the directional persistence of a path (Cheung et al., 2007). Larger values indicate straighter, more persistent paths.

```python
from movement.kinematics import compute_path_sinuosity, compute_path_emax

# a centroid trajectory from a poses dataset `ds`
centroid = ds.position.mean(dim="keypoint")

sinuosity = compute_path_sinuosity(centroid)
emax = compute_path_emax(centroid)
```

### ✨ New features

* Add `keep_points_with_nan_confidence` kwarg to `filter_by_confidence` by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1037
* Add frame_array support to ValidPosesInputs by @ishan372or in https://github.com/neuroinformatics-unit/movement/pull/1051
* Log warning when saving dataset with individual-wise conf to NWB format by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1045

### 🐛 Bug fixes

* Handle individual-wise confidence during DLC/LP export by @ishan372or in https://github.com/neuroinformatics-unit/movement/pull/1017
* Fix `ValidPosesInputs.to_dataset` for datasets with individual-wise confidence arrays by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1039
* Save single-individual datasets to the given path in `to_dlc_file` by @NoiceHax in https://github.com/neuroinformatics-unit/movement/pull/1078
* Fix `rolling_filter` returning an empty array when `window=1` by @NoiceHax in https://github.com/neuroinformatics-unit/movement/pull/1077
* Simplifying NaN handling in compute_path_length by @aliviahossain in https://github.com/neuroinformatics-unit/movement/pull/1042
* Fix lifetime of napari layer callbacks by @sfmig in https://github.com/neuroinformatics-unit/movement/pull/1090
* Widen frame slider range only if movement layers by @sfmig in https://github.com/neuroinformatics-unit/movement/pull/1107

### 🛠️ Refactoring

* Centralise load registry and refactor `infer_source_software` by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1076

### 📚 Documentation

* Improve look of contributor cards by @PP1703 in https://github.com/neuroinformatics-unit/movement/pull/1023
* Fix broken DeepLabCut URLs by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1031
* A single Zenodo link leading to multiple movement posters by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1035
* Add example for annotating behavioural events with BORIS by @HollyMorley in https://github.com/neuroinformatics-unit/movement/pull/865
* Fix some typos in blogposts by @PolarBean in https://github.com/neuroinformatics-unit/movement/pull/1043
* Update FOSDEM links to archive version by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1087
* Fix broken setuptools-scm links by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1095

### 🤝 Improving the contributor experience

* Fix git commands in contributing guide by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1070
* Run benchmark tests on CI without benchmarking by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1071
* Migrate tox INI to TOML by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1073
* Trusted pypi publishing and release guide by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/1055

### 🧹 Housekeeping

* Bump actions/checkout from 6 to 7 by @dependabot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1030
* Bump actions/cache from 5 to 6 by @dependabot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1029
* Contributors-Readme-Action: Update contributors list by @neuroinformatics-unit-bot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1028
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/neuroinformatics-unit/movement/pull/1033
* Pin netCDF4>1.7.3 by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1046
* Contributors-Readme-Action: Update contributors list by @neuroinformatics-unit-bot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1061
* Bump neuroinformatics-unit/actions from 2 to 3 by @dependabot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1062
* Update linkcheck configuration and permanently redirected URLs by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1068
* Migrate data repository to SWC GIN by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1080
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/neuroinformatics-unit/movement/pull/1064
* Ignore all g-node subdomains during linkcheck by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1082
* Fix redirected UCL ARC URLs by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/1085
* Contributors-Readme-Action: Update contributors list by @neuroinformatics-unit-bot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1086
* Bump pyvista/setup-headless-display-action from 4 to 5 by @dependabot[bot] in https://github.com/neuroinformatics-unit/movement/pull/1089
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/neuroinformatics-unit/movement/pull/1096

## New Contributors
* @PP1703 made their first contribution in https://github.com/neuroinformatics-unit/movement/pull/1023
* @PolarBean made their first contribution in https://github.com/neuroinformatics-unit/movement/pull/1043
* @NoiceHax made their first contribution in https://github.com/neuroinformatics-unit/movement/pull/1078
* @aliviahossain made their first contribution in https://github.com/neuroinformatics-unit/movement/pull/1042

**Full Changelog**: https://github.com/neuroinformatics-unit/movement/compare/v0.17.0...v0.18.0
