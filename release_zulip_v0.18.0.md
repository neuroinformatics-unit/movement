## Release [v0.18.0](https://github.com/neuroinformatics-unit/movement/releases/tag/v0.18.0) is out

### :high_voltage: Unified `save_dataset` entry point

Thanks to @lochhh, saving `movement` datasets now mirrors loading them: a single entry point, `save_dataset`, writes a dataset to any supported third-party format or to `movement`'s native netCDF.

By default, `save_dataset` writes netCDF; pass `target_software` to export to a specific tool's format:

```Python
from movement.io import save_dataset

# movement native netCDF (the default)
save_dataset(ds, "output.nc")

# DeepLabCut
save_dataset(ds, "output.h5", target_software="DeepLabCut")

# VGG Image Annotator tracks
save_dataset(ds, "output.csv", target_software="VIA-tracks")
```

:warning: **This release includes a few small breaking changes to unify the interface:**
- Save functions now take a `file` argument instead of `file_path`.
- `to_via_tracks_file` no longer returns a `Path` (it now returns `None`, like every other save function).
- `to_nwb_file` is deprecated and will be renamed to `to_nwb_file_object`; use `save_dataset(..., target_software="NWB")` to write `.nwb` files to disk.

### :high_voltage: Interactive pose editing and saving in the napari plugin

Huge thanks to @anna-teruel, who drove a whole new workflow in the napari plugin. You can now **manually correct pose predictions directly in napari and save the result back to `movement`**.

Load your poses into napari, then use napari's native Points tools to drag keypoints to the right place or delete inaccurate detections. Edited points are tracked as you go: they are marked as edited, drawn with a distinct ring symbol, and their confidence is set to `NaN`. The Tracks layer stays in sync with your edits to the Points layer.

When you are happy with your corrections, the new **Save tracked data** widget reconstructs a `movement` dataset from the edited Points layer (including points you removed) and writes it to `movement`'s native netCDF format.

### :high_voltage: Two new path metrics — sinuosity and maximum expected displacement

Two new ways to characterise how tortuous or how straight a trajectory is, contributed by @isha822 and @vybhav72954.

`compute_path_sinuosity` quantifies the tortuosity of a path by combining turning-angle statistics with step-length variability. Higher values indicate more tortuous movement; a perfectly straight path has `S = 0`.

`compute_path_emax` computes the maximum expected displacement (E_max), a straightness measure capturing the directional persistence of a path. Larger values indicate straighter, more persistent paths.

```Python
from movement.kinematics import compute_path_sinuosity, compute_path_emax

# a centroid trajectory from a poses dataset `ds`
centroid = ds.position.mean(dim="keypoint")

sinuosity = compute_path_sinuosity(centroid)
emax = compute_path_emax(centroid)
```

See the [release notes](https://github.com/neuroinformatics-unit/movement/releases/tag/v0.18.0) for more info and a full list of changes.
