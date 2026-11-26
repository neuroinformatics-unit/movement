# Design: exporting `movement` datasets to `pynapple`

- **Issue:** [movement#1098](https://github.com/neuroinformatics-unit/movement/issues/1098)
- **Branch:** `export-to-pynapple` (based on PR 1197, to be rebased on `main` once it merges)
- **Status:** design agreed, implementation not started (2026-11-26)
- **Working notes / exploration history:** `export-to-pynapple.md`

## 1. Summary

Add a direct `movement` → `pynapple` export, owned by `movement` (as SpikeInterface owns `to_pynapple_tsgroup()`). It consists of two functions in a new module, `movement/io/save_pynapple.py`:

- `to_pynapple_object(ds, ...)` returns pynapple objects in memory.
- `to_pynapple_file(ds, path, ...)` writes them to a folder that `nap.load_folder()` can read, and is registered with `save_dataset(..., target_software="pynapple")`.

Each data variable becomes one `TsdFrame`. All individuals, keypoints and spatial components are stacked as columns, and each column carries `individual` / `keypoint` / `space` metadata, so no labels are lost. pynapple is an optional dependency (`movement[pynapple]`).

## 2. Background

There is already an indirect route (`to_nwb_file_object()` → `nap.NWBFile`), but it has limitations:

- Confidence scores are lost.
- It supports only one individual per file (an NWB constraint).
- It doesn't support bounding boxes.
- Extra variables (e.g. `head_direction`) don't survive.
- Each keypoint becomes a separate `TsdFrame`.

Agreed with the pynapple devs (gviejo) in [pynapple#617](https://github.com/pynapple-org/pynapple/pull/617) and movement#1098:

- Pursue **both** routes. pynapple will improve its `ndx-pose` reading ([pynapple#650](https://github.com/pynapple-org/pynapple/issues/650), pynapple-side), and movement will add a direct exporter (this work).
- The original suggestion was a numpy-only `.npz` export, to avoid depending on pynapple while it isn't on conda-forge. We've since moved away from that (see decision D-1).

Other context:

- [pynapple#617](https://github.com/pynapple-org/pynapple/pull/617) prototypes `nap.from_movement()` on the pynapple side (nested dict of per-individual frames, no column metadata).
- [pynaviz#118](https://github.com/pynapple-org/pynaviz/pull/118) prototypes skeleton plotting. It expects a `TsdFrame` with `<kp>_x`/`<kp>_y` column pairs, plus optional per-column `parent` metadata for edges.
- [pynapple#689](https://github.com/pynapple-org/pynapple/issues/689) proposes `TsdFrame.sel(individual=..., keypoint=...)` over column metadata, motivated by movement data. Our data model fits it directly.
- [movement#1127](https://github.com/neuroinformatics-unit/movement/pull/1127) is an earlier community attempt (a single `TsdFrame` holding all variables, numpy-written, no metadata). We are not building on its data model.

## 3. Planned design

### 3.1 Public API

```python
# movement/io/save_pynapple.py

def to_pynapple_object(
    ds: xr.Dataset,
    variables: list[str] | None = None,
    split_individuals: bool = False,
) -> dict[str, nap.TsdFrame | nap.Tsd] | dict[str, dict[str, nap.TsdFrame | nap.Tsd]]:
    """Convert a movement dataset to pynapple objects (in memory)."""


@register_writer("pynapple", path_kind="dir")  # ds_type=None: poses and bboxes
def to_pynapple_file(
    ds: xr.Dataset,
    path: str | Path,
    variables: list[str] | None = None,
    split_individuals: bool = False,
) -> None:
    """Save a movement dataset as a pynapple folder of .npz files."""
```

- `to_pynapple_file` = `to_pynapple_object` + `obj.save()` for each object + a JSON sidecar. There's one code path, and pynapple builds, validates and serialises its own objects.
- pynapple is imported lazily inside these functions. If it's missing, they raise an informative `ImportError` (`pip install movement[pynapple]`).
- This follows the existing "in-memory + registered file writer" pairs: `to_nwb_file_object` / `_write_nwb_file` and `to_dlc_style_df` / `to_dlc_file`.

Usage:

```python
from movement.io import save_dataset
from movement.io.save_pynapple import to_pynapple_object

objs = to_pynapple_object(ds)                        # {"position": TsdFrame, "confidence": TsdFrame, ...}
save_dataset(ds, "/data/session", target_software="pynapple")

import pynapple as nap
folder = nap.load_folder("/data/session")
pos = folder["position"]
pos[:, (pos.individual == "mouse1") & (pos.keypoint == "snout")]
```

### 3.2 Mapping movement variables to pynapple objects

**Which variables are exported.** By default, every data variable that is numeric and has a `time` dimension. Each one becomes its own object, keyed by the variable name. Ineligible variables are skipped with a warning. `variables=[...]` selects a subset and raises if a requested variable is ineligible.

**Object type.**
- `TsdFrame` of shape `(n_time, n_columns)`: the non-time dimensions are stacked into columns.
- `Tsd`: only when no non-time dimensions remain (e.g. `head_direction` with `split_individuals=True`).
- `TsdTensor` is not used (no metadata support, so labels would be lost).

**Column order** is individual → keypoint → space, with `space` innermost, so consecutive columns form `x`, `y` (, `z`) pairs.

**Column metadata (source of truth):** one metadata field per stacked dimension (`individual`, `keypoint`, `space`), with string values.

**Column names:** the labels of the stacked dimensions joined with `_`. Names are for display. Code should select via metadata, because names can be ambiguous when labels contain `_` (e.g. `tail_base`).

Example: a 2D poses dataset with individuals `mouse1`, `mouse2` and keypoints `snout`, `tail_base`, plus a user-added `speed` (dims `time, keypoint, individual`).

| Variable | Default (`split_individuals=False`) | `split_individuals=True` (per individual) |
|---|---|---|
| `position` | `TsdFrame`, 8 cols: `mouse1_snout_x`, `mouse1_snout_y`, `mouse1_tail_base_x`, ... | `TsdFrame`, 4 cols: `snout_x`, `snout_y`, `tail_base_x`, `tail_base_y` |
| `confidence` | `TsdFrame`, 4 cols: `mouse1_snout`, `mouse1_tail_base`, ... | `TsdFrame`, 2 cols: `snout`, `tail_base` |
| `speed` | `TsdFrame`, 4 cols (as confidence) | `TsdFrame`, 2 cols |
| `head_direction` (time, individual) | `TsdFrame`, 2 cols: `mouse1`, `mouse2` | `Tsd` |

Bounding boxes work the same way. `position`, `shape` and `confidence` each become one object, and `shape` keeps movement's `space` labels (`x`/`y`), not `width`/`height`.

With `split_individuals=False`, the naming scheme is identical for one or many individuals, so the schema never depends on the individual count. With `split_individuals=True`, `position` frames have exactly the `<kp>_x`/`<kp>_y` layout that the pynaviz skeleton mode parses today.

**Time.**
- The time coordinate must be in seconds (`ds.attrs["time_unit"] == "seconds"`). Otherwise the export raises, telling the user to provide `fps`, because pynapple always interprets time as seconds.
- The time support is `[t[0], t[-1]]`, which is pynapple's default when building objects from timestamps.
- NaNs pass through unchanged.

### 3.3 On-disk layout

*Provisional (D-4); see open question 1.*

`to_pynapple_file(ds, "session")` creates a new directory:

```
session/
├── position.npz        # written by pynapple's TsdFrame.save()
├── position.json       # sidecar: dataset attrs
├── confidence.npz
├── confidence.json
└── speed.npz / speed.json
```

With `split_individuals=True`, there is one subfolder per individual. `nap.load_folder` walks subfolders, so `folder["mouse1"]["position"]` works:

```
session/
├── mouse1/
│   ├── position.npz / position.json
│   └── confidence.npz / confidence.json
└── mouse2/
    └── ...
```

**Sidecars** follow pynapple's `Folder` convention: one `<name>.json` per `<name>.npz`, which `Folder.info(name)` displays. Its reader requires a **flat dict of strings**, so all values are stringified. Content: the dataset `attrs` (`ds_type`, `source_software`, `fps`, `time_unit`, `source_file`, `log`, ...), plus `time` (export timestamp) and `info` (short description), matching the keys `Folder.save()` writes.

The in-memory objects from `to_pynapple_object` don't carry `ds.attrs`, since pynapple objects have no free-form attrs. Attrs are only preserved on disk.

### 3.4 Changes to shared save machinery

*Provisional (D-6); only needed if the folder output (D-4) stands.*

`register_writer` and `validate_file_path` currently assume a single output file. We'll add an explicit directory mode (working name `path_kind="dir"`):

- The path must not exist (no overwrite). If it does, the error says "directory already exists", instead of the current misleading "expected a file path but got a directory".
- The path must have no suffix (so `"out.npz"` doesn't silently become a folder).
- The parent directory must be writable.

Also update the `WriterProtocol` docstring, `save_dataset`'s `file` parameter docs, and the contributing guide ("Implementing new writers").

### 3.5 Packaging

- `pyproject.toml`: a new optional extra `pynapple = ["pynapple>=0.8"]`, plus pynapple in the dev/test and docs groups.
- conda-forge: no recipe change, since optional dependencies aren't listed. Conda-only users `pip install pynapple` to use the export. pynapple itself is not on conda-forge yet.
- `"pynapple"` is added to the `TargetSoftware` literal. `movement/io/__init__.py` imports `save_pynapple` so the writer registers. `docs/make_api.py` gets the new module in `PACKAGE_MODULES`.

### 3.6 Docs and tests (in this PR)

**Docs:**
- API reference for the new module.
- IO guide: a row in the supported-formats table and a "pynapple" saving tab.
- Contributing guide: directory-based writers.
- A **small gallery example** showcasing the export and its options:
  - in-memory vs folder export
  - `variables=`
  - `split_individuals=`
  - an added derived variable (e.g. speed)
  - loading back with `nap.load_folder`
  - metadata-based column selection
  - the `set_info(parent=...)` recipe for pynaviz skeletons

  There's no ephys data in this PR, so no new public dataset is needed.

**Tests:**
- Mapping: types, shapes, column names/order, metadata, `Tsd` vs `TsdFrame`.
- Poses and bboxes, 2D and 3D, single and multiple individuals.
- `variables=`, including errors and skip warnings.
- `split_individuals=`.
- Error for time in frames.
- Directory-path validation.
- Round-trip via `nap.load_folder`; sidecar readable by `Folder.info`.
- `ImportError` when pynapple is missing (monkeypatched).

## 4. Decisions

**Firm** means the decision is settled unless new information appears. **Provisional** means we chose it to unblock, and reviewers and pynapple devs are welcome to challenge it.

| # | Decision | Rationale | Status |
|---|---|---|---|
| D-1 | Provide both an in-memory export and a file export, with pynapple as an **optional** dependency for both. | An optional, lazily imported dependency doesn't block conda-forge packaging. pynapple validates and serialises its own objects, so we don't hand-write its file format (including the private `_metadata` key). One code path. | Firm |
| D-2 | **One `TsdFrame` per data variable**, with all individuals/keypoints/space stacked as columns and labelled via column metadata. | Lossless. Schema independent of individual count. Units not mixed within an object. Generic for extra variables. Matches pynapple#689. | Firm |
| D-3 | Optional `split_individuals=False`, giving per-individual objects. | Behaviour is most often analysed alongside ephys one animal at a time, which is pynapple's main use case. Gives pynaviz-ready frames today. Mirrors DLC's `split_individuals`. | Firm |
| D-4 | File export writes a **folder** of `.npz` + `.json` sidecars, loadable with `nap.load_folder`. | Best pynapple UX. Uses pynapple's `Folder` convention. Sidecars preserve `ds.attrs`. Downside: it doesn't match movement's `to_<format>_file(ds, file)` pattern or the `file` argument of `save_dataset`/`WriterProtocol` (see open question 1). | **Provisional** |
| D-5 | New module `movement/io/save_pynapple.py` with a single writer for poses and bboxes (`ds_type=None`). | Logic is generic across dataset types. The name matches `save_poses`/`save_bboxes` and avoids clashing with the `pynapple` package name. | Firm |
| D-6 | Add an explicit directory mode to `register_writer`. | Clear errors and documented intent, instead of special-casing inside one writer. Only needed if D-4 stands. | **Provisional** (depends on D-4) |
| D-7 | Error on datasets whose time is in frames. | Exporting frames as seconds would silently misalign data. | Firm |
| D-8 | Time support `[t[0], t[-1]]`. | pynapple's own default. | Firm |
| D-9 | Export all eligible variables by default, with an optional `variables=` filter. | Preserves derived variables (a stated limitation of the NWB route) while letting users opt out. | Firm |
| D-10 | Skeletons out of scope for v1. Document the `set_info(parent=...)` recipe. | movement datasets don't store skeletons, and the pynaviz representation isn't settled (pynaviz#118 open). | Firm for v1 |
| D-11 | `pynapple>=0.8`. | Column metadata was introduced in 0.8.0. Save/load round-trip verified on 0.8.0, 0.8.5, 0.9.0 and 0.11.4, and on Python 3.14. CI only tests the latest version. | Firm |
| D-12 | Column naming: labels of stacked dimensions joined by `_`, always including the individual (unless split). | Consistent schema. Metadata is the source of truth. | **Provisional:** revisit with pynaviz |
| D-13 | Name `to_pynapple_object`. Returns a plain dict keyed by variable (nested `{individual: {variable: obj}}` when split). Subfolders per individual on disk. `Tsd` only for time-only variables. | Mirrors `to_nwb_file_object`. A plain dict is the simplest container, and the in-memory and on-disk shapes match. | **Provisional** |
| D-14 | Bboxes `shape` keeps `x`/`y` labels. | One generic rule, faithful to movement's data model. | **Provisional** |

## 5. Alternatives considered

- **numpy-only `.npz` writer** (originally agreed first pass, also PR 1127): needs no dependency, but means hand-writing pynapple's format and gets no validation at write time. Superseded by D-1.
- **A single `TsdFrame` for everything** (PR 1127): mixes position, confidence and extras in one object, breaks the x/y column pairing, and the schema varies with the individual count.
- **Per-(individual, variable) frames as the default** (pynapple#617 `from_movement`): many objects, and individual identity lives in names rather than data. Kept as the `split_individuals=True` option instead.
- **`TsdTensor` per variable:** keeps the n-D shape but loses all labels.
- **Sibling files** (`session_position.npz`, ...) instead of a folder: fits the current file-based contract and the existing precedent for multi-file outputs (NWB and DLC with multiple individuals), but gives a worse pynapple UX (`load_folder` keys become `session_position`, and files mix with unrelated ones in the parent dir). This is the main fallback if D-4 is revisited.
- **Keeping the converter on the pynapple side only** (`nap.from_movement`): we prefer domain packages exporting towards the integrating package (the SpikeInterface precedent), and movement already maintains many format converters.

## 6. Open questions (ours, to settle during implementation)

1. **Folder output vs movement's file-based pattern (D-4, D-6).** Every movement writer is `to_<format>_file(ds, file)`, and `save_dataset`/`WriterProtocol` take a `file` argument. A folder output breaks both: `to_pynapple_file` would write a folder, and `save_dataset(ds, file="session", ...)` would receive a directory. Options:
   - Accept the mismatch, and document in `save_dataset` that `file` may be a directory for some formats.
   - Rename the writer to `to_pynapple_folder` (consistent with pynapple, inconsistent with movement), and/or generalise the contract's argument name (e.g. `path`). That's a breaking change for a public API, so probably not.
   - Fall back to sibling files (section 5), which keeps the file pattern intact at some cost to pynapple UX.
2. **If the folder output stays:** the name of the directory option in `register_writer` (`path_kind="dir"`, `is_dir=True`, ...), and whether it ships as a separate commit or PR.
3. **Extra dimensions** (e.g. `view` in multi-view datasets, or user-defined dims). Should they be stacked as additional metadata fields (generic, lossless), or should variables with them be skipped? Proposed: stack them, ordered after `individual` and before `keypoint`, `space`.
4. **Boolean variables** (e.g. ROI "inside" results): `np.issubdtype(bool, np.number)` is `False`. Export them as-is, cast them, or skip them?
5. **Duplicate column names** from labels containing `_`: warn, raise, or accept, given that metadata disambiguates?
6. **Sidecar content:** whether to also include the movement version and the variable's dims/units.

## 7. Questions for pynapple devs

1. **Dependency.** We moved from a numpy-only `.npz` export to building objects with pynapple (optional dependency) and saving via `obj.save()`, which means **file export requires pynapple installed**. OK with you? Is there any timeline for pynapple on conda-forge?
2. **Data model.** Does one wide `TsdFrame` per variable, with `individual`/`keypoint`/`space` column metadata, match how you'd want users to work in pynapple? Any plans for pynapple#689 (`.sel`-style selection on metadata)?
3. **pynaviz.** Could skeleton mode resolve keypoints, individuals and spatial components from **column metadata** rather than parsing `_x`/`_y` column names? That would make it work with multi-individual frames and 3D, and make our column-naming choice (D-12) irrelevant.
4. **Skeleton representation.** Is per-column `parent` metadata the long-term choice, or an edge list (which can also express non-tree graphs)? Where should it live (metadata, sidecar, separate object)? We'd like to add a `skeleton=` option to the exporter once this is settled.
5. **Folder layout and sidecars.** Is a folder of `.npz` + `.json` sidecars (with subfolders per individual) idiomatic? Are arbitrary string keys in the sidecar fine, beyond `time`/`info`?
6. **`nap.from_movement` (pynapple#617).** With the export living in movement, do you plan to close it or keep it? We'd like to avoid two converters with diverging conventions.

## 8. Out of scope / follow-ups

- A `skeleton=` option (after Q7.4).
- A docs example combining tracking with ephys (needs a suitable public dataset).
- NWB route improvements: pynapple reading `ndx-pose` (pynapple#650), and movement storing extra variables in NWB.
- Exporting ROI events/occupancy as `IntervalSet`.
- Re-checking pynapple's conda-forge availability (for an eventual non-optional integration).

## 9. Process notes

- Message gviejo with section 7 once the draft PR is up.
- PR 1127: close it with thanks and credit (e.g. `Co-authored-by`), or invite the contributor to test the new PR.
- Rebase on `main` after PR 1197 merges.
