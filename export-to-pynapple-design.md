# Design: exporting `movement` data to `pynapple`

- **Issue:** [movement#1098](https://github.com/neuroinformatics-unit/movement/issues/1098)
- **Branch:** `export-to-pynapple` (based on `main`)
- **Status:** in-memory export agreed, implementation not started (2026-11-27). Saving to disk deferred to a follow-up PR (section 7).
- **Working notes / exploration history:** `export-to-pynapple.md`

## 1. Summary

This PR adds a direct, **in-memory** `movement` → `pynapple` export, owned by `movement` (as SpikeInterface owns `to_pynapple_tsgroup()`). It consists of two functions in a new module, `movement/io/to_pynapple.py`:

- `to_pynapple_tsdframe(da)` converts one `xr.DataArray` to one `TsdFrame`.
- `to_pynapple_dict(ds, variables=None)` converts an `xr.Dataset` to `{variable_name: TsdFrame}` by calling `to_pynapple_tsdframe` on each eligible variable.

All non-time dimensions (individuals, keypoints, spatial components, ...) are stacked as columns. Each column carries one metadata field per stacked dimension, so no labels are lost. pynapple is an optional dependency (`movement[pynapple]`).

Writing to disk (a `save_dataset(..., target_software="pynapple")` writer) is **not** part of this PR. Users save with pynapple's own `obj.save()`, as SpikeInterface users do. The on-disk layout will be settled in a follow-up, after discussion with the pynapple devs and movement co-maintainers (section 7).

## 2. Background

### 2.1 The existing NWB route

There is already an indirect route (`to_nwb_file_object()` → `nap.NWBFile`), but it has limitations:

- Confidence scores are lost.
- It supports only one individual per file (an NWB constraint).
- It doesn't support bounding boxes.
- Extra variables (e.g. `head_direction`) don't survive.
- Each keypoint becomes a separate `TsdFrame`.

Agreed with the pynapple devs (gviejo) in [pynapple#617](https://github.com/pynapple-org/pynapple/pull/617) and movement#1098:

- Pursue **both** routes. pynapple will improve its `ndx-pose` reading ([pynapple#650](https://github.com/pynapple-org/pynapple/issues/650), pynapple-side), and movement will add a direct exporter (this work).
- The original suggestion was a numpy-only `.npz` export, to avoid depending on pynapple while it isn't on conda-forge. We've since moved away from that (see D-1).

### 2.2 Precedent: SpikeInterface's `to_pynapple_tsgroup`

`spikeinterface.exporters.to_pynapple_tsgroup` (checked on SpikeInterface `main`, 2026-11-27):

- **In-memory only.** Returns one `TsGroup`. The docs tell users to save it with pynapple's `.save("x.npz")`. There is no file writer, folder convention or sidecar.
- **Exports only what SpikeInterface owns** (spike trains), with rich metadata attached (unit locations, quality/template metrics, sorting properties).
- **Adapts to pynapple constraints via metadata:** non-integer unit ids are replaced by `0..n-1`, the originals are kept as a `unit_id` metadata column, and a warning is issued.
- **One segment per call** (`segment_index=`). Users concatenate in pynapple.
- **Optional dependency**, checked at call time, with an `ImportError` hint. Listed in the `spikeinterface[exporters]` extra.

Our design follows the same shape: in-memory, returning stock pynapple objects, labels carried as metadata, slicing left to the user.

### 2.3 pynapple conventions

Checked in the pynapple 0.11.4 source:

- **NWB loader** (`io/interface_nwb.py`): each NWB `TimeSeries` becomes its own object, keyed by name. Columns are the components or channels of that one series (`x`/`y`/`z` for a `SpatialSeries`, or electrodes/ROIs with their tables as metadata). So the dominant convention is **one object per signal, columns = its components**.
- **Legacy loader** (`io/loader.py`): merges `CompassDirection` and `Position` into one `position` frame (`rx, ry, rz, x, y, z`). So frames that mix units have precedent, but only in the old API.
- **`compute_tuning_curves(data, features)`** treats every column of `features` as a feature dimension. A frame is used directly only when all its columns are wanted, so in practice users select columns first. Selection is unavoidable for multi-keypoint or multi-individual data in any design.

This supports one `TsdFrame` per variable (D-2) rather than one frame holding everything.

### 2.4 Related work

- [pynapple#617](https://github.com/pynapple-org/pynapple/pull/617) prototypes `nap.from_movement()` on the pynapple side. It returns a `MovementDataset(UserDict)` with an `NWBFile`-style summary repr, lazy building and per-individual nesting. It handles only `position`/`confidence`/`shape` and has no column metadata.
- [pynaviz#118](https://github.com/pynapple-org/pynaviz/pull/118) prototypes skeleton plotting. It expects a `TsdFrame` with `<kp>_x`/`<kp>_y` column pairs, plus optional per-column `parent` metadata for edges.
- [pynapple#689](https://github.com/pynapple-org/pynapple/issues/689) proposes `TsdFrame.sel(individual=..., keypoint=...)` over column metadata, motivated by movement data. Our data model fits it directly.
- [movement#1127](https://github.com/neuroinformatics-unit/movement/pull/1127) is an earlier community attempt (a single `TsdFrame` holding all variables, numpy-written, no metadata). We are not building on its data model.

## 3. Design (this PR)

### 3.1 Public API

```python
# movement/io/to_pynapple.py


def to_pynapple_tsdframe(da: xr.DataArray) -> nap.TsdFrame:
    """Convert a movement DataArray to a pynapple TsdFrame."""


def to_pynapple_dict(
    ds: xr.Dataset,
    variables: list[str] | None = None,
) -> dict[str, nap.TsdFrame]:
    """Convert a movement dataset to a dict of pynapple TsdFrames."""
```

- `to_pynapple_dict` is a thin loop: select eligible variables, warn about skipped ones, and call `to_pynapple_tsdframe` on each.
- Naming follows the "suffix = return type" pattern: `to_dlc_style_df`, `to_nwb_file_object` (movement), `to_pynapple_tsgroup` (SpikeInterface).
- pynapple is imported lazily inside the functions. If it's missing, they raise an informative `ImportError` (`pip install movement[pynapple]`).
- Nothing is registered with `save_dataset`, so `movement/io/__init__.py` doesn't need to import the module.

Usage:

```python
import pynapple as nap
from movement.io.to_pynapple import to_pynapple_dict, to_pynapple_tsdframe
from movement.kinematics import compute_speed

# {"position": TsdFrame, "confidence": TsdFrame, ...}
objs = to_pynapple_dict(ds)
pos = objs["position"]
pos[:, (pos.individual == "mouse1") & (pos.keypoint == "snout")]

# no need to add speed to ds first
speed = to_pynapple_tsdframe(compute_speed(ds.position))

# per individual: select first (see 3.3)
mouse1 = to_pynapple_dict(ds.sel(individual="mouse1"))

# saving is pynapple's job
for name, obj in objs.items():
    obj.save(f"session/{name}.npz")  # later: nap.load_folder("session")
```

### 3.2 Mapping a DataArray to a `TsdFrame`

**Object type.** Always a `TsdFrame` of shape `(n_time, n_columns)`.
- The non-time dimensions are stacked into columns.
- A time-only array (e.g. `head_direction.sel(individual="mouse1")`) becomes a single-column `TsdFrame`, with the column named after `da.name`.
- `Tsd` is not used, so the return type never depends on the input shape.
- `TsdTensor` is not used either (no metadata support, so labels would be lost).

**Column order** is individual → keypoint → space, with `space` innermost, so consecutive columns form `x`, `y` (, `z`) pairs.

**Column metadata (source of truth):** one metadata field per stacked dimension (`individual`, `keypoint`, `space`), with string values.

**Column names:** the labels of the stacked dimensions joined with `_`. Names are for display. Code should select via metadata, because names can be ambiguous when labels contain `_` (e.g. `tail_base`).

Example: a 2D poses dataset with individuals `mouse1`, `mouse2` and keypoints `snout`, `tail_base`, plus a user-added `speed` (dims `time, keypoint, individual`).

| Variable | `to_pynapple_dict(ds)` | `to_pynapple_dict(ds.sel(individual="mouse1"))` |
|---|---|---|
| `position` | 8 cols: `mouse1_snout_x`, `mouse1_snout_y`, `mouse1_tail_base_x`, ... | 4 cols: `snout_x`, `snout_y`, `tail_base_x`, `tail_base_y` |
| `confidence` | 4 cols: `mouse1_snout`, `mouse1_tail_base`, ... | 2 cols: `snout`, `tail_base` |
| `speed` | 4 cols (as confidence) | 2 cols |
| `head_direction` (time, individual) | 2 cols: `mouse1`, `mouse2` | 1 col: `head_direction` |

Bounding boxes work the same way. `position`, `shape` and `confidence` each become one frame, and `shape` keeps movement's `space` labels (`x`/`y`), not `width`/`height`.

The naming scheme is the same for one or many individuals, so the schema never depends on the individual count. After a scalar `.sel(individual=...)`, `position` frames have exactly the `<kp>_x`/`<kp>_y` layout that the pynaviz skeleton mode parses today.

**Time.**
- pynapple interprets timestamps as seconds. The functions **assume** the `time` coordinate is in seconds and don't check it. Both docstrings document this, with the conversion recipe:

  ```python
  da = da.assign_coords(time=da.time / fps)
  ```

  A proper fix (storing the time unit in the array's attrs) is planned separately in movement. Once it lands, the exporter can read and check it.
- The time support is `[t[0], t[-1]]`, which is pynapple's default when building objects from timestamps.
- NaNs pass through unchanged.

**Attributes.** pynapple objects have no free-form attrs, so `ds.attrs` (`source_software`, `fps`, `log`, ...) are not carried over. This is documented. Preserving them on disk is a question for the follow-up writer (section 7).

### 3.3 Mapping a Dataset to a dict

- **Which variables.** By default, every data variable that is numeric and has a `time` dimension. Each one becomes one entry, keyed by the variable name. Ineligible variables are skipped with a warning. `variables=[...]` selects a subset and raises if a requested variable is ineligible.
- **Return type.** A plain `dict[str, TsdFrame]`, always flat.
  - No wrapper class (unlike `MovementDataset` in pynapple#617): a summary repr is a pynapple-side concern, next to `NWBFile`/`Folder`. Lazy building gains nothing for in-memory xarray data. A plain dict works directly with `obj.save()` loops and the future writer. See D-6 and Q8.5.
- **Per-individual export** uses xarray selection instead of a `split_individuals` parameter (D-3). Docstrings document the recipe:

  ```python
  # one individual
  to_pynapple_dict(ds.sel(individual="mouse1"))
  # all individuals, split
  {ind: to_pynapple_dict(ds.sel(individual=ind)) for ind in ds.individual.values}
  ```

  Note the scalar `.sel()`: `.sel(individual=["mouse1"])` with a list keeps the dimension, so the columns keep the individual prefix and metadata. The same selection approach covers keypoint subsets and time windows.

### 3.4 Packaging

- `pyproject.toml`: a new optional extra `pynapple = ["pynapple>=0.8"]`, plus pynapple in the dev/test and docs groups.
- conda-forge: no recipe change, since optional dependencies aren't listed. Conda-only users `pip install pynapple` to use the export. pynapple itself is not on conda-forge yet.
- API docs: the module is picked up automatically by `docs/make_api.py` (only packages need a `PACKAGE_MODULES` entry).

### 3.5 Docs and tests

**Docs:**
- API reference for the new module (automatic).
- IO guide: a short "Exporting to pynapple" section (in-memory conversion, then `obj.save()`).
- Docstrings: the seconds assumption with the `assign_coords` recipe, and the per-individual `.sel()` recipe.
- A **small gallery example** covering:
  - `to_pynapple_dict` and `to_pynapple_tsdframe`, the latter on a derived variable (e.g. speed)
  - `variables=`
  - per-individual export via `.sel()`
  - metadata-based column selection
  - saving with `obj.save()` and loading back with `nap.load_folder`
  - the `set_info(parent=...)` recipe for pynaviz skeletons

  There's no ephys data in this PR, so no new public dataset is needed.

**Tests:**
- Mapping: shapes, column names/order, metadata, single-column frames for time-only arrays.
- Poses and bboxes, 2D and 3D, single and multiple individuals.
- Scalar `.sel()` drops the individual from names and metadata.
- `variables=`, including errors and skip warnings.
- `ImportError` when pynapple is missing (monkeypatched).

## 4. Decisions (this PR)

**Firm** means the decision is settled unless new information appears. **Provisional** means we chose it to unblock, and reviewers and pynapple devs are welcome to challenge it.

| # | Decision | Rationale | Status |
|---|---|---|---|
| D-1 | In-memory export only, building objects with pynapple as an **optional**, lazily imported dependency. Saving is left to pynapple's `obj.save()`. | Matches the SpikeInterface precedent. pynapple validates its own objects. Defers the folder-vs-file question (section 7) without blocking the export. | Firm |
| D-2 | **One `TsdFrame` per data variable**, with all non-time dims stacked as columns and labelled via column metadata. | Matches pynapple's one-object-per-signal convention (2.3). Lossless. Schema independent of individual count. Units not mixed within an object. Matches pynapple#689. | Firm |
| D-3 | No `split_individuals` parameter. Per-individual export = `.sel(individual=...)` first, documented as a recipe. | Same result as the flag, with no duplicated mechanism and no flag-dependent nested return type. Also covers keypoint/time subsets. May come back in the writer, where it would decide the on-disk layout. | Firm |
| D-4 | Two functions: `to_pynapple_tsdframe(da)` as the core, `to_pynapple_dict(ds, variables=None)` as a loop over it. Not one function with an input-dependent return type. | DataArray input fits movement's pipeline style (kinematics/filtering return DataArrays). Two names are clearer than `@overload`s. | Firm |
| D-5 | Always `TsdFrame`, never `Tsd`. Time-only arrays become a single column named after `da.name`. | The return type matches the function name and never depends on input shape. | Firm |
| D-6 | `to_pynapple_dict` returns a plain `dict`, not a container class. | A pynapple-style session view belongs in pynapple. A movement copy would drift and add a dependency. | Firm |
| D-7 | Time is assumed to be in seconds, documented, not checked. | DataArrays don't carry `time_unit` today. Proper fix planned separately (time unit in array attrs). | Firm for v1 |
| D-8 | Time support `[t[0], t[-1]]`. | pynapple's own default. | Firm |
| D-9 | Convert all eligible variables by default, with an optional `variables=` filter. | Preserves derived variables (a stated limitation of the NWB route) while letting users opt out. | Firm |
| D-10 | Skeletons out of scope for v1. Document the `set_info(parent=...)` recipe. | movement datasets don't store skeletons, and the pynaviz representation isn't settled (pynaviz#118 open). | Firm for v1 |
| D-11 | `pynapple>=0.8`. | Column metadata was introduced in 0.8.0. Save/load round-trip verified on 0.8.0, 0.8.5, 0.9.0 and 0.11.4, and on Python 3.14. CI only tests the latest version. | Firm |
| D-12 | Column naming: labels of stacked dimensions joined by `_`. | Consistent schema. Metadata is the source of truth. | **Provisional:** revisit with pynaviz |
| D-13 | Bboxes `shape` keeps `x`/`y` labels. | One generic rule, faithful to movement's data model. | **Provisional** |
| D-14 | Module `movement/io/to_pynapple.py`. | Mirrors SpikeInterface's `exporters/to_pynapple.py`. Avoids clashing with the `pynapple` package name. The follow-up writer can live in the same module. | **Provisional** |

## 5. Alternatives considered

- **numpy-only `.npz` writer** (originally agreed first pass, also PR 1127): needs no dependency, but means hand-writing pynapple's format and gets no validation at write time. Superseded by D-1.
- **A single `TsdFrame` for everything** (PR 1127): works technically (verified: 14 columns with a `variable` metadata field, round-trips through `save`/`load_file`), but goes against pynapple's one-object-per-signal convention, mixes units and dtypes, and always needs a `variable` filter. Revisited for the on-disk format in section 7.
- **`split_individuals=` parameter / per-(individual, variable) frames as the default** (pynapple#617): replaced by `.sel()` (D-3).
- **One function accepting Dataset or DataArray:** the return type would depend on the input type. Replaced by two functions (D-4).
- **`MovementDataset`-style container** (pynapple#617): see D-6.
- **`TsdTensor` per variable:** keeps the n-D shape but loses all labels.
- **Keeping the converter on the pynapple side only** (`nap.from_movement`): we prefer domain packages exporting towards the integrating package (the SpikeInterface precedent), and movement already maintains many format converters.

## 6. Open questions (this PR, to settle during implementation)

1. **Extra dimensions** (e.g. `view` in multi-view datasets, or user-defined dims). Should they be stacked as additional metadata fields (generic, lossless), or should variables with them be skipped? Proposed: stack them, ordered after `individual` and before `keypoint`, `space`.
2. **Boolean variables** (e.g. RoI "inside" results): `np.issubdtype(bool, np.number)` is `False`. Export them as-is, cast them, or skip them?
3. **Duplicate column names** from labels containing `_`: warn, raise, or accept, given that metadata disambiguates?
4. **Unnamed time-only DataArrays** (`da.name is None`): what to call the single column?
5. **Scalar coordinates** left by `.sel()` (e.g. `individual="mouse1"`): turn them into constant metadata columns, so frames record which animal they're from? Default: no.

## 7. Follow-up PR: saving to disk

Not in this PR. To be settled with gviejo and the movement co-maintainers first.

**Goal:** a writer registered with `save_dataset(..., target_software="pynapple")` (working name `to_pynapple_file`), built on `to_pynapple_dict` + `obj.save()`, so pynapple builds and serialises its own objects.

**The main question is the on-disk layout.** Pynapple's `.npz` format holds one object per file (`nap.load_file` returns one object), while `nap.load_folder` reads a folder of them. Every movement writer is `to_<format>_file(ds, file)`, and `save_dataset`/`WriterProtocol` take a single `file`. Options:

1. **Folder of `.npz` + `.json` sidecars**, loadable with `nap.load_folder`:

   ```
   session/
   ├── position.npz        # written by TsdFrame.save()
   ├── position.json       # sidecar: dataset attrs
   ├── confidence.npz
   └── confidence.json
   ```

   - Best pynapple UX (`folder["position"]`). Sidecars follow pynapple's `Folder` convention (one `<name>.json` per `<name>.npz`, shown by `Folder.info(name)`). Its reader requires a **flat dict of strings**, so all values are stringified. Content: `ds.attrs` plus `time` (export timestamp) and `info` (short description), matching the keys `Folder.save()` writes. This is the only option that keeps `ds.attrs`.
   - Breaks movement's single-file contract. It needs a directory mode in `register_writer` (working name `path_kind="dir"`):
     - the path must not exist (no overwrite), with a clear "directory already exists" error;
     - the path must have no suffix;
     - the parent must be writable.

     The `WriterProtocol` docstring, `save_dataset`'s `file` docs and the contributing guide ("Implementing new writers") would need updating.
2. **Sibling files** (`session_position.npz`, ...): fits the file contract and the precedent for multi-file outputs (NWB and DLC with multiple individuals), but `load_folder` keys become `session_position`, and files mix with unrelated ones in the parent dir.
3. **One `.npz` holding one all-variables `TsdFrame`** (with a `variable` metadata field): fits the file contract perfectly with no `register_writer` change, but loses `ds.attrs` (no sidecar convention for `load_file`), forces one dtype, and breaks the per-variable shape of the in-memory export (D-2).

**Related questions for that PR:**
- Whether to bring back a `split_individuals`-style option to choose the layout (e.g. one subfolder per individual, which `load_folder` walks).
- Sidecar content: whether to also include the movement version and the variable's dims/units.
- Whether the directory mode in `register_writer` ships as a separate commit or PR.

## 8. Questions for pynapple devs

1. **Dependency.** The export builds objects with pynapple (optional dependency) instead of a numpy-only `.npz`. OK with you? Is there any timeline for pynapple on conda-forge?
2. **Data model.** Does one wide `TsdFrame` per variable, with `individual`/`keypoint`/`space` column metadata, match how you'd want users to work in pynapple? Would your users expect one frame per variable, or one frame for everything? Any plans for pynapple#689 (`.sel`-style selection on metadata)?
3. **pynaviz.** Could skeleton mode resolve keypoints, individuals and spatial components from **column metadata** rather than parsing `_x`/`_y` column names? That would make it work with multi-individual frames and 3D, and make our column-naming choice (D-12) irrelevant.
4. **Skeleton representation.** Is per-column `parent` metadata the long-term choice, or an edge list (which can also express non-tree graphs)? Where should it live (metadata, sidecar, separate object)? We'd like to add a `skeleton=` option once this is settled.
5. **`nap.from_movement` (pynapple#617).** Proposal: movement owns the conversion (`to_pynapple_tsdframe` / `to_pynapple_dict`). If pynapple wants a container with a summary repr (like `MovementDataset`), `nap.from_movement` becomes a thin wrapper around `to_pynapple_dict`. One set of conversion rules, each side owning what it's good at. Does that work for you?
6. **Saving (for the follow-up PR).** Is a folder of `.npz` + `.json` sidecars idiomatic? Are arbitrary string keys in the sidecar fine, beyond `time`/`info`? Would you prefer sibling files or a single file instead (section 7)?

## 9. Out of scope / follow-ups

- Saving to disk (section 7).
- A `skeleton=` option (after Q8.4).
- Storing the time unit in DataArray attrs (movement-wide), then checking it in the exporter.
- A docs example combining tracking with ephys (needs a suitable public dataset).
- NWB route improvements: pynapple reading `ndx-pose` (pynapple#650), and movement storing extra variables in NWB.
- Exporting RoI events/occupancy as `IntervalSet`.
- Re-checking pynapple's conda-forge availability (for an eventual non-optional integration).

## 10. Process notes

- Message gviejo with section 8 once the draft PR is up.
- Discuss the saving options (section 7) with movement co-maintainers.
- PR 1127: close it with thanks and credit (e.g. `Co-authored-by`), or invite the contributor to test the new PR.
