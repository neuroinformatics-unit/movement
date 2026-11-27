# Design: exporting `movement` data to `pynapple`

- **Issue:** [movement#1098](https://github.com/neuroinformatics-unit/movement/issues/1098)
- **Branch:** `export-to-pynapple` (based on `main`)
- **Status:** in-memory export agreed; `to_pynapple_tsdframe` implemented, `to_pynapple_dict` next (2026-11-27). Saving to disk deferred to a follow-up PR (section 7).
- **Working notes / exploration history:** `export-to-pynapple.md`

## 1. Summary

This PR adds a direct, **in-memory** `movement` → `pynapple` export, owned by `movement` (as SpikeInterface owns `to_pynapple_tsgroup()`). It consists of two functions in a new module, `movement/io/to_pynapple.py`:

- `to_pynapple_tsdframe(da)` converts one `xr.DataArray` to one `TsdFrame`.
- `to_pynapple_dict(ds, variables=None)` converts an `xr.Dataset` to `{variable_name: TsdFrame}` by calling `to_pynapple_tsdframe` on each eligible variable.

**One individual is converted at a time**, matching the main use case: analysing an animal's behaviour alongside its neural recordings, which almost always come from a single animal per session. Each data variable becomes its own `TsdFrame`. The `keypoint` and `space` dimensions are stacked as columns named `<keypoint>_<space>` (e.g. `snout_x`), and each column also carries `keypoint` / `space` metadata, plus the individual's label. The conversion is generic over movement's canonical dimensions, so **derived variables** (speed, head direction, ...) convert as readily as tracked positions. Section 3 opens with why we chose this design. pynapple is an optional dependency (`movement[pynapple]`).

Writing to disk (a `save_dataset(..., target_software="pynapple")` writer) is **not** part of this PR. Users save with pynapple's own `obj.save()`, as SpikeInterface users do. The on-disk layout will be settled in a follow-up, after discussion with the pynapple devs and movement co-maintainers (section 7).

## 2. Background

### 2.1 The existing NWB route

There is already an indirect route (`to_nwb_file_object()` → `nap.NWBFile`), but it has limitations:

- Confidence scores are lost.
- One individual per file (an NWB constraint). We keep the same one-individual scope (D-3), but by choice, as it matches the use case.
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

Our design follows the same shape: in-memory, returning stock pynapple objects, labels carried as metadata, one unit of data (here, one individual) per call.

### 2.3 pynapple conventions

Checked in the pynapple 0.11.4 source:

- **NWB loader** (`io/interface_nwb.py`): each NWB `TimeSeries` becomes its own object, keyed by name. Columns are the components or channels of that one series (`x`/`y`/`z` for a `SpatialSeries`, or electrodes/ROIs with their tables as metadata). So the dominant convention is **one object per signal, columns = its components**.
- **Legacy loader** (`io/loader.py`): merges `CompassDirection` and `Position` into one `position` frame (`rx, ry, rz, x, y, z`). So frames that mix units have precedent, but only in the old API.
- **`compute_tuning_curves(data, features)`** treats every column of `features` as a feature dimension. A frame is used directly only when all its columns are wanted, so in practice users select columns first. Selection is unavoidable for multi-keypoint or multi-individual data in any design.

This supports one `TsdFrame` per variable (D-2) rather than one frame holding everything.

### 2.4 Related work

- [pynapple#617](https://github.com/pynapple-org/pynapple/pull/617) prototypes `nap.from_movement()` on the pynapple side. It returns a `MovementDataset(UserDict)` with an `NWBFile`-style summary repr, lazy building and per-individual nesting. It handles only `position`/`confidence`/`shape` and has no column metadata. Its per-individual column names (`f"{kp}_{sp}"`, e.g. `nose_x`) are the same as ours.
- [pynaviz#118](https://github.com/pynapple-org/pynaviz/pull/118) prototypes skeleton plotting. It expects a `TsdFrame` with `<kp>_x`/`<kp>_y` column pairs, plus optional per-column `parent` metadata for edges.
- [pynapple#689](https://github.com/pynapple-org/pynapple/issues/689) proposes `TsdFrame.sel(individual=..., keypoint=...)` over column metadata, motivated by movement data. Our data model fits it directly.
- [movement#1127](https://github.com/neuroinformatics-unit/movement/pull/1127) is an earlier community attempt (a single `TsdFrame` holding all variables, numpy-written, no metadata). We are not building on its data model.

## 3. Design (this PR)

### Why one individual and one variable per `TsdFrame`

A `TsdFrame` is a 2D table: time × columns, with optional per-column metadata. A movement data variable has up to three non-time dimensions (`individual`, `keypoint`, `space`), and a dataset holds several variables. Putting more than one individual, or more than one variable, into pynapple objects leaves three options, each with a real cost:

1. **Encode everything in column names**, e.g. `mouse1_snout_x` or `position_mouse1_snout_x`. Names get long and are **not automatically parseable**: labels can themselves contain `_` (`tail_base`, `mouse_1`), so `mouse_1_tail_base_x` can't be split back into its parts. Users would have to rely on metadata for any reliable selection, and only the first two metadata fields show when printing a frame.
2. **A tidy-like layout**, where the individual (or variable) is just another label to filter on, e.g. an `individual` metadata field that differs across columns. Every analysis then needs a selection step first: `compute_tuning_curves` treats every column as a feature, so a frame mixing individuals (or position with confidence and speed) is never usable as is. Mixing variables also mixes units and dtypes in one object.
3. **Nested outputs**, e.g. `{individual: {variable: TsdFrame}}` (as in pynapple#617's `MovementDataset`) or a list of objects per individual (as `to_nwb_file_object` returns). The return type then depends on how many individuals there are or on an option, which makes the API harder to use and to type.

Restricting each `TsdFrame` to **one individual and one variable** avoids all three:
- Column names are short, unique and parseable (`<keypoint>_<space>`), even when labels contain `_`.
- Metadata only holds what tells columns apart (`keypoint`, `space`), plus single labels such as the individual, with the same value in every column.
- The return type is fixed: one `TsdFrame` per DataArray. `to_pynapple_dict` is a convenience over that: always a flat dict keyed by variable name, never nested by individual.

Converting several individuals or variables stays a one-line comprehension over `.sel()` or data variables (see 3.3), so nothing is lost; it's just explicit.

**Derived variables are first-class.** The behavioural variables most often related to neural activity are usually *derived* from tracking, not the raw keypoints: head direction (head-direction cells), speed (speed cells), a centroid position (place and grid cells), angular velocity, distance to a partner animal. So the conversion must not be limited to `position`/`confidence`. It works on any DataArray with movement's canonical dimensions, whichever of `keypoint`/`space` it has. That's why the core function takes a DataArray (D-4): kinematics outputs convert directly, without adding them to a dataset first. The tests cover one derived variable per combination of dimensions (velocity, speed, head direction vector, forward vector angle). By contrast, the NWB route and pynapple#617 only carry `position`/`confidence`/`shape` (2.1, 2.4).

This also matches what pynapple users expect:
- **pynapple#617** already makes one frame per individual, with the same `<keypoint>_<space>` column names (`nose_x`).
- **pynapple's own loaders** give one object per signal, with columns as its components (2.3).
- **The main use case** is co-analysing an animal's behaviour with its neural recordings, which almost always come from one animal per session.

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
- Both functions are re-exported from `movement.io` (next to `load_dataset` and `save_dataset`), so users write `from movement.io import to_pynapple_tsdframe`. Importing `movement.io` does not import pynapple (tested). Nothing is registered with `save_dataset` in this PR.

Usage:

```python
import pynapple as nap
from movement.io import to_pynapple_dict, to_pynapple_tsdframe
from movement.kinematics import compute_speed

# {"position": TsdFrame, "confidence": TsdFrame, ...}
objs = to_pynapple_dict(ds.sel(individual="mouse1"))
pos = objs["position"]
pos[["snout_x", "snout_y"]]  # by name
pos[:, pos.keypoint == "snout"]  # or via metadata

# no need to add speed to ds first
speed = to_pynapple_tsdframe(
    compute_speed(ds.position.sel(individual="mouse1"))
)

# single-animal datasets (individual dim of size 1) need no selection
objs = to_pynapple_dict(ds_single)

# saving is pynapple's job
for name, obj in objs.items():
    obj.save(f"session/{name}.npz")  # later: nap.load_folder("session")
```

### 3.2 Mapping a DataArray to a `TsdFrame`

**Accepted input.** Only movement's canonical dimensions (`time`, `space`, `keypoint`, `individual`, from `ValidPosesInputs.DIM_NAMES`), with:
- a `time` dimension (required);
- at most one individual: a size-1 `individual` dimension is dropped automatically (single-animal datasets need no selection); more than one raises, pointing to `.sel(individual=...)`;
- no other dimensions (e.g. `view`): they raise, asking the user to select a single label first.

**Object type.** Always a `TsdFrame` of shape `(n_time, n_columns)`.
- The `keypoint` and `space` dimensions (whichever are present) are stacked into columns.
- A time-only array (e.g. `head_direction.sel(individual="mouse1")`) becomes a single-column `TsdFrame`, with the column named after `da.name` (or `Unnamed`).
- `Tsd` is not used, so the return type never depends on the input shape.
- `TsdTensor` is not used either (no metadata support, so labels would be lost).

**Column order** is keypoint → space, with `space` innermost, so consecutive columns form `x`, `y` (, `z`) pairs. The order is derived from `ValidPosesInputs.DIM_NAMES` (reversed); bboxes dims are a subset in the same order.

**Column names** are `<keypoint>_<space>` (e.g. `snout_x`), or the single stacked label (`snout` for confidence, `x` for bboxes position). They are always **unique and parseable**, even for keypoints containing `_` (`tail_base_x`): keypoint labels are unique and the space label is the last `_`-separated part. So users can select by name (`pos[["snout_x", "snout_y"]]`).

**Column metadata:** one field per stacked dimension (`keypoint`, `space`), holding each column's labels, followed by one field per canonical dimension with a **single label**, with the same value in every column: `individual` (from a size-1 dimension or `.sel(individual="mouse1")`), or e.g. `keypoint` after `.sel(keypoint="snout")`. So the individual's label is always kept. Single labels of non-canonical dimensions (e.g. `view` after `.sel(view="cam0")`) are not carried over. All values are strings. Fields are accessible as attributes (`pos.keypoint`), used for selection via `pos[:, pos.keypoint == "snout"]`, and fit pynapple#689-style `.sel`. The fields that vary across columns come first because pynapple's repr only shows the first two fields; the others are always available in `.metadata`. A size-1 `keypoint` *dimension* (from `.sel(keypoint=["snout"])`) stays a column dimension (`snout_x`), unlike `individual`.

Example: a 2D poses dataset with keypoints `snout`, `tail_base`, plus a user-added `speed` (dims `time, keypoint, individual`), for one individual:

| Variable | `to_pynapple_dict(ds.sel(individual="mouse1"))` |
|---|---|
| `position` | 4 cols: `snout_x`, `snout_y`, `tail_base_x`, `tail_base_y` |
| `confidence` | 2 cols: `snout`, `tail_base` |
| `speed` | 2 cols (as confidence) |
| `head_direction` (time, individual) | 1 col: `head_direction` |

Bounding boxes work the same way. `position`, `shape` and `confidence` each become one frame, and `shape` keeps movement's `space` labels (`x`/`y`), not `width`/`height`.

`position` frames always have the `<kp>_x`/`<kp>_y` layout that the pynaviz skeleton mode parses today, and match the per-individual frames of pynapple#617.

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
- **One individual per dataset**, by the same rule as for DataArrays: a size-1 `individual` dimension is dropped, and more than one raises once (checked up front, not per variable), pointing to `.sel(individual=...)`. Docstrings document the recipe for converting several:

  ```python
  {ind: to_pynapple_dict(ds.sel(individual=ind)) for ind in ds.individual.values}
  ```

  Inter-individual measures (e.g. `compute_pairwise_distances(..., "individual", ...)`) are already reduced to `time`-only arrays, so they convert fine. Add them to the selected dataset, or convert them directly with `to_pynapple_tsdframe`.

### 3.4 Packaging

- `pyproject.toml`: a new optional extra `pynapple = ["pynapple>=0.8"]`, plus pynapple in the dev/test and docs groups.
- conda-forge: no recipe change, since optional dependencies aren't listed. Conda-only users `pip install pynapple` to use the export. pynapple itself is not on conda-forge yet.
- API docs: the module is picked up automatically by `docs/make_api.py` (only packages need a `PACKAGE_MODULES` entry).

### 3.5 Docs and tests

**Docs:**
- API reference for the new module (automatic).
- IO guide: a short "Exporting to pynapple" section (in-memory conversion, then `obj.save()`).
- Docstrings: the seconds assumption with the `assign_coords` recipe, the one-individual rule, and the recipe for converting several individuals.
- A **small gallery example** covering:
  - `to_pynapple_dict` and `to_pynapple_tsdframe`, the latter on a derived variable (e.g. speed)
  - `variables=`
  - selecting an individual via `.sel()`, and converting several
  - name-based and metadata-based column selection
  - saving with `obj.save()` and loading back with `nap.load_folder`
  - the `set_info(parent=...)` recipe for pynaviz skeletons

  There's no ephys data in this PR, so no new public dataset is needed.

**Tests:**
- Mapping (`to_pynapple_tsdframe`, done): poses and bboxes, 2D and 3D. A shared helper checks every column's values, timestamps, name (`_`-joined labels), metadata (including single labels) and attribute access to metadata fields.
- Derived variables covering each combination of `keypoint`/`space` dimensions: velocity, speed, head direction vector, forward vector angle (time-only).
- Time-only arrays: named after the DataArray, or `Unnamed`.
- Single labels: a single keypoint label (`x`, `y` columns), a size-1 keypoint dimension (`left_x`, `left_y`), and a size-1 `individual` dimension (same result as selecting it).
- Errors: no `time` dimension, more than one individual, non-canonical dims (e.g. `view`).
- `variables=`, including errors and skip warnings.
- `ImportError` when pynapple is missing (monkeypatched), and importing `movement.io` doesn't import pynapple (in a subprocess).

## 4. Decisions (this PR)

**Firm** means the decision is settled unless new information appears. **Provisional** means we chose it to unblock, and reviewers and pynapple devs are welcome to challenge it.

| # | Decision | Rationale | Status |
|---|---|---|---|
| D-1 | In-memory export only, building objects with pynapple as an **optional**, lazily imported dependency. Saving is left to pynapple's `obj.save()`. | Matches the SpikeInterface precedent. pynapple validates its own objects. Defers the folder-vs-file question (section 7) without blocking the export. | Firm |
| D-2 | **One `TsdFrame` per data variable**, with `keypoint`/`space` stacked as columns and labelled via column metadata. | Avoids long, unparsable names, a tidy layout needing selection first, or nested outputs (see "Why one individual and one variable per `TsdFrame`"). Matches pynapple's one-object-per-signal convention (2.3). Units not mixed within an object. Matches pynapple#689. | Firm |
| D-3 | **One individual at a time.** A size-1 `individual` dim is dropped (its label kept in metadata); more than one raises, pointing to `.sel(individual=...)`. No `split_individuals` parameter; converting several is a documented one-line recipe. | Same reasoning as D-2 (see "Why one individual and one variable per `TsdFrame`"). Matches pynapple#617's per-individual frames and pynapple users' expectations. The integration's purpose is co-analysing behaviour with neural data, recorded from one animal per session in almost all cases. Short, unique names; pynaviz-ready frames; schema never depends on individual count. Relaxing later is non-breaking, restricting later would be breaking. Inter-individual measures are `time`-only and convert fine. Cost: a partner animal's trajectory needs its own frame. | Firm for v1 |
| D-4 | Two functions: `to_pynapple_tsdframe(da)` as the core, `to_pynapple_dict(ds, variables=None)` as a loop over it. Not one function with an input-dependent return type. | Derived variables (speed, head direction, ...) are what's most often compared to neural data, and kinematics/filtering functions return DataArrays, so they convert directly (see "Derived variables are first-class"). Two names are clearer than `@overload`s. | Firm |
| D-5 | Always `TsdFrame`, never `Tsd`. Time-only arrays become a single column named after `da.name`. | The return type matches the function name and never depends on input shape. | Firm |
| D-6 | `to_pynapple_dict` returns a plain `dict`, not a container class. | A pynapple-style session view belongs in pynapple. A movement copy would drift and add a dependency. | Firm |
| D-7 | Time is assumed to be in seconds, documented, not checked. | DataArrays don't carry `time_unit` today. Proper fix planned separately (time unit in array attrs). | Firm for v1 |
| D-8 | Time support `[t[0], t[-1]]`. | pynapple's own default. | Firm |
| D-9 | Convert all eligible variables by default, with an optional `variables=` filter. | Preserves derived variables (a stated limitation of the NWB route) while letting users opt out. | Firm |
| D-10 | Skeletons out of scope for v1. Document the `set_info(parent=...)` recipe. | movement datasets don't store skeletons, and the pynaviz representation isn't settled (pynaviz#118 open). | Firm for v1 |
| D-11 | `pynapple>=0.8`. | Column metadata was introduced in 0.8.0. Save/load round-trip verified on 0.8.0, 0.8.5, 0.9.0 and 0.11.4, and on Python 3.14. CI only tests the latest version. | Firm |
| D-12 | Column names `<keypoint>_<space>` (or the single stacked label), plus `keypoint`/`space` metadata. Non-canonical dims (e.g. `view`) raise. | Names are unique and parseable even when keypoints contain `_`, so both name- and metadata-based selection are reliable. Same as pynapple#617 and the pynaviz layout. | Firm |
| D-13 | Bboxes `shape` keeps `x`/`y` labels. | One generic rule, faithful to movement's data model. | **Provisional** |
| D-14 | Module `movement/io/to_pynapple.py`, with both functions re-exported from `movement.io`. | `from movement.io import to_pynapple_tsdframe` is the short public path, next to the other entry points (`load_dataset`, `save_dataset`); these functions are pynapple's entry point while no `save_dataset` writer exists. Not in `movement.io.save`: that module is the file-saving machinery (`save_dataset`, writer registry), and this conversion doesn't save. Not in `save_poses`: it works for bboxes too. Module not named `pynapple.py` (like `nwb.py`): `from movement.io import pynapple`, the IO guide's import pattern, would shadow the pynapple package in users' scripts. The module name mirrors SpikeInterface's `exporters/to_pynapple.py`; the follow-up writer can live in it too. | Firm |

## 5. Alternatives considered

- **numpy-only `.npz` writer** (originally agreed first pass, also PR 1127): needs no dependency, but means hand-writing pynapple's format and gets no validation at write time. Superseded by D-1.
- **A single `TsdFrame` for everything** (PR 1127): works technically (verified: 14 columns with a `variable` metadata field, round-trips through `save`/`load_file`), but goes against pynapple's one-object-per-signal convention, mixes units and dtypes, and always needs a `variable` filter. Revisited for the on-disk format in section 7.
- **Stacking all individuals (and other dims, e.g. `view`) in one frame**, with names like `mouse1_snout_x`: lossless for multi-animal data, but names get long and can't be parsed automatically for labels containing `_`, metadata grows beyond what pynapple's repr shows, and frames don't match the pynaviz layout. Rejected in favour of one individual at a time (D-3, D-12).
- **A tidy-like layout**, with the individual (or variable) as a metadata field that varies across columns: every analysis would need a selection step first, and mixing variables mixes units and dtypes. Rejected for the same reasons (see "Why one individual and one variable per `TsdFrame`").
- **`split_individuals=` parameter** (pynapple#617 nests per individual): redundant with `.sel()` and gives a flag-dependent return type (D-3).
- **Auto-splitting like `to_nwb_file_object`**, which returns one `NWBFile` per individual (a list, or a single file for one individual): needed there because NWB holds one subject per file. Here it would mean a nested, count-dependent return type. An explicit error plus a one-line recipe is simpler.
- **One function accepting Dataset or DataArray:** the return type would depend on the input type. Replaced by two functions (D-4).
- **`MovementDataset`-style container** (pynapple#617): see D-6.
- **`TsdTensor` per variable:** keeps the n-D shape but loses all labels.
- **Keeping the converter on the pynapple side only** (`nap.from_movement`): we prefer domain packages exporting towards the integrating package (the SpikeInterface precedent), and movement already maintains many format converters.

## 6. Open questions (this PR, to settle during implementation)

1. **Boolean variables** (e.g. RoI "inside" results): `np.issubdtype(bool, np.number)` is `False`. Export them as-is, cast them, or skip them?

Settled during implementation:
- **Extra dimensions** (e.g. `view`, user-defined dims) raise; users select a single label first (D-12).
- **Duplicate or ambiguous column names** can't occur: with only `keypoint`/`space` stacked, names are unique and parseable (D-12).
- **Unnamed time-only DataArrays** get a column named `Unnamed`. Users rename first (`da.rename("speed")`), since pynapple reserves `columns` after creation. Most `movement.kinematics` outputs are named; `compute_kinetic_energy` (returns `None`) and `compute_time_derivative`/`compute_norm` (keep the input name) should be fixed separately.
- **Scalar coordinates** of canonical dims (e.g. `individual="mouse1"` after `.sel()`, or a squeezed size-1 `individual` dim) become constant metadata fields. Non-canonical ones are not carried over.

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
2. **Data model.** Does one `TsdFrame` per variable, for one individual at a time, with `<keypoint>_<space>` columns and `keypoint`/`space` metadata, match how you'd want users to work in pynapple? Is one individual at a time right for your users (we assume neural data comes from one animal per session)? Any plans for pynapple#689 (`.sel`-style selection on metadata)?
3. **pynaviz.** Our `position` frames match the `<kp>_x`/`<kp>_y` layout skeleton mode parses today. Could it also use the `keypoint`/`space` **column metadata**, which would make it robust to naming and support 3D (`_z`)?
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
- **Open question:** should `to_nwb_file_object` also be re-exported from `movement.io`, for symmetry with `to_pynapple_tsdframe`? Today it is only reachable as `movement.io.save_poses.to_nwb_file_object`. Separate change, not in this PR.

## 10. Process notes

- Message gviejo with section 8 once the draft PR is up.
- Discuss the saving options (section 7) with movement co-maintainers.
- PR 1127: close it with thanks and credit (e.g. `Co-authored-by`), or invite the contributor to test the new PR.
