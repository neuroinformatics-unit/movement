# Handling DeepLabCut `uniquebodyparts` in movement

Design document for [issue #977](https://github.com/neuroinformatics-unit/movement/issues/977).

## Background

DeepLabCut multi-animal projects can define two types of keypoints:

- **`multianimalbodyparts`**: tracked per individual (e.g. `leftwing`,
  `rightwing`, `middle` for each bird).
- **`uniquebodyparts`**: appear once per frame, not tied to any tracked
  individual (e.g. fixed landmarks like `boatBL`, `boatBR`, `boatTip`).

In DLC output files, `uniquebodyparts` are stored under a pseudo-individual
called `"single"`. The resulting DataFrame has a **ragged multi-index**:
not every individual has every bodypart.

```
bird1  / leftwing   ✓      bird1  / boatBL    ✗ (does not exist)
bird1  / rightwing  ✓      bird1  / boatBR    ✗
bird1  / middle     ✓      bird1  / boatTip   ✗
...                        ...
single / leftwing   ✗      single / boatBL    ✓
single / rightwing  ✗      single / boatBR    ✓
single / middle     ✗      single / boatTip   ✓
```

## The current bug

`from_dlc_style_df` (in `load_poses.py`) extracts unique individual and
bodypart names, then does a blind `numpy.reshape` assuming every individual
has every bodypart (a full rectangular grid).

When the grid is ragged, the reshape **silently corrupts data** if the
total element count happens to divide evenly. For example, with the test
fixture included in this PR
(5 birds × 3 bodyparts + 1 "single" × 3 uniquebodyparts):

| | Actual | Expected by reshape |
|---|---|---|
| Columns | 18 combos × 3 coords = **54** | 6 individuals × 6 bodyparts × 3 coords = **108** |
| Total elements (10 rows) | 10 × 54 = **540** | 5 × 108 = **540** |

NumPy reshapes without error. The result:

- **Frame count halved**: 10 raw frames become 5.
- **Data scrambled**: values from different frames/individuals/keypoints
  are mixed together.
- **Invalid combinations get garbage**: `bird1/boatBL` (shouldn't exist)
  gets real-looking values from other individuals' data.

This is a silent data corruption bug. Our docs claim we error out on
non-shared keypoints ([PR #658](https://github.com/neuroinformatics-unit/movement/pull/658)),
but #658 only added a documentation note — no validation code was added.
Whether we get an error or silent corruption is entirely down to whether
the arithmetic happens to divide evenly.

## Is this DLC-specific?

**Yes.** We investigated all supported source software:

| Software | Multi-individual? | Ragged keypoints possible? | Why |
|---|---|---|---|
| DeepLabCut | Yes | **Yes** | `uniquebodyparts` stored under `"single"` pseudo-individual |
| SLEAP | Yes | No | One skeleton per project, enforced by SLEAP ([issue #2005](https://github.com/talmolab/sleap/issues/2005)) |
| LightningPose | No (single-individual) | No | One skeleton, one individual |
| Anipose | No (single-individual per file) | No | Inherits from upstream DLC |
| NWB (ndx-pose) | No (single-individual per `PoseEstimation`) | No | Different keypoint sets live in separate `PoseEstimation` containers |
| VIA-tracks | Yes | N/A | Bounding boxes only, no keypoints |

No other mainstream pose estimation tool supports heterogeneous skeletons
in one project. The standard workaround across tools (SLEAP, DLC) for
tracking different entity types is a superset skeleton with NaN for
inapplicable keypoints — which already produces a rectangular grid.

## Proposed solutions

### Option 1: NaN-fill to a rectangular grid

Before reshaping, reindex the DataFrame to the full cartesian product of
`individuals × bodyparts × coords`, filling missing combinations with NaN.
The existing reshape then works correctly.

```python
ds = load_dataset("file.h5", source_software="DeepLabCut")
ds.position.sel(individual="bird1", keypoint="boatBL")  # all NaN
ds.position.sel(individual="bird1", keypoint="leftwing")  # real data
```

**Pros:**

- Minimal code change (reindex before reshape, ~5 lines).
- `load_dataset` returns one `xr.Dataset` — no API changes.
- All data accessible in one object; users filter with `.sel()`.
- NaN semantics are correct: "this combination doesn't exist".
- Downstream kinematics safely produce NaN for invalid combinations.

**Cons:**

- Semantically misleading: `bird1/boatBL` appears as a real
  (individual, keypoint) pair that happens to be missing.
- Every `ds.keypoint.values` shows bodyparts that don't apply to most
  individuals — potentially confusing for exploratory analysis.
- Some memory overhead from NaN padding (negligible in practice).
- No way to distinguish "this keypoint was tracked but lost" (legitimate
  NaN) from "this combination never existed" (structural NaN).

### Option 2: Kwarg to select bodypart type (recommended)

Detect the ragged structure. By default, load only the
`multianimalbodyparts` (the bodyparts shared across tracked individuals)
and warn that `uniquebodyparts` were dropped. A kwarg switches to loading
only the `uniquebodyparts`.

```python
# Default: multi-animal bodyparts only (5 individuals × 3 keypoints)
ds = load_dataset("file.h5", source_software="DeepLabCut")

# Unique bodyparts only (1 individual × 3 keypoints)
ds_unique = load_dataset(
    "file.h5", source_software="DeepLabCut",
    unique_bodyparts_only=True,
)
```

**Pros:**

- Each returned dataset is internally consistent — no NaN padding, no
  phantom combinations.
- Default gives users exactly the tracked animals they expect.
- Warning makes the behavior transparent and discoverable.
- `load_dataset` still returns one `xr.Dataset` — no protocol changes.
- The kwarg passes through via existing `**kwargs` in `load_dataset`.
- Contained to the DLC loader — no changes to the unified loading
  infrastructure.

**Cons:**

- Two calls needed to load all data from one file.
- The kwarg is DLC-specific, slightly polluting the unified interface.
- Users must read the warning to discover `uniquebodyparts` exist.

**Implementation scope:**

- `from_dlc_style_df`: detect ragged multi-index, split DataFrame, add
  kwarg.
- `from_dlc_file` / `_ds_from_lp_or_dlc_file`: forward the kwarg.
- No changes to `load.py`, `LoaderProtocol`, or any other loader.

### Option 3: Flexible `LoaderProtocol` (return multiple datasets)

Change `LoaderProtocol` to allow returning multiple datasets (e.g. as a
`dict[str, xr.Dataset]`). The DLC loader would return both the
multi-animal and unique datasets in one call.

**Pros:**

- Single call returns all data.
- Generalises cleanly if other software ever needs multi-dataset returns.
- Clean semantic separation between dataset types.

**Cons:**

- The protocol change itself is trivial, but the single-`xr.Dataset`
  return type is assumed by every consumer of `load_dataset`:
  - `load_dataset` (accesses `.attrs` on the result directly)
  - `load_multiview_dataset` (passes results to `xr.concat`)
  - `napari/loader_widgets.py` (passes result to `ds_to_napari_layers`)
  - `sample_data.py` (accesses `.attrs` on the result)
  - Every downstream user script
- All of these would need conditional handling for single vs. multi
  return, or `load_dataset` would need to flatten internally (defeating
  the purpose).
- No other supported software needs this — the generalisation is
  speculative.
- Higher risk of subtle breakage across the codebase.

## Recommendation

**Option 2 (kwarg)** is the best fit. It solves the problem cleanly, stays
contained to the DLC loader, and doesn't require changes to shared
infrastructure for a problem that is DLC-specific. If another source
software ever produces ragged data, we can revisit Option 3 then — but
current evidence suggests that won't happen.

## Test fixture

A pytest fixture `dlc_csv_file_with_uniquebodyparts` has been added to
`tests/fixtures/files.py` (on branch `handle-unique-keypoints`). It
generates a minimal DLC-style CSV with the ragged structure described
above: 5 birds with 3 multi-animal bodyparts each, plus 3
`uniquebodyparts` under `"single"`, with 10 data rows.

The helper `_build_dlc_df_with_uniquebodyparts()` can also be called
directly in tests to get the raw DataFrame without the CSV round-trip.
