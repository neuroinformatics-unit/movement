# Proposal for Drag-and-drop loading of tracked data in the movement napari plugin

## Description

**What is this PR?**

This PR is a detailed plan on how to implement the drag-and-drop functionality requested in [#960](https://github.com/neuroinformatics-unit/movement/issues/960).

The plan has been generated discussing with Claude Code but is meant to be read by humans. The idea is to discuss it with @neuroinformatics-unit/movement-active-devs before we implement it.

**Why is the proposed feature needed?**

Getting tracked data into the viewer today requires opening the movement widget, picking the source software from a combo box, setting fps and browsing for a file ([movement/napari/loader_widgets.py](movement/napari/loader_widgets.py)). The issue is discussed in detail in [#960](https://github.com/neuroinformatics-unit/movement/issues/960).

For videos an images, users can already drag-and-drop the file on the canvas, which is super convenient ([docs/source/user_guide/gui.md:43](docs/source/user_guide/gui.md#L43)). It would be great if we supported the same for pose track files too.

Since PR [#920](https://github.com/neuroinformatics-unit/movement/pull/920) the backend can already infer the `source_software` given a file without asking the user. So maybe we are well positioned to implement this feature now.

**References**
* This PR proposes an implementation that addresses [#960](https://github.com/neuroinformatics-unit/movement/issues/960)
* Relates indirectly to [#959](https://github.com/neuroinformatics-unit/movement/issues/959) (netCDF via `load_dataset`)
* Relates indirectly to [#896](https://github.com/neuroinformatics-unit/movement/pull/896) (dynamic loader UI exposing relevant kwargs).


## Key aspects of suggested implementation

### napari Reader Contribution
We will need to implement a napari [Contribution](https://napari.org/stable/plugins/technical_references/contributions.html) of type `reader`.

A Contribution allows us to extend napari functionality. The `reader` type allows us to extend the file reading functionalities specifically. Contributions are defined in the plugin manifest (the `movement/napari/napari.yaml` file). Defining a reader involves defining filename patterns that will trigger it (see example [here](https://napari.org/stable/plugins/technical_references/contributions.html#readercontribution)).

For the filenames that survive the pattern test, napari will call the function returned by the `command` defined in the plugin manifest (often called `napari_get_reader()`). This function returns a callable: a **reader function** for the given input path. The reader is meant to be cheap: the napari docs describe it as lightweight validation without loading the full file content (e.g. peeking at a header rather than loading fully). `napari_get_reader()` should return a reader function or `None` to decline (and then napari moves on to reader Contributions from other plugins).

The reader function returned by `napari_get_reader()` should take the path(s) and return a list of `LayerData` tuples `(data, attributes, layer_type)` where `layer_type` is `'points'`, `'tracks'`, `'shapes'`, etc. (defaults to `'image'` if omitted).

On conflicts (i.e. when the filename patterns from several plugin readers match a single file), napari presents a reader-choice dialog with a "remember this choice" option that gets written to user settings.

Note that we can restrict what passes to `napari_get_reader()` by file suffix, but not by source software. This is because the mapping suffix to software isn't 1:1 (e.g. there are multiple source software that produce `.csv` files, see [the guide](https://movement.neuroinformatics.dev/latest/user_guide/input_output.html#supported-third-party-formats)).

napari docs provide a [Readers Contribution guide](https://napari.org/stable/plugins/building_a_plugin/guides.html#readers-contribution-guide).

### Layer wiring

A reader contribution is registered by npe2 from `napari.yaml` at install time. This means it runs whether or not any `movement` widget has ever been instantiated.

The reader path returns `(data, attributes, layer_type)` tuples that napari turns into layers directly, without going through `DataLoader._add_points_layer` / `_add_tracks_layer`, which is where the **per-layer** wiring happens ([:365-370](movement/napari/loader_widgets.py#L365-L370), [:396](movement/napari/loader_widgets.py#L396)). The **viewer-level** connections are separately in `connect_viewer_callbacks` ([layer_wiring.py:44](movement/napari/layer_wiring.py#L44)). The reader can call those itself. So we will need to run the per-layer wiring steps explicitly.


> [!NOTE]
> **The following assumes `fix/layer-wiring-lifetime`.**
> That branch moves every sync callback out of `DataLoader` and into
> module-level functions in
> [movement/napari/layer_wiring.py](movement/napari/layer_wiring.py), together
> with the metadata keys ([:32-35](movement/napari/layer_wiring.py#L32-L35)).
> The viewer-level connections are made by `connect_viewer_callbacks(viewer)`
> ([:44](movement/napari/layer_wiring.py#L44)).


These are the wirings that would need to be manually wired during drag-and-drop:
* three things on the Points layer — connect `events.data`, set `editable`, and resolve `TRACKS_LAYER_KEY` to the companion Tracks object
* one `connect_viewer_callbacks` call.

Everything else rides along in the layer tuple's `metadata`. For a more detailed description, see the collapsed table below.


<details>
<summary>Table: per callback details </summary>

| function (`layer_wiring.py`) | needs manual wiring on drop? |
|---|---|
| `on_points_data_changed` | **Yes** to`layer.events.data.connect(...)`, but it is now a plain function, so the connect is a one-liner with no widget involved. |
| `sync_tracks_layer` | **Yes, indirectly** — no connection of its own, but it reads `points_layer.metadata[TRACKS_LAYER_KEY]` ([:208](movement/napari/layer_wiring.py#L208)), which holds a *layer object* and so can't be put in a `LayerDataTuple`. Must be resolved after both layers exist. |
| `remove_from_tracks_layer` | **Yes, indirectly** — same `TRACKS_LAYER_KEY` reference ([:232](movement/napari/layer_wiring.py#L232)). Resolved by the same step. |
| `set_tracks_layer_data` | No — a pure helper called by the two functions above; nothing to wire. |
| `set_point_symbol_by_edited` | No — `symbol` *is* a `Points.__init__` kwarg, so the edited-point symbols can be pre-computed into the tuple's `attributes` instead of being applied post-creation as they are today ([loader_widgets.py:368](movement/napari/loader_widgets.py#L368)). |
| `frame_axis_is_sliced` | No — a pure query, now taking `viewer` explicitly ([:123](movement/napari/layer_wiring.py#L123)). But its *result* has to be applied once as `layer.editable` at insert time ([loader_widgets.py:364](movement/napari/loader_widgets.py#L364)), and `editable` is **not** a `Points.__init__` kwarg, so that one assignment is manual. |
| `update_points_layers_editable` | No — connected to `viewer.dims.events.order`/`ndisplay` inside `connect_viewer_callbacks` ([:66-67](movement/napari/layer_wiring.py#L66-L67)) and it re-scans *all* layers carrying `POINTS_LAYER_KEY`. Free for dropped layers, provided the reader sets `metadata[POINTS_LAYER_KEY] = True` — but it only fires on a dims change, hence the one-off `editable` assignment in the row above. |
| `update_frame_slider_range` | No — connected to `viewer.layers.events.inserted`/`removed` inside `connect_viewer_callbacks` ([:57-60](movement/napari/layer_wiring.py#L57-L60)), so it fires on the drop itself. |
| `connect_viewer_callbacks` | **Yes, but trivial** — the reader calls `connect_viewer_callbacks(napari.current_viewer())` before returning. Idempotent, so a drop with the widget already open is a no-op, and a drop without it wires the viewer anyway. |

</details>


There is an additional function worth running manually too for parity with the Load button: `_set_initial_state` ([loader_widgets.py:322](movement/napari/loader_widgets.py#L322)), which puts the slider at frame 0 and makes the Points layer active.


### What files are drag-and-droppable?

We rely on `load_dataset` for the drag-and-drop, so what is droppable is what the loader registry supports, which is not the same as what the widget's combo box offers:

|                             | `load_dataset` | combobox |
|---|---|---|
| DLC, LP, SLEAP, VIA-tracks  | ✅ | ✅ |
| Anipose, NWB                | ✅ | ❌ |
| movement `.nc`              | ❌ (until #959) | ✅ |

* So Anipose and NWB files will be droppable, but not selectable through the
  form widget yet.
* ROI `.geojson`/`.json` drops are out of scope.




## Detailed implementation

1. Define `napari_get_reader` in XXXX.py :
   ```python
   def napari_get_reader(path: str | list[str]) -> ReaderFunction | None:
    """Return a reader for movement-supported files, else None."""
    IF FILE IN LIST_SUPPORTED_FILES:
      return read_dataset
   ELSE:
      None

   ```
   - Do **not** validate content in `napari_get_reader`. It runs for every candidate drop  and should be very lightweight.

2. Define readers in XXXX.py that return `(data, attributes, layer_type)`:
   A reader loads a validated movement dataset, and transforms it to the data that napari needs.
   ```python
   def read_dataset(..):
      """load a validated movement dataset, and transform it to the data that napari needs."""
      IF FILE_EXTENSION IS .NC:
          # load ds from nc file
         ds = load_dataset_from_nc(..)
         # validate ds from nc
         ds = validate(ds,...)
      ELSE:
         ds = load_dataset(..)
      return ds_to_layer_data_tuples(ds, Path(path).name)
   ```
   - `load_dataset_from_nc` won't be needed after PR #959, so maybe it can be defined in the same file as `read_dataset`.
   - **`fps=None`** for dropped files. Users that need the data to show in seconds can use the form widget.
   - **Loader kwargs are not settable on drop**: default values are used. Exposed kwargs can be edited by the users in the widget form (autopopulating that form from the dropped file is future work, see below).

3. Define `ds_to_layer_data_tuples`
   - Can be factored out from existing implementation
   - Returns `(data, attributes, layer_type)`

4. Wire up the layers
   The reader function returns the data required by napari to build the layers, but their syncing logic is not implemented. We need a `wire_movement_layers(viewer, event)` in `layer_wiring.py`, connected to `viewer.layers.events.inserted` by `connect_viewer_callbacks` — which the reader calls itself, so it works with the widget open or closed.

5. Add reader Contribution to
   [movement/napari/napari.yaml](movement/napari/napari.yaml). Set filename pattern as `["*.h5", "*.csv", "*.slp", "*.nwb", "*.nc"]`
   ```yaml
   commands:
      - id: movement.get_reader
         python_name: movement.napari.reader:napari_get_reader
         title: Open tracked data with movement
   readers:
      - command: movement.get_reader
         filename_patterns: ["*.h5", "*.csv", "*.slp", "*.nwb", "*.nc"]
         accepts_directories: false
   ```
   - Question: can we avoid the duplication in filename_patterns? I think no, because npe2 manifests are static, so the patterns are hard-coded — and suffix alone cannot separate a DLC `.h5` from a SLEAP one, so the patterns have to come from `get_supported_source_software()` rather than from the source software. We can add a test asserting the list is equal to `get_supported_source_software()` union `{".nc"}` so the two can't silently diverge

6. Tests
   - Can we check that all current functionality works if the data is loaded via drag-and-drop?

7. Documentation updates
   - The drag-and-drop functionality should be mentioned in the `movement` guide.







--------


## Step 1 — Extract layer construction into a reusable, viewer-free function

**New file: `movement/napari/layers.py`** (could also fold into `convert.py`).

```python
def ds_to_layer_data_tuples(
    ds: xr.Dataset, name_suffix: str
) -> list[FullLayerData]:
    """Build napari (data, meta, layer_type) tuples from a movement dataset."""
```

It absorbs, near-verbatim, the viewer-free parts of:

- `DataLoader._format_data_for_layers` ([:238-262](movement/napari/loader_widgets.py#L238-L262)) —
  `ds_to_napari_layers`, the `data_not_nan` mask, the `position_is_nan` property.
- `_set_common_color_property` ([:315](movement/napari/loader_widgets.py#L315))
  and `_set_text_property` ([:334](movement/napari/loader_widgets.py#L334)).
- The style/kwargs assembly in `_add_points_layer` ([:356](movement/napari/loader_widgets.py#L356)),
  `_add_tracks_layer` ([:537](movement/napari/loader_widgets.py#L537)),
  `_add_boxes_layer` ([:562](movement/napari/loader_widgets.py#L562)) — the
  existing `PointsStyle`/`TracksStyle`/`BoxesStyle` `.as_kwargs()` calls and the
  `metadata` dicts, unchanged.

Two changes make the metadata expressible before layers exist:

1. `TRACKS_LAYER_KEY` currently holds a *layer object*, set in `_add_tracks_layer`
   ([:559](movement/napari/loader_widgets.py#L559)). Add
   `TRACKS_LAYER_NAME_KEY = "movement_tracks_layer_name"` which the pure function
   can set (`f"tracks: {name_suffix}"`). Step 4's wiring resolves it to the
   object under the existing `TRACKS_LAYER_KEY`, so `_sync_tracks_layer` and
   `_remove_from_tracks_layer` are untouched.
2. `editable` is **not** a `Points.__init__` kwarg (verified against the pinned
   napari 0.6.6), so it stays a post-creation assignment in Step 4. Same for
   `_set_point_symbol_by_edited` ([:426](movement/napari/loader_widgets.py#L426)).

`DataLoader._on_load_clicked` then loads the dataset (branching unchanged),
calls `ds_to_layer_data_tuples`, and adds each via
`self.viewer.add_layer(Layer.create(*tup))` (`napari.layers.Layer.create` is
public API).

## Step 2 — Extract the netCDF loading path so the reader can use it

`load_dataset` cannot open `.nc` files — there is no registered loader for it (that is #959) — so the reader needs a netCDF branch. The widget
already has one, in `DataLoader._load_netcdf_file`
([:272-313](movement/napari/loader_widgets.py#L272-L313)). Move its body to a
module-level function in `movement/napari/layers.py`:

```python
def load_movement_netcdf(path) -> xr.Dataset:
    """Open a movement netCDF file, raising ValueError if unusable."""
```

The logic is unchanged — `xr.open_dataset` → `rename_legacy_dimensions` →
`ds_type` must be `poses` or `bboxes` → `ValidPosesInputs.validate` /
`ValidBboxesInputs.validate` — with one difference: it **raises** instead of
`show_error(...); return None`, leaving the caller to decide how to report.
The widget wraps it and calls `show_error` with the same messages, so its
user-facing behaviour and `test_data_loader_widget.py`'s netCDF error cases are
untouched; the reader reports through its own error path (Step 3).

This one function is also the seam that disappears when #959 lands: the reader's
`.nc` branch collapses to a plain `load_dataset(path)` call and the helper is
deleted.

Note this deliberately does *not* validate third-party datasets. Those come out
of `load_dataset` already built through `ValidPosesInputs` / `ValidBboxesInputs`
(which is where `ds_type` is set,
[datasets.py:470](movement/validators/datasets.py#L470)), so re-checking them
would be a no-op. Whether the GUI's stricter-than-`load_dataset` requirements
should nevertheless be enforced at the conversion layer, for every dataset, is a
#959 design question rather than something this PR needs to settle — see
discussion point 1.

## Step 3 — The reader contribution

**New file: `movement/napari/reader.py`**

```python
SUPPORTED_SUFFIXES = set().union(*get_supported_source_software().values()) | {".nc"}

def napari_get_reader(path: str | list[str]) -> ReaderFunction | None:
    """Return a reader for movement-supported files, else None."""
```

- Normalise `path` to a list and return a reader function in essentially every
  case. The manifest's `filename_patterns` already gate on exactly
  `SUPPORTED_SUFFIXES`, so a second suffix check here is almost dead code — the
  only case it catches is a *mixed* multi-file drop, where napari passes a list
  in which some paths match the patterns and some don't; return `None` for that.
  Deliberately do **not** use `None` as a general "can't read this" signal:
  inside napari it is not a graceful fall-through. For npe2 plugins the
  reader-choice dialog is built from `filename_patterns` alone
  (`get_potential_readers` → `pm.iter_compatible_readers`), *before* the hook
  runs, so declining neither hides us from the dialog nor hands off to another
  plugin — if the user picked movement, they get a `ReaderPluginError`
  ("was selected to open …, but returned no data") instead of our own message.
  Everything we can't load is therefore handled *inside* the reader function,
  via `show_error` plus the `[(None,)]` sentinel (see the error bullet below).
- Do **not** validate content in `napari_get_reader`. It runs for every candidate
  drop and `ValidVIATracksCSV` does a full `pd.read_csv`. Suffix matching only;
  inference happens in the reader function.
- Reader function, per path: `.nc` → `load_movement_netcdf(path)` (Step 2;
  becomes `load_dataset(path)` after #959), otherwise
  `load_dataset(path, source_software="auto", fps=None)`. Then
  `ds_to_layer_data_tuples(ds, Path(path).name)`. Concatenate across paths so a
  multi-file drop yields one set of layers per file.
- Per-file errors (`ValueError`/`OSError`, including `infer_source_software`'s
  "Could not infer source_software…" and `load_movement_netcdf`'s messages) →
  `show_error` naming the file and pointing at the movement widget for explicit
  source-software selection and loader kwargs; skip that file. If nothing loads,
  return `[(None,)]` (napari's "no layers" sentinel) rather than raising.

**Manifest** — [movement/napari/napari.yaml](movement/napari/napari.yaml) gains:

```yaml
  commands:
    - id: movement.get_reader
      python_name: movement.napari.reader:napari_get_reader
      title: Open tracked data with movement
  readers:
    - command: movement.get_reader
      filename_patterns: ["*.h5", "*.csv", "*.slp", "*.nwb", "*.nc"]
      accepts_directories: false
```

npe2 manifests are static, so the patterns are hard-coded — add a test asserting
they equal `get_supported_source_software()` ∪ `{".nc"}` so the two can't
silently diverge (this also becomes the tripwire when a new loader is registered).

No packaging change needed: `MANIFEST.in` ships `napari.yaml` and the entry point
already exists ([pyproject.toml:52](pyproject.toml#L52)).

## Step 4 — Wire up layers created by the reader

**Why this is needed.** A reader can only hand napari *static* data —
`(data, attributes, layer_type)`. It cannot attach behaviour. But a movement
Points layer is not just data: `DataLoader` wires three live things onto it
after creation, none of which survive a `LayerDataTuple`.

| wiring | today | what it does |
|---|---|---|
| `events.data.connect(on_points_data_changed)` | [loader_widgets.py:363](movement/napari/loader_widgets.py#L363) | keeps the Tracks layer in sync when a point is dragged or deleted |
| `editable = frame_axis_is_sliced(viewer)` | [loader_widgets.py:364](movement/napari/loader_widgets.py#L364) | blocks editing when frame isn't the slider axis, so a drag can't move a point to another frame |
| `metadata[TRACKS_LAYER_KEY] = self.tracks_layer` | [loader_widgets.py:394](movement/napari/loader_widgets.py#L394) | the Points→Tracks object reference `sync_tracks_layer` needs |

Without this step the failure is not cosmetic: a user drags a point on a dropped
layer, the Points layer moves, the Tracks layer doesn't, and they silently
diverge — after which `DataSaver` writes that inconsistent state to `.nc`. Nor
can we dodge it by shipping dropped layers read-only: `editable` is not a
`Points.__init__` kwarg (see Step 1), so the reader cannot set it. Some
post-creation step is unavoidable.

**The function.** Since `fix/layer-wiring-lifetime` this belongs in
`layer_wiring.py`, not on the widget — it changes layer state that must outlive
the dock, which is exactly the rule in that module's docstring:

```python
def wire_movement_layers(viewer, event=None):
    """Wire up movement Points layers that aren't wired up yet."""
```

For each `Points` layer with `metadata[POINTS_LAYER_KEY]` and no
`TRACKS_LAYER_KEY` yet: resolve `TRACKS_LAYER_NAME_KEY` to the `Tracks` object
(skip if renamed or deleted), connect
`layer.events.data` → `on_points_data_changed`, set
`layer.editable = frame_axis_is_sliced(viewer)`, and call
`set_point_symbol_by_edited(layer)`.

**When it runs — and an ordering trap.** Connect it to
`viewer.layers.events.inserted` from inside `connect_viewer_callbacks`
([:57-60](movement/napari/layer_wiring.py#L57-L60)), alongside the existing
`update_frame_slider_range` wiring, using the same
`partial(wire_movement_layers, viewer)` pattern. Two paths reach it, and both
are covered by that one connection:

1. The widget is already open — `DataLoader.__init__` called
   `connect_viewer_callbacks` ([loader_widgets.py:78](movement/napari/loader_widgets.py#L78)),
   so the drop's `inserted` events fire the handler.
2. The widget was never opened — the reader calls `connect_viewer_callbacks`
   itself before returning its tuples, so the connection is in place by the time
   napari inserts them.

No "once at the end of `__init__`" pass is needed any more: on `main` that
existed to catch layers dropped before the widget was opened, but the reader now
wires the viewer itself. (`connect_viewer_callbacks`'s `_WIRED_VIEWERS` guard
makes the double call harmless, and
`test_connect_viewer_callbacks_is_idempotent`
([tests/test_unit/test_napari_plugin/test_layer_wiring.py:94](tests/test_unit/test_napari_plugin/test_layer_wiring.py#L94))
already pins that.)

The trap: napari adds a reader's layers **one at a time**
(`_add_layers_with_plugins` loops over the returned tuples calling
`_add_layer_from_data` for each), so `inserted` fires separately per layer. When
the *Points* layer's event fires, the *Tracks* layer does not exist yet, and
resolving `TRACKS_LAYER_NAME_KEY` will fail. "Not added yet" is therefore the
common case, not an edge case — and it fails quietly: the connection and
`editable` get set, the Tracks reference doesn't, and drag-sync stays broken.

So the handler must be **idempotent and scan `viewer.layers` in full on
every insert**, never just `event.value`. The Points insert wires what it can and
leaves the reference unresolved; the Tracks insert re-scans and completes it. The
"no `TRACKS_LAYER_KEY` yet" condition is what keeps repeated scans cheap and
safe. Two alternatives are worse: returning the tracks tuple first is fragile
(it depends on napari preserving order and on nobody reordering the list later),
and deferring via a single-shot timer adds async behaviour that is awkward to
test.

Also apply `_set_initial_state`'s effect
([loader_widgets.py:322](movement/napari/loader_widgets.py#L322)) from the
insert handler, guarded to fire only when a movement Points layer was inserted,
so a drop gets the same "slider to frame 0, points layer active" behaviour as
the Load button. It touches viewer state only, so it can move to
`layer_wiring.py` as `set_initial_state(viewer)` and be called from both the
widget and the insert handler.

No known limitation left here: with the reader calling
`connect_viewer_callbacks`, dropped layers are fully wired whether or not the
widget is ever opened, and they stay wired after it is closed. See discussion
point 2.

## Step 5 — Deliberately *not* touching the widget's suffix dicts

`SUPPORTED_POSES_FILES` / `SUPPORTED_BBOXES_FILES`
([:37-56](movement/napari/loader_widgets.py#L37-L56)) duplicate
`get_supported_source_software()` and already drift from it (Anipose and NWB are
registered in the backend but missing from the combo box). **Open PR #896 fixes
exactly this** — it adds Anipose and NWB to the dropdown and changes the form's
`rowCount()` — so deriving these dicts from the registry here would collide.

Leave them alone in this PR; note in the PR description that once #896 merges,
replacing the hard-coded dicts with a `get_supported_source_software()`-derived
mapping (plus the manual netCDF entry) is a small, safe follow-up. Also worth a
rebase check: #896 rewrites `_on_source_software_changed` and the form layout,
which are adjacent to but not overlapping Steps 1 and 4.


## Future work — autopopulating the form widget from a dropped file

**Not part of this PR.** Sketched here because it is the natural next step and
because it constrains nothing in the plan above: a user who dropped a file could
then tweak `fps` (or, post-#896, loader kwargs) without re-entering the path and
source software by hand. What follows is one possible shape for it.

No reader→widget coupling is needed: the information rides on the layer
metadata, exactly as `POINTS_LAYER_KEY` and `DATASET_ATTRS_KEY` already do, and
`wire_movement_layers` (Step 4) already runs for every inserted movement layer.

**1. One new metadata key**, set by `ds_to_layer_data_tuples` from the path it
was handed:

```python
SOURCE_FILE_KEY: str = "movement_source_file"
```

Do *not* reuse the existing `ds.attrs["source_file"]`: `from_anipose_file`
([load_poses.py:677-711](movement/io/load_poses.py#L677-L711)) never sets it, and
for a `.nc` file it round-trips from the original third-party file rather than
the `.nc` that was actually dropped. (Making `source_file` consistent across
loaders is worth a separate backend issue — discussion point 8.)

**2. One method on `DataLoader`**, called from `wire_movement_layers`:

```python
def _populate_form_from_layer(self, points_layer):
    """Fill the form fields from a wired-up layer's metadata."""
```

- `file_path_edit.setText(metadata[SOURCE_FILE_KEY])`
- source software: `.nc` suffix → `"movement (netCDF)"`, otherwise
  `metadata[DATASET_ATTRS_KEY]["source_software"]`, normalised through a small
  label map (currently one entry: `"DeepLabCut/LightningPose"` → `"DeepLabCut"`,
  which is what `load_dataset` actually loaded with,
  [load.py:416-419](movement/io/load.py#L416-L419))
- fps: set the spinbox only if `DATASET_ATTRS_KEY` carries an `fps` (i.e. the
  netCDF case); leave it alone otherwise, since dropped third-party files load
  with `fps=None`

`_on_source_software_changed` fires off the combo change for free, so the fps
spinbox correctly greys out for netCDF. Running this after a widget-initiated
load is a harmless no-op (the fields are already set), so there is no need to
branch on "was this a drop".

**3. Ask the combo, don't hard-code a list.** The form currently offers fewer
source software options than the backend supports: `SUPPORTED_DATA_FILES`
([:37-56](movement/napari/loader_widgets.py#L37-L56)) omits Anipose and NWB,
which *are* in the loader registry. So drag-and-drop will be able to load
Anipose `.csv` and `.nwb` files — a new capability for the GUI, since the form
cannot load them today — but the combo has no item to select for them.

Rather than special-casing those two, the lookup asks the widget what it can
currently represent:

```python
idx = self.source_software_combo.findText(software)
if idx >= 0:
    self.source_software_combo.setCurrentIndex(idx)
else:
    self._flag_software_not_selectable(software)
```

This is the bit that **extends naturally when #896 merges**: that PR adds
Anipose and NWB to the combo, `findText` starts returning a valid index, and the
fallback simply stops firing. Nothing to delete, no names to update — the code
never mentions Anipose or NWB at all.

The fallback must *not* be a silent no-op. `setCurrentText` on a non-editable
`QComboBox` leaves the previous selection in place, so dropping an Anipose CSV
would show that file's path next to `DeepLabCut`, and pressing **Load** would
attempt a wrong load. Instead `_flag_software_not_selectable` should disable the
Load button and set a tooltip naming the inferred software (e.g. *"Anipose files
can be dropped, but cannot yet be configured here"*), re-enabling on the next
user edit to the form. Note it deliberately does **not** add the item to the
combo: `_on_browse_clicked` indexes `SUPPORTED_DATA_FILES[currentText()]`
([:176-181](movement/napari/loader_widgets.py#L176-L181)) and would raise
`KeyError` for a software missing from that dict — extending the dict is #896's
job, see Step 5.

**Other edge cases:**

- Multi-file drop: the last inserted layer wins. Acceptable.

**Tests**: drop a DLC `.csv` (combo shows `DeepLabCut`, path filled, fps
untouched), a `.nc` (combo shows `movement (netCDF)`, fps from attrs, spinbox
disabled), and a VIA `.csv`. Then a registry-driven test that survives #896:
parametrised over `get_supported_source_software()`, assert that for every
registered software either the combo can select it *or* the fallback fired —
so the test keeps passing as the combo grows, and gets stricter for free.

See discussion point 3 for the UX question this raises about what **Load** then
does.


## Files changed

| File | Change |
|---|---|
| `movement/napari/layers.py` | **new** — `ds_to_layer_data_tuples`, `load_movement_netcdf` |
| `movement/napari/reader.py` | **new** — `napari_get_reader` |
| `movement/napari/napari.yaml` | add `movement.get_reader` command + `readers` contribution |
| `movement/napari/layer_wiring.py` | add `wire_movement_layers` (connected in `connect_viewer_callbacks`) and `TRACKS_LAYER_NAME_KEY`; optionally absorb `set_initial_state` |
| `movement/napari/loader_widgets.py` | delegate to the new module; `_load_netcdf_file` becomes a thin `show_error` wrapper |
| `docs/source/user_guide/gui.md` | document drag-and-drop of tracked data (§ *Load the tracked dataset*, ~line 122): the reader-choice dialog, the fps-in-frames caveat, and "use the widget for loader kwargs" |
| `docs/source/api_index.rst` | add `movement.napari.reader` / `layers` next to `convert`/`convert_roi` (lines 26-28) |
| `tests/test_unit/test_napari_plugin/test_reader.py` | **new** |
| `tests/test_unit/test_napari_plugin/test_layer_wiring.py` | add `wire_movement_layers` tests, alongside the existing widget-lifetime ones |
| `tests/test_unit/test_napari_plugin/test_data_loader_widget.py` | adapt to the refactor |


## Overview of tests to write

1. **Unit — reader as a pure function** (no viewer needed, a first for this test
   package): `napari_get_reader` returns a callable for `dlc_h5_file`,
   `dlc_csv_file`, `lp_csv_file`, `sleap_slp_file`, `sleap_analysis_file`,
   `anipose_csv_file`, `via_tracks_csv`, `valid_netcdf_file`
   ([tests/fixtures/files.py](tests/fixtures/files.py)) and `None` for
   `wrong_extension_file`, `directory`, `nonexistent_file`. Reader function
   returns 2 tuples for poses, 3 for bboxes, with layer types
   `("points","tracks"[,"shapes"])` and `meta["metadata"][POINTS_LAYER_KEY] is True`.
   Bad content (`readable_csv_file`, `invalid_dstype_netcdf_file`,
   `unopenable_netcdf_file`, `invalid_netcdf_file_missing_confidence`) →
   `show_error` called, `[(None,)]` returned. Manifest patterns equal
   `get_supported_source_software()` ∪ `{".nc"}`; `npe2.PluginManifest` validates
   the YAML.
2. **Unit — parity with the widget**: for a sample file, the layer
   data/properties/metadata from `ds_to_layer_data_tuples` are identical to what
   the `loaded_data_loader` fixture ([tests/fixtures/napari.py](tests/fixtures/napari.py))
   produces via the Load button. This is the key regression guard for the refactor.
3. **Unit — layer wiring**: with `make_napari_viewer_proxy`, three cases —
   widget then `viewer.open(path, plugin="movement")`, open-then-widget, and
   **open with the widget never instantiated at all** (the case
   `fix/layer-wiring-lifetime` makes work; it is the one that would silently
   regress if the reader forgot its `connect_viewer_callbacks` call). Assert
   `TRACKS_LAYER_KEY` resolved, `editable is True`, the frame slider range
   correct, and that the existing `move_point`/`remove_point` fixtures still keep
   the Tracks layer in sync. Add a fourth case mirroring
   `test_point_edit_syncs_tracks_layer_after_widget_is_gone`
   ([test_layer_wiring.py:36](tests/test_unit/test_napari_plugin/test_layer_wiring.py#L36)):
   drop, open then close the widget, edit, sync still holds.
4. **Integration**: drop → edit → `DataSaver` save → re-open round trip, since
   [save_widget.py](movement/napari/save_widget.py) reads `POINTS_PROPERTIES_KEY`
   and `DATASET_ATTRS_KEY` off reader-created layers.


## Verifications for agent to run
* **Manual**: `movement launch`; drag a DLC `.h5`, a DLC `.csv` (confirm the
   reader-choice dialog), a `.slp`, a VIA `.csv` and a movement `.nc` — with the
   widget open and closed, and several files at once. Confirm layer names,
   colours, tooltips and slider match the Load-button result.
* `pytest tests/test_unit/test_napari_plugin tests/test_unit/test_io` and
   `pre-commit run --all-files`.


## Points to discuss

1. **Sequencing against #959, and where GUI-strict validation should live.**
   Step 2 is written so the reader doesn't care when #959 lands, but the team may
   prefer to land #959 first and skip `load_movement_netcdf` entirely.

   The larger question is #959's, not this PR's, but this PR is what surfaces it.
   @niksirbi suggested there that `load_dataset` should validate netCDF only
   *minimally*, with the GUI's stricter requirements enforced instead at the
   movement→napari conversion layer. If that's adopted, the natural shape is a
   `validate_ds_for_napari(ds)` called at the top of `ds_to_layer_data_tuples`
   and applied to **every** dataset entering the conversion — which would give
   the GUI-compatibility rules a single home, and let the
   `netCDF files compatible with the GUI` dropdown in `gui.md` point at one
   function instead of restating them.

   This proposal deliberately stops short of that, for two reasons. It isn't
   needed for drag-and-drop: third-party datasets are valid by construction
   (see Step 2), so the check would be a no-op for everything but `.nc`. And it
   is a behaviour change — a DLC file that somehow lacked `confidence` would
   start failing at conversion rather than producing odd layers. Defensible, but
   it should be decided in #959 alongside the minimal-validation question, not
   smuggled in here.
2. **What happens when the widget isn't open?** *Largely settled by
   `fix/layer-wiring-lifetime`.* The original options were: (a) accept and
   document that drops without the widget give display-only layers; (b) have the
   reader call
   `napari.current_viewer().window.add_plugin_dock_widget("movement")` so the
   widget auto-opens on a movement drop; (c) move the sync logic off the widget
   onto something with viewer/layer lifetime. `fix/layer-wiring-lifetime` does
   (c) — `layer_wiring.py`'s callbacks are module-level functions and
   `connect_viewer_callbacks` binds them to the viewer, not the dock — so this
   plan now assumes (c) and neither (a)'s limitation nor (b)'s workaround
   applies.

   What remains open is smaller: the reader must remember to call
   `connect_viewer_callbacks(napari.current_viewer())`, and it depends on
   `current_viewer()` being non-`None` inside a reader hook (true for a drop on
   the canvas; worth an explicit guard for the headless/`viewer.open` case).
   Worth noting @TimMonko offered napari-side help in #960 — the underlying gap
   ("reader plugins can't attach behaviour to the layers they create") is still
   worth raising upstream, since every plugin has to hand-roll this.
3. **What should "Load" do after the form is autopopulated?** Only relevant if
   the future-work section is taken up, but it is the real design decision
   there, not the code. As things stand,
   drop → change fps to 30 → **Load** adds a *second* set of layers and leaves
   the user to delete the first. Options: (a) accept it — it matches today's
   behaviour when you load the same file twice; (b) detect that the form still
   describes an existing movement layer and offer to replace it in place;
   (c) add a distinct "Reload" affordance that appears once a layer is wired up.
   This proposal assumes (a) for a first iteration, but (b) is arguably what a
   user expects after the form has been filled in *for* them.
4. **fps consistency.** `fps=None` for drops is settled, but it means dropped
   data shows frame indices while the widget defaults to `1.0`. Should the widget
   default to frames too? Or should there be a way to set fps on an
   already-loaded layer instead of re-loading? Note that (b) above would largely
   answer this.
5. **Validation cost on inference.** `infer_source_software` probes every `.csv`
   validator and `ValidVIATracksCSV` parses the whole file, so dropping a large
   non-VIA `.csv` pays that cost before falling through. A cheap header-only
   pre-check in `ValidVIATracksCSV` would help; separate backend issue.
6. **Ambiguous `.h5`.** A file matching both DLC and SLEAP validators makes
   `infer_source_software` raise (only the DLC/LP pair is whitelisted). On drop
   we can only error and redirect to the widget. Should napari get a
   disambiguation prompt, or should the backend expose the candidate list rather
   than collapsing to a `ValueError`?
7. **Order of #960 vs #896.** #896 rewrites the same widget's dropdown and form
   layout. Whichever merges second eats a rebase. Given #896 is already open, it
   probably should go first — but Step 1's extraction is mostly in methods #896
   doesn't touch, so a concurrent merge is survivable. The future-work sketch is
   written to extend automatically when #896 lands (it queries the combo via
   `findText` rather than naming any software), so neither PR needs to know
   about the other beyond a rebase.
8. **`ds.attrs["source_file"]` is inconsistent.** It is set by the DLC/LP, SLEAP,
   VIA-tracks and NWB loaders but *not* by `from_anipose_file`
   ([load_poses.py:677-711](movement/io/load_poses.py#L677-L711)), and it
   survives a netCDF round trip pointing at the original third-party file. The
   future-work sketch sidesteps this with its own metadata key, but should the
   backend guarantee
   `source_file` on every loaded dataset? Separate issue if so.
9. **ROI `.geojson`/`.json` drops** (out of scope here): `RegionsWidget` owns
   region layers plus a Qt table model, so a dropped Shapes layer needs its own
   wiring path of its own. Follow-up issue — including whether `*.json` is too greedy a
   pattern for movement to claim.

* How can Claude verify a correct implementation?
When implemented, drag-and-dropping any of the third-party file supported via the widget (DLC `.h5`, DLC `.csv`, SLEAP analysis `.h5` or `.slp`, Anipose `.csv`, LP `.csv`, VIA `.csv` or `.nwb`) should produce produce the same Points, Tracks (and if loading bounding boxes data, Shapes layers) as the widget would produce today when loading those files via the file path input and selecting the corresponding source software.

* How should we document the drag-and-drop functionality?


* Thoughts on autopopulation
