# PR 1 — Give movement layer wiring a lifetime that isn't the widget's

## Why

**Symptom.** Layers become unsynced if the movement meta-widget is closed.

Sample sequence that would trigger this:
1. Load via the panel.
2. Close the panel (e.g. for canvas space) .
3. Edit points. At first they still sync (e.g. the symbol changes to ring), because the orphaned `DataLoader` is not garbage collected yet, but at an unpredictable time they will stop being marked as 'edited'.
4. Reopen the movement meta-widget to save the changes. This triggers the GC for sure, the old `DataLoader` is collected and its connections go dead. The new `DataLoader` does not adopt the existing layers, because `events.data` is connected only in `_add_points_layer`.
5. Edit a few more points before saving (now silently unsynced).
6. Save → DataSaver writes the inconsistent state.

**Cause.** All the sync logic between layers is in **bound methods of the `DataLoader` widget**, and napari prunes them silently once the widget dies.

* napari's `EventEmitter._normalize_cb` dereferences a bound method into a
weak `(ref, method_name)` pair, and prunes it silently once the referent dies
(`napari/utils/events/event.py`, ~L660).
* So the connections live exactly as long as
the `DataLoader` instance — which is not as long as the layers they act on.

Closing the panel calls napari's `destroyOnClose` → `remove_dock_widget`, which
detaches the inner widget and `deleteLater`s the dock. The `DataLoader` is then
unreachable but sits in a **reference cycle** (its `CollapsibleWidget`, that
widget's `__dict__`, the `QPropertyAnimation` driving the collapse, and a closure
cell all refer to it), so it is freed only by the *cyclic* collector. Verified
against napari 0.6.6: `gc.collect(0)` leaves it alive, `gc.collect(1)` frees it.

That is what makes the failure intermittent rather than immediate. The sync keeps
working for an arbitrary period after the panel is closed, then stops at a moment
governed by GC pressure rather than by anything the user did. Reopening the panel
tends to provoke it — building a fresh widget tree triggers a gen-1 pass — but is
not the causal step. Hiding the dock (the View-menu toggle) is safe; only the **x**
destroys.


## What the fix rests on

This could be fixed if we change what owns the callbacks, keeping their functionality intact.

More specifically: napari's `EventEmitter._normalize_cb` treats any *non-method* callable as a strong reference. So a module-level function connected to a layer's emitter lives as long as that layer, and no owner object is needed.


## Suggested implementation

**New file `movement/napari/layer_wiring.py`.**

Five handlers out of the eight total ones move across as module-level functions, unchanged. They are already
stateless — two are `@staticmethod`, and `_on_points_data_changed` works off
`event.source`:

| moved from `loader_widgets.py` | note |
|---|---|
| `_on_points_data_changed` ([:435](movement/napari/loader_widgets.py#L435)) | uses `event.source`, no `self` |
| `_sync_tracks_layer` ([:472](movement/napari/loader_widgets.py#L472)) | resolves the Tracks layer from `points_layer.metadata` |
| `_remove_from_tracks_layer` ([:492](movement/napari/loader_widgets.py#L492)) | same |
| `_set_tracks_layer_data` ([:516](movement/napari/loader_widgets.py#L516)) | already `@staticmethod` |
| `_set_point_symbol_by_edited` ([:425](movement/napari/loader_widgets.py#L425)) | already `@staticmethod` |

Three take `viewer` as their first argument instead of reading `self.viewer`:
`frame_axis_is_sliced` ([:394](movement/napari/loader_widgets.py#L394)),
`update_points_layers_editable` ([:407](movement/napari/loader_widgets.py#L407)),
`update_frame_slider_range` ([:583](movement/napari/loader_widgets.py#L583)).

**Two changes at the connection sites:**

1. `_add_points_layer` ([:385](movement/napari/loader_widgets.py#L385)) connects the
   module function rather than the bound method, so that connection now has the
   *layer's* lifetime.
2. The four viewer-level subscriptions currently made in `DataLoader.__init__`
   ([:85-101](movement/napari/loader_widgets.py#L85-L101)) move behind one
   idempotent entry point:

```python
# movement/napari/layer_wiring.py
from functools import partial
from weakref import WeakSet

_WIRED: WeakSet = WeakSet()   # viewers already subscribed

def ensure_layer_wiring(viewer) -> None:
    """Idempotently subscribe a viewer to movement layer wiring."""
    if viewer in _WIRED:
        return
    _WIRED.add(viewer)
    for action in ("inserted", "removed"):
        getattr(viewer.layers.events, action).connect(
            partial(update_frame_slider_range, viewer)
        )
    for event in (viewer.dims.events.order, viewer.dims.events.ndisplay):
        event.connect(partial(update_points_layers_editable, viewer))
```

`partial` is not a bound method, so the emitter holds it strongly and the
subscription lasts as long as the viewer. The resulting
viewer → emitter → `partial` → viewer cycle is ordinary cyclic garbage, collected
with the viewer, so no viewer is leaked. The `WeakSet` is only a "already
subscribed?" guard and keeps nothing alive.

`DataLoader.__init__` then calls `ensure_layer_wiring(self.viewer)` in place of its
four `connect` calls, and keeps thin methods delegating to the new functions — so
`test_data_loader_widget.py` and the `move_point` / `remove_point` fixtures are
untouched. Migrating those tests off Qt is a separate, safe follow-up.


## Testing

The bug is intermittent in the UI but deterministic in pytest, because the test can
force the collection the UI performs eventually:

```python
loader = ...                 # load a file through the widget
dw.destroyOnClose(); del loader, meta, dw
gc.collect()                 # what the UI does at an unpredictable moment
move_point(...)              # existing fixture
assert <tracks row matches the moved point>
```

This fails on `main` and passes with the fix. Add the delete case
(`remove_point`) and an assertion that `editable` still flips when the axes are
rolled after the panel is closed. The existing widget tests validate that no functionality is broken from the move.
