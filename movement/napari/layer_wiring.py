"""Layer callbacks that must outlive the widget connecting them.

Put a callback here if it should be active as long as the viewer
or a layer exists (rather than sharing lifetime with a widget that
can be closed).

- Typical candidates are callbacks that mutate layer state (data, properties,
  symbols, ``editable``): that state is part of the user's work and may be
  saved to a file later, so it must survive the widget being closed.
- A callback that only refreshes a widget's own UI (a dropdown, a table, a
  button) can stay a method on that widget: once the widget is gone there is
  nothing left to update, which is fine.

"""

import warnings
from functools import partial
from weakref import WeakSet

import numpy as np
from napari.components.dims import RangeTuple
from napari.layers import Points
from napari.layers.base import ActionType

from movement.napari.layer_styles import EDITED_POINT_SYMBOL

# Metadata keys stored on the movement Points layer.
# - POINTS_LAYER_KEY marks the layer as movement-created.
# - POINTS_PROPERTIES_KEY holds the full properties df, incl. the NaN rows
#   dropped from the live layer, needed to reconstruct the dataset.
# - DATASET_ATTRS_KEY holds the source dataset's attrs (source_software, fps…).
# - TRACKS_LAYER_KEY holds a reference to the companion Tracks layer.
# - MAX_FRAME_IDX_KEY holds the last frame index of the source data,
#   including leading/trailing all-NaN frames (which are dropped from the
#   napari layer data array).
POINTS_LAYER_KEY: str = "movement_points_layer"
POINTS_PROPERTIES_KEY: str = "movement_points_properties"
DATASET_ATTRS_KEY: str = "movement_dataset_attrs"
CONFIDENCE_DIMS_KEY: str = "movement_confidence_dims"
TRACKS_LAYER_KEY: str = "movement_tracks_layer"
MAX_FRAME_IDX_KEY: str = "movement_max_frame_idx"

# Keep a set of viewers already wired by connect_viewer_callbacks,
# so we don't wire them twice. We use a WeakSet so tracking a viewer here
# does not prevent it from being garbage-collected when it is no longer used.
_WIRED_VIEWERS: WeakSet = WeakSet()


# ---- Callbacks with viewer lifetime --------------------
def connect_viewer_callbacks(viewer) -> None:
    """Wire the layer callbacks to a viewer, skipping if already wired.

    These wirings last as long as the viewer, no matter which widget requested
    them.
    """
    # Check if viewer has been wired by this function
    viewer = getattr(viewer, "__wrapped__", viewer)
    if viewer in _WIRED_VIEWERS:
        return

    # Connect relevant layer callbacks to the viewer.
    # Connect frame slider range update to layer events
    for action in (
        "inserted",
        "removed",
    ):
        getattr(viewer.layers.events, action).connect(
            partial(update_frame_slider_range, viewer)
        )

    # Point drags are only guaranteed to stay within their own frame
    # when frame is the sliced (non-displayed) axis in a 2D view. If
    # axes are rolled or a 3D view is used in the viewer, disable editing
    # rather than risk a drag moving a point onto a different frame.
    for event in (
        viewer.dims.events.order,
        viewer.dims.events.ndisplay,
    ):
        event.connect(partial(update_points_layers_editable, viewer))

    # Update set
    _WIRED_VIEWERS.add(viewer)


def update_frame_slider_range(viewer, event=None):
    """Widen the frame slider range to cover NaN-trimmed frames.

    napari derives ``viewer.dims.range`` from the world-coordinate union of
    all layer extents, and it does so before this callback runs. Movement's
    layers have their leading/trailing all-NaN rows dropped, so a dataset
    that starts or ends with NaNs yields an extent — and therefore a slider
    range — narrower than the real frame span.

    Extend napari's range to cover the true span of every movement layer,
    but never replace it: layers movement did not create (a video the user
    opened, another plugin's layer) keep the range napari computed for them,
    including any scale or translate they carry.
    """
    max_frame_indices = [
        ly.metadata[MAX_FRAME_IDX_KEY]
        for ly in viewer.layers
        if MAX_FRAME_IDX_KEY in getattr(ly, "metadata", {})
    ]
    if not max_frame_indices:
        return

    current_range = viewer.dims.range[0]
    start = min(current_range.start, 0.0)
    stop = max(current_range.stop, *max_frame_indices)

    if (start, stop) != (current_range.start, current_range.stop):
        viewer.dims.range = (
            RangeTuple(start=start, stop=stop, step=current_range.step),
        ) + viewer.dims.range[1:]


def frame_axis_is_sliced(viewer) -> bool:
    """Determine whether the frame axis is the sliced axis in a 2D view."""
    return viewer.dims.ndisplay == 2 and viewer.dims.order[0] == 0


def update_points_layers_editable(viewer, event=None):
    """Disable point editing while the frame axis isn't sliced.

    Connected to ``viewer.dims.events.order``/``ndisplay``.
    It disables editing on every movement Points layer if
    the viewer axes are rolled or switched to 3D; napari greys
    out the select/add/delete controls.
    """
    is_editable = frame_axis_is_sliced(viewer)
    for layer in viewer.layers:
        if isinstance(layer, Points) and layer.metadata.get(POINTS_LAYER_KEY):
            layer.editable = is_editable


# ---- Callbacks with layer lifetime --------------------
def set_point_symbol_by_edited(layer: Points) -> None:
    """Show points flagged as edited with a distinct marker symbol."""
    edited = layer.properties.get("edited")
    if edited is None or not edited.any():
        return
    symbols = np.asarray(layer.symbol).copy()
    symbols[edited] = EDITED_POINT_SYMBOL
    layer.symbol = symbols


def on_points_data_changed(event):
    """Keep the corresponding Tracks layer in sync with the Points layer.

    Connected to ``points_layer.events.data``. Handles two actions:

    - ``ActionType.CHANGED`` (a point was dragged): sets the
      confidence score of moved points to NaN, marks them as
      edited, and changes their marker symbol to
      ``EDITED_POINT_SYMBOL`` so edited points are visually
      distinguishable. The Tracks layer row is updated in place
      via `sync_tracks_layer`.
    - ``ActionType.REMOVED`` (one or more points were deleted):
      removes the corresponding rows from
      the Tracks layer via `remove_from_tracks_layer`.
    """
    layer = event.source
    if not isinstance(layer, Points):
        return

    if event.action == ActionType.CHANGED:
        moved_indices = list(event.data_indices)
        props = layer.properties
        props["confidence"] = props["confidence"].copy()
        props["confidence"][moved_indices] = float("nan")
        if "edited" in props:
            props["edited"] = props["edited"].copy()
        else:
            props["edited"] = np.full(len(props["confidence"]), False)
        props["edited"][moved_indices] = True
        layer.properties = props
        set_point_symbol_by_edited(layer)
        sync_tracks_layer(layer, moved_indices)

    elif event.action == ActionType.REMOVED:
        removed_indices = list(event.data_indices)
        remove_from_tracks_layer(layer, removed_indices)


def sync_tracks_layer(points_layer, moved_indices):
    """Update the corresponding Tracks layer to match an edited point.

    A moved point's new (frame, y, x) is written to the same row
    in the Tracks layer, so the track segment connecting the
    previous frame to this one terminates at the dragged position.
    """
    tracks_layer = points_layer.metadata[TRACKS_LAYER_KEY]

    # Points and Tracks layers are built from the same NaN-filtered
    # array in the same row order (see _add_points_layer/
    # _add_tracks_layer). The Tracks layer only has an extra
    # leading track_id column.
    tracks_data = tracks_layer.data
    tracks_data[moved_indices, 1:] = points_layer.data[moved_indices]

    set_tracks_layer_data(tracks_layer, tracks_data, tracks_layer.properties)


def remove_from_tracks_layer(points_layer, removed_indices):
    """Remove the rows corresponding to deleted points.

    Users edit the Points layer directly, either dragging points
    or removing inaccurate predictions. The Tracks layer has no
    interactive editing of its own, so it must be kept in sync
    with the Points layer instead.

    ``removed_indices`` are indices in the Points layer which line
    up with rows in the Tracks layer, the same way
    :func:`sync_tracks_layer` relies on for edits.
    """
    tracks_layer = points_layer.metadata[TRACKS_LAYER_KEY]

    tracks_data = np.delete(tracks_layer.data, removed_indices, axis=0)
    tracks_properties = {
        key: np.delete(np.asarray(values), removed_indices, axis=0)
        for key, values in tracks_layer.properties.items()
    }

    set_tracks_layer_data(tracks_layer, tracks_data, tracks_properties)


def set_tracks_layer_data(tracks_layer, data, properties):
    """Set a Tracks layer's data and properties, preserving color_by.

    Setting ``.data`` on a napari Tracks layer resets its internal
    features to empty, which transiently invalidates ``color_by``
    (napari warns and falls back to "track_id") even though we
    restore the same properties and colour-by property right
    after. Suppress that spurious warning around the sequence.
    """
    color_by = tracks_layer.color_by
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Previous color_by key.*",
            category=UserWarning,
        )
        tracks_layer.data = data
        tracks_layer.properties = properties
        tracks_layer.color_by = color_by
