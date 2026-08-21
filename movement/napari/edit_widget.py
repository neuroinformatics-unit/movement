"""Widget flagging which frames contain edited points.

Adapted from a pattern for docking a ``matplotlib`` canvas in napari and
syncing it to the current frame via ``viewer.dims.events.current_step``:
https://gist.github.com/jni/a0ae9793a0ca43868dd7dce7ea21df79

Instantiated by :class:`~movement.napari.meta_widget.MovementMetaWidget`,
which docks it at the bottom of the viewer (rather than nesting it as a
collapsible section) since it's a wide timeline, not a compact control
panel.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from napari.layers import Points
from napari.layers.base import ActionType
from napari.utils.theme import get_theme
from napari.viewer import Viewer
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QCheckBox, QVBoxLayout, QWidget

from movement.napari.loader_widgets import (
    POINTS_LAYER_KEY,
    POINTS_PROPERTIES_KEY,
)

PLAYHEAD_COLOR = "#55606E"  # bar indicating current frame
# this is same color as napari slidebar.

# A click never lands exactly on a frame (e.g. 42.3, not 42), so treat
# any click within this fraction of the visible frame range as a hit.
CLICK_TOLERANCE_FRACTION = 0.02

MIN_VISIBLE_FRAMES = 5
ZOOM_IN_FACTOR = 0.8
ZOOM_OUT_FACTOR = 1.25

# Mouse movements below this many pixels are treated as a click, not
# the start of a pan drag (avoids a shaky click being read as a pan).
DRAG_THRESHOLD_PIXELS = 3


class EditWidget(QWidget):
    """Dock widget flagging frames with edited points.

    Draws one lane per individual, with a vertical bar for every frame
    that contains an edited point on the currently active ``movement``
    Points layer. Bars are coloured to match that point's colour in the
    Points/Tracks layers. A playhead line marks the frame currently
    shown in the viewer. Scroll to zoom in/out on the timeline, and
    click a bar to jump to that frame.
    """

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialise the widget and connect it to viewer events."""
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.active_layer: Points | None = None
        self._show_individuals = False
        self._bars: list = []
        self._edited_frames = np.array([])
        self._removed_points: list = []
        self._max_frame = 0
        self._press_pixel_x: float | None = None
        self._press_xlim: tuple[float, float] | None = None
        self._press_event = None
        self._pan_scale = 0.0
        self._dragged = False

        self.figure = Figure(figsize=(6, 2.5))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(200)
        self.ax = self.figure.subplots()
        self._style_axes()
        self.playhead = self.ax.axvline(0, color=PLAYHEAD_COLOR, linewidth=1)
        self._apply_theme()

        self.show_individuals_checkbox = QCheckBox("Display individuals")
        self.show_individuals_checkbox.setChecked(self._show_individuals)
        self.show_individuals_checkbox.toggled.connect(
            self._on_show_individuals_toggled
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.show_individuals_checkbox)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.viewer.dims.events.current_step.connect(self._on_step_changed)
        self.viewer.layers.selection.events.active.connect(
            self._on_active_layer_changed
        )
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        self.viewer.events.theme.connect(self._apply_theme)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        for layer in self.viewer.layers:
            self._track_layer(layer)
        self._on_active_layer_changed()

    def _style_axes(self):
        """Set the static appearance of the timeline axes."""
        self.ax.set_yticks([])
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("frame")
        self.ax.set_title("Edited frames", fontsize="small")
        self.figure.tight_layout()

    def _apply_theme(self, event=None):
        """Style the plot to match the current napari theme.

        Connected to ``viewer.events.theme`` so the plot updates live
        if the user switches between napari's dark and light themes.
        """
        theme = get_theme(self.viewer.theme)
        background = theme.background.as_hex()
        foreground = theme.text.as_hex()

        self.figure.set_facecolor(background)
        self.ax.set_facecolor(background)
        self.ax.xaxis.label.set_color(foreground)
        self.ax.title.set_color(foreground)
        self.ax.tick_params(axis="both", colors=foreground)
        for spine in self.ax.spines.values():
            spine.set_color(foreground)

        self.canvas.draw_idle()

    def _track_layer(self, layer):
        """Connect to a movement Points layer's data-change event."""
        if isinstance(layer, Points) and layer.metadata.get(POINTS_LAYER_KEY):
            layer.events.data.connect(self._on_layer_data_changed)

    def _on_layer_inserted(self, event):
        """Track newly added movement Points layers."""
        self._track_layer(event.value)

    def _on_layer_removed(self, event):
        """Clear the display if the active layer was removed."""
        if event.value is self.active_layer:
            self.active_layer = None
            self._max_frame = 0
            self._removed_points = []
            self._redraw_bars()

    def _on_show_individuals_toggled(self, checked):
        """Switch between one shared lane and one lane per individual."""
        self._show_individuals = checked
        self._redraw_bars()

    def _on_active_layer_changed(self, event=None):
        """Switch to displaying edited frames for the active layer."""
        # viewer.layers.selection.active is accessed through napari's
        # PublicOnlyProxy (wraps viewer access for plugin widgets), which
        # returns a fresh proxy wrapping the real layer on every access.
        # Unwrap it so later `is` comparisons against `event.source`/
        # `event.value` (always the raw layer, since those events are
        # emitted from inside the unwrapped layer/LayerList) succeed.
        active = self.viewer.layers.selection.active
        active = getattr(active, "__wrapped__", active)
        self.active_layer = (
            active
            if isinstance(active, Points)
            and active.metadata.get(POINTS_LAYER_KEY)
            else None
        )
        self._max_frame = (
            self.active_layer.metadata.get("max_frame_idx", 0)
            if self.active_layer is not None
            else 0
        )
        self._removed_points = self._reconstruct_previously_removed_points(
            self.active_layer
        )
        self._redraw_bars()
        self._reset_xlim()

    @staticmethod
    def _reconstruct_previously_removed_points(layer):
        """Restore removed-point bars from a previously-saved session.

        A point removed and saved in an earlier session comes back as a
        NaN row, dropped from the live layer entirely -- so there is no
        ``REMOVING`` event to capture it from this session. Reconstruct
        those bars from ``POINTS_PROPERTIES_KEY``, the full properties
        table (including NaN rows) stashed on the layer at load time.
        Points edited without being removed need no such reconstruction,
        since they're still live rows and ``_redraw_bars`` already reads
        their ``edited`` property directly.
        """
        if layer is None:
            return []
        full_properties = layer.metadata.get(POINTS_PROPERTIES_KEY)
        if full_properties is None or "edited" not in full_properties:
            return []
        previously_removed = full_properties[
            full_properties["position_is_nan"] & full_properties["edited"]
        ]
        if previously_removed.empty:
            return []
        individual_colors = dict(
            zip(
                layer.properties["individual"],
                layer.face_color,
                strict=False,
            )
        )
        return [
            (row.time, row.individual, individual_colors[row.individual])
            for row in previously_removed.itertuples()
            if row.individual in individual_colors
        ]

    def _on_layer_data_changed(self, event):
        """React to point moves and removals on the active layer.

        Connected to ``events.data`` rather than ``events.properties``:
        napari's ``Points.remove()`` (Delete/Backspace) only fires
        ``events.features``, never ``events.properties``, so a
        properties-only listener misses removals entirely.

        For a move (``CHANGED``), ``DataLoader._on_points_data_changed``
        (loader_widgets.py) is also connected to this same event and
        sets the ``edited`` property that ``_redraw_bars`` reads. But
        the two callbacks' relative order isn't guaranteed. Deferring
        the redraw with ``QTimer.singleShot(0, ...)`` runs it on the
        next Qt event-loop tick, after every callback for this event
        has finished, so ``edited`` is always up to date by then.

        For a removal, the row (and its identity) is about to be
        deleted from the layer entirely, so there will be nothing left
        to read afterwards. ``REMOVING`` fires first, while the data is
        still intact, so that's when we snapshot it.
        """
        if event.source is not self.active_layer:
            return
        if event.action == ActionType.REMOVING:
            self._capture_removed_points(event)
            self._redraw_bars()
        elif event.action == ActionType.CHANGED:
            QTimer.singleShot(0, self._redraw_bars)

    def _capture_removed_points(self, event):
        """Snapshot the identity of points about to be removed."""
        layer = event.source
        frames = layer.data[:, 0]
        individuals = layer.properties["individual"]
        colors = layer.face_color
        for idx in event.data_indices:
            self._removed_points.append(
                (frames[idx], individuals[idx], colors[idx])
            )

    def _on_step_changed(self, event=None):
        """Move the playhead line to the current frame."""
        step = self.viewer.dims.current_step
        if not step:
            return
        frame = step[0]
        self.playhead.set_xdata([frame, frame])
        self.canvas.draw_idle()

    def _redraw_bars(self):
        """Redraw the per-individual lanes of edited-frame bars."""
        for artist in self._bars:
            artist.remove()
        self._bars = []
        self._edited_frames = np.array([])

        if self.active_layer is None:
            self.ax.set_yticks([])
            self._on_step_changed()
            return

        individuals = self.active_layer.properties["individual"]
        removed_individuals = [ind for _, ind, _ in self._removed_points]
        # Union with removed individuals so a lane survives even if an
        # individual's last remaining point has just been removed.
        unique_individuals = list(
            dict.fromkeys([*individuals, *removed_individuals])
        )

        if self._show_individuals:
            n_lanes = len(unique_individuals)
            lane_of = {ind: i for i, ind in enumerate(unique_individuals)}
            self.ax.set_yticks([(i + 0.5) / n_lanes for i in range(n_lanes)])
            self.ax.set_yticklabels(unique_individuals, fontsize="small")
        else:
            # A single shared lane: one bar per edited frame, regardless
            # of how many (or which) individuals were edited on it.
            n_lanes = 1
            lane_of = dict.fromkeys(unique_individuals, 0)
            self.ax.set_yticks([])
        lane_height = 1.0 / n_lanes

        # Combine moved (still-live) and removed points into one list of
        # (frame, individual, colour) triples to draw as bars.
        edited = self.active_layer.properties.get("edited")
        moved_points = []
        if edited is not None and edited.any():
            frames = self.active_layer.data[:, 0]
            colors = self.active_layer.face_color
            moved_points = [
                (frames[idx], individuals[idx], colors[idx])
                for idx in np.nonzero(edited)[0]
            ]
        all_points = moved_points + self._removed_points

        if all_points:
            # A frame may have several edited keypoints for the same
            # individual; only draw one bar per (frame, individual) --
            # or, with lanes collapsed, one bar per frame regardless of
            # individual.
            seen = set()
            for frame, individual, color in all_points:
                key = (frame, individual) if self._show_individuals else frame
                if key in seen:
                    continue
                seen.add(key)
                lane = lane_of[individual]
                y0, y1 = lane * lane_height, (lane + 1) * lane_height
                self._bars.append(
                    self.ax.vlines(frame, y0, y1, colors=[color], linewidth=2)
                )
            self._edited_frames = np.unique([p[0] for p in all_points])

        self._on_step_changed()

    def _reset_xlim(self):
        """Reset the visible frame range to the full extent."""
        self.ax.set_xlim(0, max(self._max_frame, 1))
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        """Zoom the timeline in/out around the cursor position."""
        if event.inaxes != self.ax or event.xdata is None:
            return
        xmin, xmax = self.ax.get_xlim()
        span = xmax - xmin
        factor = ZOOM_IN_FACTOR if event.button == "up" else ZOOM_OUT_FACTOR
        full_span = max(self._max_frame, 1)
        new_span = min(max(span * factor, MIN_VISIBLE_FRAMES), full_span)

        frac = (event.xdata - xmin) / span if span else 0.5
        new_xmin = event.xdata - frac * new_span
        new_xmin = max(0, min(new_xmin, full_span - new_span))
        self.ax.set_xlim(new_xmin, new_xmin + new_span)
        self.canvas.draw_idle()

    def _on_mouse_press(self, event):
        """Record the drag start point, in case this becomes a pan."""
        if event.inaxes != self.ax or event.xdata is None:
            return
        self._press_event = event
        self._press_pixel_x = event.x
        self._press_xlim = self.ax.get_xlim()
        # Data units per screen pixel, fixed for the duration of this
        # drag so panning doesn't compound as the view shifts (see
        # _on_mouse_motion).
        axes_width_pixels = self.ax.get_window_extent().width
        span = self._press_xlim[1] - self._press_xlim[0]
        self._pan_scale = span / axes_width_pixels if axes_width_pixels else 0
        self._dragged = False

    def _on_mouse_motion(self, event):
        """Pan the timeline while the mouse is dragged.

        Uses the fixed pixel-to-data scale captured on press, rather
        than the live axes transform, so that repeated small moves
        accumulate into the total drag distance instead of being
        computed (and partly cancelled out) against an already-shifted
        view on every step.
        """
        if self._press_pixel_x is None or event.x is None:
            return
        pixel_dx = event.x - self._press_pixel_x
        if abs(pixel_dx) > DRAG_THRESHOLD_PIXELS:
            self._dragged = True
        if not self._dragged:
            return

        data_dx = pixel_dx * self._pan_scale
        xmin0, xmax0 = self._press_xlim
        span = xmax0 - xmin0
        full_span = max(self._max_frame, 1)
        new_xmin = max(0, min(xmin0 - data_dx, full_span - span))
        self.ax.set_xlim(new_xmin, new_xmin + span)
        self.canvas.draw_idle()

    def _on_mouse_release(self, event):
        """Treat the gesture as a click if the mouse never really moved."""
        if self._press_pixel_x is None:
            return
        if not self._dragged:
            self._handle_click(self._press_event)
        self._press_pixel_x = None
        self._press_xlim = None
        self._press_event = None

    def _handle_click(self, event):
        """Jump the viewer to the edited frame nearest the click.

        Double-clicking resets the timeline to the full frame range.
        Clicks too far from any edited-frame bar are ignored.
        """
        if event.dblclick:
            self._reset_xlim()
            return
        if self._edited_frames.size == 0:
            return

        nearest = self._edited_frames[
            np.argmin(np.abs(self._edited_frames - event.xdata))
        ]
        xmin, xmax = self.ax.get_xlim()
        tolerance = max(1.0, CLICK_TOLERANCE_FRACTION * (xmax - xmin))
        if abs(nearest - event.xdata) > tolerance:
            return

        current_step = self.viewer.dims.current_step
        self.viewer.dims.current_step = (int(nearest),) + current_step[1:]
