"""Generate annotated screenshots of the movement napari GUI.

Each screenshot is built by driving the real GUI: load sample data,
set widget values, grab the window, then draw labels next to chosen
widgets. Label positions come from the widgets' on-screen geometry,
so they follow the widgets if the GUI layout changes.

Images are written to ``docs/source/_static``, replacing the existing
ones. Use ``git diff`` to see which changed, and ``git restore`` to undo.

Usage::

    python docs/make_gui_screenshots.py                 # all screenshots
    python docs/make_gui_screenshots.py poses_layers    # only some

Not covered (still made by hand): ``napari_video_reader.png`` (a modal
dialog) and any ``.mp4`` recordings.

See the "Updating GUI screenshots" section of the contributing guide
for how to add a new screenshot.
"""

import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import napari
from PIL import Image, ImageDraw, ImageFont
from qtpy.QtCore import QPoint, QRect, Qt
from qtpy.QtWidgets import QApplication, QLayout, QWidget

from movement import sample_data
from movement.napari.meta_widget import MovementMetaWidget

DEFAULT_WINDOW_SIZE = (1100, 660)
LINE_COLOR = (255, 255, 255, 255)
TEXT_BACKGROUND = (38, 40, 46, 200)  # napari's dark grey, see-through
CALLOUT_BACKGROUND = (255, 255, 255, 245)
CALLOUT_TEXT = (20, 20, 20, 255)
LINE_WIDTH = 3
FONT_SIZE = 22
PADDING = 6
# Space between an arrow's tip and the box (or widget) it points at
ARROW_GAP = 12
# Labels are drawn this many times larger, then shrunk for smooth edges
SMOOTHING = 4

# Images are saved next to the other docs images
STATIC_PATH = Path(__file__).resolve().parent / "source" / "_static"

OCTAGON = "SLEAP_two-mice_octagon.analysis.h5"
CRAB = "VIA_single-crab_MOCA-crab-1_linear-interp.csv"
EPM = "SLEAP_single-mouse_EPM.analysis.h5"

# Titles of the movement widget sections
LOAD_SECTION = "Load tracked data"
REGIONS_SECTION = "Define regions of interest"


# ----------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------
@dataclass
class Label:
    """A text label pointing at a region of the window.

    Parameters
    ----------
    target
        Region to point at, in window coordinates (logical pixels).
    text
        Label text. Use a newline character for line breaks.
    side
        Where the text goes relative to ``target``: ``"left"``,
        ``"right"``, ``"above"``, ``"below"``, or ``"inside"``
        (top-left corner of the target, with no arrow).
    box
        Whether to draw a box around ``target``.
    gap
        Distance between the text and ``target``.
    shift
        Moves the text along the side it sits on (down for left/right,
        right for above/below). Useful to avoid overlaps.
    callout
        Draw the text in a white bubble (as in the edit overview image)
        instead of white text on a dark background.

    """

    target: QRect
    text: str
    side: str = "right"
    box: bool = True
    gap: int = 40
    shift: int = 0
    callout: bool = False


def text_position(label: Label, size: tuple[float, float]) -> QPoint:
    """Return the top-left corner of a label's text (logical pixels)."""
    t, (w, h) = label.target, size
    cx, cy = t.center().x(), t.center().y()
    if label.side == "right":
        x, y = t.right() + label.gap, cy - h / 2 + label.shift
    elif label.side == "left":
        x, y = t.left() - label.gap - w, cy - h / 2 + label.shift
    elif label.side == "above":
        x, y = cx - w / 2 + label.shift, t.top() - label.gap - h
    elif label.side == "below":
        x, y = cx - w / 2 + label.shift, t.bottom() + label.gap
    elif label.side == "inside":
        x, y = t.left() + label.gap, t.top() + label.gap + label.shift
    else:
        raise ValueError(f"Unknown label side: {label.side!r}")
    return QPoint(round(x), round(y))


def arrow_points(label: Label, text_box: QRect) -> tuple[QPoint, QPoint]:
    """Return the start (at the text) and end (at the target) of an arrow."""
    t = label.target
    # The box is drawn 4 px outside the target, so stop past that
    gap = 4 + ARROW_GAP
    if label.side == "right":
        start = QPoint(text_box.left() - 4, text_box.center().y())
        end = QPoint(t.right() + gap, t.center().y())
    elif label.side == "left":
        start = QPoint(text_box.right() + 4, text_box.center().y())
        end = QPoint(t.left() - gap, t.center().y())
    elif label.side == "above":
        start = QPoint(text_box.center().x(), text_box.bottom() + 4)
        end = QPoint(t.center().x(), t.top() - gap)
    else:  # below
        start = QPoint(text_box.center().x(), text_box.top() - 4)
        end = QPoint(t.center().x(), t.bottom() + gap)
    return start, end


def draw_arrow(draw, start, end, ratio) -> None:
    """Draw a line from ``start`` to ``end`` with an open arrowhead at ``end``.

    The head is a V shape made of two short lines.
    """
    width = round(LINE_WIDTH * ratio)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 12 * ratio
    left, right = (
        (
            end[0] - size * math.cos(angle + delta),
            end[1] - size * math.sin(angle + delta),
        )
        for delta in (-math.pi / 6, math.pi / 6)
    )
    draw.line((*start, *end), fill=LINE_COLOR, width=width)
    draw.line(
        (*left, *end, *right), fill=LINE_COLOR, width=width, joint="curve"
    )
    # Round the line ends, so the head's tips are not cut off square
    radius = width / 2
    for x, y in (left, end, right):
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius), fill=LINE_COLOR
        )


def draw_labels(
    draw: ImageDraw.ImageDraw, labels: list[Label], ratio: float, window: QRect
) -> None:
    """Draw boxes, arrows and text for ``labels``.

    ``ratio`` converts window coordinates to pixels of the drawing,
    and ``window`` is the window's region in window coordinates.
    """
    # Size passed by position: Pillow >= 10.1 accepts it either way
    font = ImageFont.load_default(round(FONT_SIZE * ratio))
    width = round(LINE_WIDTH * ratio)

    def scaled(rect: QRect) -> tuple[float, float, float, float]:
        return (
            rect.left() * ratio,
            rect.top() * ratio,
            (rect.right() + 1) * ratio,
            (rect.bottom() + 1) * ratio,
        )

    for label in labels:
        if label.box:
            # Keep the box inside the image, so no edge gets cut off
            box = label.target.adjusted(-4, -4, 4, 4).intersected(
                window.adjusted(2, 2, -2, -2)
            )
            draw.rounded_rectangle(
                scaled(box),
                radius=4 * ratio,
                outline=LINE_COLOR,
                width=width,
            )

        # Measure the text in logical pixels, then place it
        bbox = draw.multiline_textbbox((0, 0), label.text, font=font)
        text_w = (bbox[2] - bbox[0]) / ratio + 2 * PADDING
        text_h = (bbox[3] - bbox[1]) / ratio + 2 * PADDING
        top_left = text_position(label, (text_w, text_h))
        # Keep the text inside the window
        max_x = window.width() - text_w - 2
        max_y = window.height() - text_h - 2
        top_left = QPoint(
            round(min(max(top_left.x(), 2), max_x)),
            round(min(max(top_left.y(), 2), max_y)),
        )
        text_box = QRect(
            top_left.x(), top_left.y(), round(text_w), round(text_h)
        )

        if label.side != "inside":
            start, end = arrow_points(label, text_box)
            draw_arrow(
                draw,
                (start.x() * ratio, start.y() * ratio),
                (end.x() * ratio, end.y() * ratio),
                ratio,
            )

        background = CALLOUT_BACKGROUND if label.callout else TEXT_BACKGROUND
        draw.rounded_rectangle(
            scaled(text_box),
            radius=(12 if label.callout else 4) * ratio,
            fill=background,
        )
        draw.multiline_text(
            (
                (top_left.x() + PADDING) * ratio - bbox[0],
                (top_left.y() + PADDING) * ratio - bbox[1],
            ),
            label.text,
            font=font,
            fill=CALLOUT_TEXT if label.callout else LINE_COLOR,
            align="center" if label.callout else "left",
        )


@dataclass
class Tooltip:
    """A mouse pointer with a tooltip, as napari shows on hover.

    Qt draws tooltips in their own popup window, which a window grab
    does not capture, so they are drawn onto the screenshot instead.

    Parameters
    ----------
    position
        Tip of the mouse pointer, in window coordinates (logical pixels).
    text
        Tooltip text, as napari would show it (see ``GUI.tooltip_at``).

    """

    position: QPoint
    text: str


def draw_tooltip(
    draw: ImageDraw.ImageDraw, tooltip: Tooltip, ratio: float
) -> None:
    """Draw a mouse pointer and a Qt-style tooltip."""
    x, y = tooltip.position.x() * ratio, tooltip.position.y() * ratio

    # Mouse pointer: a white arrow with a black outline
    s = ratio
    pointer = [
        (0, 0), (0, 17), (4, 13), (7, 20), (10, 19), (7, 12), (12, 12),
    ]  # fmt: skip
    draw.polygon(
        [(x + px * s, y + py * s) for px, py in pointer],
        fill="white",
        outline="black",
        width=max(1, round(s)),
    )

    # Tooltip box, offset below-right of the pointer like Qt does
    font = ImageFont.load_default(round(13 * ratio))
    bbox = draw.multiline_textbbox((0, 0), tooltip.text, font=font, spacing=4)
    pad = 6 * ratio
    left, top = x + 16 * s, y + 22 * s
    right = left + bbox[2] - bbox[0] + 2 * pad
    bottom = top + bbox[3] - bbox[1] + 2 * pad
    draw.rectangle(
        (left, top, right, bottom),
        fill=(38, 40, 46),
        outline=(160, 160, 160),
        width=max(1, round(s)),
    )
    draw.multiline_text(
        (left + pad - bbox[0], top + pad - bbox[1]),
        tooltip.text,
        font=font,
        fill=(240, 241, 242),
        spacing=4,
    )


# ----------------------------------------------------------------------
# Finding things on screen
# ----------------------------------------------------------------------
class GUI:
    """A napari viewer with the movement widget, plus lookup helpers."""

    def __init__(self, size: tuple[int, int] = DEFAULT_WINDOW_SIZE):
        """Open the viewer at ``size`` and dock the movement widget."""
        self.viewer = napari.Viewer()
        self.window = self.viewer.window._qt_window
        self.window.resize(*size)
        self.widget = MovementMetaWidget(self.viewer)
        self.viewer.window.add_dock_widget(self.widget, name="movement")
        # Open and close widget sections instantly, so screenshots never
        # catch a section halfway through its animation
        for section in self.widget.collapsible_widgets:
            section.setDuration(0)
        settle()

    # -- setup ---------------------------------------------------------
    def open_video(self, dataset: str):
        """Open the sample video of ``dataset`` and return its layer."""
        paths = sample_data.fetch_dataset_paths(dataset, with_video=True)
        layer = self.viewer.open(paths["video"], plugin="napari_video")[0]
        # Show the file name only, not the full local path
        layer.name = Path(paths["video"]).name
        return layer

    def load_tracks(self, dataset: str, source_software: str, fps: float):
        """Load a sample dataset through the movement loader widget."""
        paths = sample_data.fetch_dataset_paths(dataset)
        file_path = paths.get("poses") or paths.get("bboxes")
        loader = self.section(LOAD_SECTION).content()
        loader.source_software_combo.setCurrentText(source_software)
        loader.fps_spinbox.setValue(fps)
        loader.file_path_edit.setText(str(file_path))
        loader.load_button.click()
        settle()

    def expand_only(self, title: str) -> None:
        """Expand one movement widget section and collapse the others."""
        # Applied twice: an animation started before (e.g. when the
        # widget first opens) could otherwise undo the first pass
        for _ in range(2):
            for section in self.widget.collapsible_widgets:
                if section.text() == title:
                    section.expand(animate=False)
                else:
                    section.collapse(animate=False)
            settle()

    def layer(self, prefix: str):
        """Return the first layer whose name starts with ``prefix``."""
        return next(
            ly for ly in self.viewer.layers if ly.name.startswith(prefix)
        )

    # -- geometry ------------------------------------------------------
    def rect(self, *widgets: QWidget | QLayout) -> QRect:
        """Return the region covering ``widgets``, in window coordinates.

        Qt layouts (e.g. a grid of buttons) are accepted too.
        """
        rect = QRect()
        for w in widgets:
            if isinstance(w, QLayout):
                geometry = w.geometry()
                top_left = w.parentWidget().mapTo(
                    self.window, geometry.topLeft()
                )
                rect = rect.united(QRect(top_left, geometry.size()))
            else:
                top_left = w.mapTo(self.window, QPoint(0, 0))
                rect = rect.united(QRect(top_left, w.size()))
        return rect

    def section(self, title: str) -> QWidget:
        """Return a movement widget section (header and content)."""
        for section in self.widget.collapsible_widgets:
            if section.text() == title:
                return section
        titles = [s.text() for s in self.widget.collapsible_widgets]
        raise RuntimeError(f"No section {title!r}; found {titles}.")

    def layer_row(self, prefix: str) -> QRect:
        """Return the layer list row of the layer named ``prefix...``."""
        view = self.viewer.window._qt_viewer.layers
        model = view.model()
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            name = model.data(index, Qt.ItemDataRole.DisplayRole)
            if name.startswith(prefix):
                rect = view.visualRect(index)
                top_left = view.viewport().mapTo(self.window, rect.topLeft())
                return QRect(top_left, rect.size())
        raise RuntimeError(f"No layer list row starting with {prefix!r}.")

    def layer_controls(self, layer) -> QWidget:
        """Return the layer controls panel napari shows for ``layer``."""
        self.viewer.layers.selection.active = layer
        settle()
        return self.viewer.window._qt_viewer.controls.widgets[layer]

    def controls_rect(self, layer, *names: str) -> QRect:
        """Return the region covering napari layer controls ``names``.

        Each name is the attribute napari uses for the control,
        e.g. ``"_tail_length_slider_control"``.
        """
        controls = self.layer_controls(layer)
        widgets = []
        for name in names:
            try:
                control = getattr(controls, name)
            except AttributeError as e:
                raise RuntimeError(
                    f"napari has no layer control '{name}'. It may have "
                    "been renamed in this napari version; update the script."
                ) from e
            widgets += [
                w for pair in control.get_widget_controls() for w in pair
            ]
        return self.rect(*widgets)

    @property
    def canvas(self) -> QRect:
        """Return the region of the main canvas."""
        return self.rect(self.viewer.window._qt_viewer.canvas.native)

    @property
    def frame_slider(self) -> QRect:
        """Return the region of the frame slider below the canvas."""
        return self.rect(self.viewer.window._qt_viewer.dims)

    def world_to_window(self, y: float, x: float) -> QPoint:
        """Return where data coordinates ``(y, x)`` appear in the window."""
        camera = self.viewer.camera
        center_y, center_x = camera.center[-2:]
        canvas = self.canvas
        return QPoint(
            round(
                canvas.left()
                + canvas.width() / 2
                + (x - center_x) * camera.zoom
            ),
            round(
                canvas.top()
                + canvas.height() / 2
                + (y - center_y) * camera.zoom
            ),
        )

    def tooltip_at(self, position: tuple[float, ...]) -> str:
        """Move napari's cursor to ``position`` and return its tooltip text.

        Uses the active layer. This also updates the status bar, as
        hovering would.
        """
        # napari only computes tooltips while the mouse is on the canvas
        self.viewer.mouse_over_canvas = True
        self.viewer.tooltip.visible = True
        self.viewer.cursor.position = position
        self.viewer.update_status_from_cursor()
        settle()
        return self.viewer.tooltip.text

    # -- output --------------------------------------------------------
    def save(
        self,
        path: Path,
        labels: list[Label],
        tooltips: list[Tooltip] | None = None,
    ) -> Path:
        """Take a screenshot, draw ``labels`` on it and save it to ``path``."""
        settle()
        # Screenshot of the whole napari window, as an RGBA array
        array = self.viewer.screenshot(canvas_only=False, flash=False)
        image = Image.fromarray(array)
        # Screens with display scaling give more pixels than window size
        ratio = image.width / self.window.width()

        # Draw the labels on a larger transparent layer, then shrink it
        # onto the screenshot. Pillow does not smooth diagonal lines, so
        # this avoids jagged arrows.
        size = (image.width * SMOOTHING, image.height * SMOOTHING)
        overlay = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        window = QRect(0, 0, self.window.width(), self.window.height())
        draw_labels(draw, labels, ratio * SMOOTHING, window)
        for tooltip in tooltips or []:
            draw_tooltip(draw, tooltip, ratio * SMOOTHING)
        overlay = overlay.resize(image.size, Image.Resampling.LANCZOS)
        image = Image.alpha_composite(image.convert("RGBA"), overlay)
        image.convert("RGB").save(path)
        self.viewer.close()
        return path


def settle(seconds: float = 0.5) -> None:
    """Let Qt finish pending layout, animation and paint events."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)


# ----------------------------------------------------------------------
# Screenshots
# ----------------------------------------------------------------------
def video_slider(out: Path) -> Path:
    """Make the video layer screenshot, with the frame slider labelled."""
    gui = GUI()
    gui.open_video(OCTAGON)
    gui.viewer.dims.set_current_step(0, 4499)
    labels = [Label(gui.frame_slider, "Frame slider", side="above", shift=250)]
    return gui.save(out / "napari_video_slider.png", labels)


def poses_layers(out: Path) -> Path:
    """Make the screenshot of a poses dataset loaded over its video."""
    gui = GUI()
    gui.open_video(OCTAGON)
    gui.load_tracks(OCTAGON, "SLEAP", fps=50)
    gui.viewer.dims.set_current_step(0, 223)
    labels = [
        Label(
            gui.rect(gui.section(LOAD_SECTION)),
            "Widget for loading tracked data",
            side="left",
            shift=-60,
        ),
        *row_labels(
            gui, ["tracks", "points", ""], ["Tracks", "Points", "Image"]
        ),
    ]
    return gui.save(out / "napari_poses_layers.png", labels)


def bboxes_layers(out: Path) -> Path:
    """Make the screenshot of a bboxes dataset loaded over its video."""
    gui = GUI()
    gui.open_video(CRAB)
    gui.load_tracks(CRAB, "VIA-tracks", fps=1)
    gui.viewer.dims.set_current_step(0, 38)
    labels = [
        Label(
            gui.rect(gui.section(LOAD_SECTION)),
            "Widget for loading tracked data",
            side="left",
            shift=-60,
        ),
        *row_labels(
            gui,
            ["boxes", "tracks", "points", ""],
            ["Boxes", "Tracks", "Points", "Image"],
        ),
    ]
    return gui.save(out / "napari_bboxes_layers.png", labels)


def row_labels(gui: GUI, prefixes: list[str], names: list[str]) -> list[Label]:
    """Return labels pointing at layer list rows from the right.

    An empty prefix stands for the video layer (named after its file).
    """
    labels = []
    for prefix, name in zip(prefixes, names, strict=True):
        if not prefix:
            prefix = next(
                ly.name
                for ly in gui.viewer.layers
                if not ly.name.startswith(("tracks", "points", "boxes"))
            )
        labels.append(
            Label(gui.layer_row(prefix), f"{name} layer", box=False, gap=50)
        )
    return labels


def points_tooltip(out: Path) -> Path:
    """Make the screenshot of the tooltip shown when hovering a point."""
    gui = GUI()
    gui.open_video(OCTAGON)
    gui.load_tracks(OCTAGON, "SLEAP", fps=50)
    frame, individual, keypoint = 6865, "2", "Nose"
    gui.layer("tracks").visible = False
    points = gui.layer("points")
    points.size = 6
    points.current_size = 6
    points.text.visible = True
    gui.viewer.layers.selection.active = points
    gui.viewer.dims.set_current_step(0, frame)

    # Zoom in on one animal
    in_frame = points.data[:, 0] == frame
    features = points.features[in_frame].reset_index(drop=True)
    coords = points.data[in_frame]
    animal = coords[(features["individual"] == individual).to_numpy()]
    gui.viewer.camera.center = animal[:, 1:].mean(axis=0)
    gui.viewer.camera.zoom = 3.5
    settle()

    # Hover the chosen keypoint
    is_target = (features["individual"] == individual) & (
        features["keypoint"] == keypoint
    )
    target = coords[is_target.to_numpy()][0]
    text = gui.tooltip_at(tuple(target))
    tooltip = Tooltip(gui.world_to_window(*target[1:]), text)
    return gui.save(out / "napari_points_tooltip.png", [], [tooltip])


def tracks_sliders(out: Path) -> Path:
    """Make the tracks layer screenshot, with its length sliders labelled."""
    gui = GUI()
    gui.open_video(OCTAGON)
    gui.load_tracks(OCTAGON, "SLEAP", fps=50)
    tracks = gui.layer("tracks")
    # Show a short stretch of trajectory ahead of the current frame
    tracks.tail_length = 1
    tracks.head_length = 400
    gui.viewer.dims.set_current_step(0, 2000)
    sliders = gui.controls_rect(
        tracks, "_tail_length_slider_control", "_head_length_slider_control"
    )
    labels = [
        Label(sliders, "Tail and head length sliders", side="right", shift=-90)
    ]
    return gui.save(out / "napari_tracks_sliders.png", labels)


def new_region_layer(out: Path) -> Path:
    """Make the screenshot showing how to create a new region layer."""
    gui = GUI()
    video = gui.open_video(EPM)
    gui.viewer.dims.set_current_step(0, 8557)
    gui.expand_only(REGIONS_SECTION)
    regions = gui.section(REGIONS_SECTION).content()
    regions.add_layer_button.click()
    settle()
    labels = [
        Label(
            gui.layer_row(video.name),
            "1. Load a background layer",
            box=False,
            gap=50,
            shift=30,
        ),
        Label(
            gui.rect(gui.section(REGIONS_SECTION)),
            "2. Expand regions of\ninterest widget",
            side="left",
            gap=40,
            shift=-260,
        ),
        Label(
            gui.rect(regions.add_layer_button),
            "3a. Add new\nregion layer",
            side="left",
            box=False,
            gap=230,
            shift=-20,
        ),
        Label(
            gui.layer_row("regions"),
            "3b. Region layer appears",
            box=False,
            gap=50,
        ),
    ]
    return gui.save(out / "napari_new_region_layer.png", labels)


def drawing_regions(out: Path) -> Path:
    """Make the screenshot of a region layer and the drawing controls."""
    gui = GUI()
    video = gui.open_video(EPM)
    gui.viewer.dims.set_current_step(0, 8557)
    gui.expand_only(REGIONS_SECTION)
    regions = gui.section(REGIONS_SECTION).content()
    regions.add_layer_button.click()
    settle()

    # Draw the two open arms of the elevated plus maze
    height, width = video.data.shape[1:3]
    layer = gui.layer("regions")
    rows = (0.44 * height, 0.54 * height)
    for cols in ((0.16 * width, 0.42 * width), (0.58 * width, 0.86 * width)):
        layer.add_rectangles([[rows[0], cols[0]], [rows[1], cols[1]]])
    layer.properties = {"name": ["open-left", "open-right"]}
    layer.mode = "add_rectangle"
    layer.selected_data = {0}
    settle()

    controls = gui.layer_controls(layer)
    labels = [
        Label(
            gui.rect(controls.button_grid),
            "Shape\ndrawing\ntools",
            side="right",
            gap=60,
        ),
        Label(
            gui.controls_rect(
                layer,
                "_edge_width_slider_control",
                "_edge_color_control",
                "_face_color_control",
                "_text_visibility_control",
            ),
            "Shape\nappearance",
            side="right",
            gap=60,
        ),
        Label(gui.layer_row("regions"), "Region layer", box=False, gap=50),
        Label(
            gui.rect(regions.region_table_view),
            "Table of\nregions",
            side="left",
            box=False,
            gap=80,
            shift=60,
        ),
    ]
    return gui.save(out / "napari_drawing_regions.png", labels)


SCREENSHOTS: dict[str, Callable[[Path], Path]] = {
    "video_slider": video_slider,
    "poses_layers": poses_layers,
    "bboxes_layers": bboxes_layers,
    "points_tooltip": points_tooltip,
    "tracks_sliders": tracks_sliders,
    "new_region_layer": new_region_layer,
    "drawing_regions": drawing_regions,
}


if __name__ == "__main__":
    # Make the screenshots named on the command line, or all of them
    names = sys.argv[1:] or list(SCREENSHOTS)
    unknown = set(names) - set(SCREENSHOTS)
    if unknown:
        sys.exit(
            f"Unknown screenshots: {sorted(unknown)}. "
            f"Choose from: {list(SCREENSHOTS)}."
        )
    for name in names:
        print(f"Saved {SCREENSHOTS[name](STATIC_PATH)}")
