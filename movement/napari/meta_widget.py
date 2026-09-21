"""The main napari widget for the ``movement`` package."""

from typing import TYPE_CHECKING

from napari.layers import Points
from napari.layers.base import ActionType
from napari.viewer import Viewer
from qt_niu.collapsible_widget import CollapsibleWidgetContainer
from qtpy.QtCore import QTimer

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

from movement.napari.edit_timeline_widget import (
    EditControlsWidget,
    EditTimelineWidget,
)
from movement.napari.loader_widgets import POINTS_LAYER_KEY, DataLoader
from movement.napari.regions_widget import RegionsWidget
from movement.napari.save_widget import DataSaver


class MovementMetaWidget(CollapsibleWidgetContainer):
    """The widget to rule all ``movement`` napari widgets.

    This is a container of collapsible widgets, each responsible
    for handing specific tasks in the movement napari workflow.
    """

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the meta-widget."""
        super().__init__()
        self._viewer = napari_viewer
        self.edit_timeline_widget: EditTimelineWidget | None = None
        self._edit_timeline_dock_widget: QWidget | None = None

        # Add the data loader widget
        self.add_widget(
            DataLoader(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Load tracked data",
        )

        # A collapsible "edit controls" widget that can be used
        # to show/hide and configure the edit timeline docked
        # to the bottom of the viewer.
        self.edit_controls = EditControlsWidget(parent=self)
        self.edit_controls.show_individuals_toggled.connect(
            self._on_show_individuals_toggled
        )
        self.add_widget(
            self.edit_controls,
            collapsible=True,
            widget_title="Edit tracked data",
        )
        self._edit_timeline_collapsible = self.collapsible_widgets[-1]
        self._edit_timeline_collapsible.toggled.connect(
            self._on_edit_timeline_widget_toggled
        )

        # Add the Save widget
        self.add_widget(
            DataSaver(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Save tracked data",
        )

        # Add the Regions widget
        self.add_widget(
            RegionsWidget(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Define regions of interest",
        )

        loader_collapsible = self.collapsible_widgets[0]
        loader_collapsible.expand()  # expand the loader widget by default

        napari_viewer.layers.events.inserted.connect(self._on_layer_inserted)

        self.edit_controls.show_individuals_checkbox.setEnabled(False)
        napari_viewer.layers.selection.events.active.connect(
            self._show_individuals_enabled
        )

    @staticmethod
    def _is_movement_points(layer) -> bool:
        """Return ``True`` if ``layer`` is a movement-loaded Points layer."""
        layer = getattr(layer, "__wrapped__", layer)
        return isinstance(layer, Points) and bool(
            layer.metadata.get(POINTS_LAYER_KEY)
        )

    def _on_layer_inserted(self, event) -> None:
        """Keep the edit timeline section collapsed until a point is edited."""
        layer = event.value
        if not self._is_movement_points(layer):
            return  # ignore any layer that is not a movement Points layer
        self._show_individuals_enabled()
        # Open the edit timeline section as soon as a point is edited
        # on this layer.
        layer.events.data.connect(self._on_points_edited)
        self._edit_timeline_collapsible.collapse(False)

    def _on_points_edited(self, event) -> None:
        """Expand the edit timeline section when a point is dragged or removed.

        Expanding creates the timeline widget on first edit. We defer this
        until the event loop is next free (via ``QTimer.singleShot``). This
        allows the layer's ``edited`` property to be fully set before
        the timeline widget reads it, and thus ensures the first edit
        is not missed.
        """
        if event.action in (ActionType.CHANGED, ActionType.REMOVING):
            QTimer.singleShot(0, self._edit_timeline_collapsible.expand)

    def _on_edit_timeline_widget_toggled(self, expanded: bool) -> None:
        """Show/hide the edited-frames timeline docked at the bottom."""
        if not expanded:
            if self._edit_timeline_dock_widget is not None:
                self._edit_timeline_dock_widget.hide()
            return
        self._autoselect_points_layer()
        if self.edit_timeline_widget is None:
            self.edit_timeline_widget = EditTimelineWidget(self._viewer)
            self.edit_timeline_widget.set_show_individuals(
                self.edit_controls.show_individuals_checkbox.isChecked()
            )
            self._edit_timeline_dock_widget = (
                self._viewer.window.add_dock_widget(
                    self.edit_timeline_widget,
                    area="bottom",
                    name="edited frames",
                )
            )
            # Handle closing the dock via its title-bar "X"
            self._edit_timeline_dock_widget.destroyed.connect(
                self._on_edit_timeline_dock_gone
            )
        elif self._edit_timeline_dock_widget is not None:
            self._edit_timeline_dock_widget.show()

    def _on_edit_timeline_dock_gone(self, _=None) -> None:
        """Reset state after the docked timeline is closed via its X."""
        self.edit_timeline_widget = None
        self._edit_timeline_dock_widget = None
        # Collapse the edit controls, in line with the now-missing dock.
        self._edit_timeline_collapsible.collapse(False)

    def _autoselect_points_layer(self) -> None:
        """Make a movement Points layer active for the timeline.

        Leave the active layer alone if it is already a movement Points
        layer; otherwise select the last one in the layer list.
        """
        if self._is_movement_points(self._viewer.layers.selection.active):
            return
        layer = self._active_movement_points_layer()
        if layer is not None:
            self._viewer.layers.selection.active = layer

    def _on_show_individuals_toggled(self, checked: bool) -> None:
        """Forward the "Display individuals" checkbox to the timeline."""
        if self.edit_timeline_widget is not None:
            self.edit_timeline_widget.set_show_individuals(checked)

    def _active_movement_points_layer(self):
        """Return the active movement Points layer, or the last one."""
        active = self._viewer.layers.selection.active
        if self._is_movement_points(active):
            return getattr(active, "__wrapped__", active)
        for layer in reversed(self._viewer.layers):
            if self._is_movement_points(layer):
                return getattr(layer, "__wrapped__", layer)
        return None

    def _show_individuals_enabled(self, *_) -> None:
        """Enable "Display individuals" only for multi-individual data.

        With a single individual the checkbox does nothing useful, so it
        is disabled (and unchecked, falling back to the single-colour
        shared lane).
        """
        layer = self._active_movement_points_layer()
        if layer is None:
            return
        individuals = layer.properties.get("individual")
        multiple = individuals is not None and len(set(individuals)) > 1
        checkbox = self.edit_controls.show_individuals_checkbox
        checkbox.setEnabled(multiple)
        if not multiple and checkbox.isChecked():
            checkbox.setChecked(False)
