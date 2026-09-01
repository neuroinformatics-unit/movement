"""The main napari widget for the ``movement`` package."""

from napari.layers import Points
from napari.layers.base import ActionType
from napari.viewer import Viewer
from qt_niu.collapsible_widget import CollapsibleWidgetContainer

from movement.napari.edit_widget import EditControlsWidget, EditWidget
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
        self.edit_widget: EditWidget | None = None
        self._edit_dock_widget = None

        # Add the data loader widget
        self.add_widget(
            DataLoader(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Load tracked data",
        )

        # Add the Regions widget
        self.add_widget(
            RegionsWidget(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Define regions of interest",
        )

        # The edit widget is a timeline, best shown full-width rather
        # than squeezed into this side panel. This collapsible section
        # instead acts as a switch: expanding it docks the widget at
        # the bottom of the viewer; collapsing it hides it again.
        self.edit_controls = EditControlsWidget(parent=self)
        self.edit_controls.show_individuals_toggled.connect(
            self._on_show_individuals_toggled
        )
        self.add_widget(
            self.edit_controls,
            collapsible=True,
            widget_title="Edit tracked data",
        )
        self._edit_collapsible = self.collapsible_widgets[-1]
        self._edit_collapsible.toggled.connect(self._on_edit_widget_toggled)

        # Add the Save widget
        self.add_widget(
            DataSaver(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Save tracked data",
        )

        loader_collapsible = self.collapsible_widgets[0]
        loader_collapsible.expand()  # expand the loader widget by default

        # A freshly loaded dataset with no prior edits should keep the
        # edit section collapsed (the default above); one loaded with
        # previously edited points should instead open it right away
        # so those edits are visible without an extra click.
        napari_viewer.layers.events.inserted.connect(self._on_layer_inserted)

        # "Display individuals" is meaningless with a single individual;
        # keep it disabled until a multi-individual layer is active.
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
        """Show the edit section only for a layer with prior edits."""
        layer = event.value
        if not self._is_movement_points(layer):
            return  # ignore any layer that is not a movement Points layer
        self._show_individuals_enabled()
        # Open the edit section as soon as a point is edited on this layer.
        layer.events.data.connect(self._on_points_edited)
        edited = layer.properties.get("edited")
        if edited is not None and edited.any():
            self._edit_collapsible.expand()
        else:
            self._edit_collapsible.collapse(False)

    def _on_points_edited(self, event) -> None:
        """Expand the edit section when a point is dragged or removed."""
        if event.action in (ActionType.CHANGED, ActionType.REMOVING):
            self._edit_collapsible.expand()

    def _on_edit_widget_toggled(self, expanded: bool) -> None:
        """Show/hide the edited-frames timeline docked at the bottom."""
        if not expanded:
            if self._edit_dock_widget is not None:
                self._edit_dock_widget.hide()
            return
        self._autoselect_points_layer()
        if self.edit_widget is None:
            self.edit_widget = EditWidget(self._viewer)
            self.edit_widget.set_show_individuals(
                self.edit_controls.show_individuals_checkbox.isChecked()
            )
            self._edit_dock_widget = self._viewer.window.add_dock_widget(
                self.edit_widget, area="bottom", name="edited frames"
            )
        elif self._edit_dock_widget is not None:
            self._edit_dock_widget.show()

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
        if self.edit_widget is not None:
            self.edit_widget.set_show_individuals(checked)

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
