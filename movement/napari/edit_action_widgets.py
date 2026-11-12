"""Subwidgets for handling the edit action UI in the editor sidebar.

Each 'edit action' consists of a subclass of the abstract superclass
``EditActionContainer``, which contains an instance of a QWidget and
a QLayout; common machinery for the edit action UI is handled by
``EditActionContainer`` and action-specific machinery is handled by
the subclass.

To add an edit action, create a subclass of ``EditActionContainer``,
provide implementations for all its abstract methods, and edit
``get_supported_edit_actions()`` to include an instance of the
subclass in the list it returns.
"""

from abc import ABC, abstractmethod

import numpy as np
from napari.utils.notifications import show_warning
from napari.viewer import Viewer
from qtpy.QtCore import Qt
from qtpy.QtGui import QIntValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from movement.napari.layer_wiring import (
    MAX_FRAME_IDX_KEY,
    POINTS_PROPERTIES_KEY,
    active_movement_points_layer,
)


def get_supported_edit_actions(napari_viewer: Viewer):
    """Instantiate a list of supported edit actions.

    Called in ``EditControlsWidget`` to manage a local instance
    of each edit action class.
    """
    return [
        DropPointsEditActionContainer(napari_viewer),
    ]


class EditActionStack(QStackedWidget):
    """Variant of ``QStackedWidget`` for ``EditActionContainer`` widgets.

    Unlike the ``QStackedWidget``, the ``EditActionStack``'s size hints
    are based off of the current widget instead of the largest widget
    in the stack, which prevents the widget from occupying more space than
    it should when a smaller widget is currently selected.
    """

    def sizeHint(self):
        """Get the size hint of the stack's active widget."""
        return self.currentWidget().sizeHint()

    def minimumSizeHint(self):
        """Get the minimum size hint of the stack's active widget."""
        return self.currentWidget().minimumSizeHint()


class EditActionContainer(ABC):
    """Abstract superclass for action-specific sidebar editor controls.

    This superclass instantiates a ``QWidget`` containing UI elements
    common to all edit actions: i.e. the frame selection textboxes and the
    Apply Changes button.

    This superclass also provides abstract methods for subclasses
    to implement details specific to each edit action:
    what widgets beyond defaults should be added to the sidebar, what happens
    when the commit button is pushed, how the widgets should update when the
    active movement points layer changes, and what its label in the
    ``EditControlsWidget``'s edit action dropdown selector is.
    """

    def __init__(self, napari_viewer: Viewer):
        """Initialize common widgets and add all widgets to the layout."""
        super().__init__()
        self.widget = QWidget()  # Container for all other widgets
        self.layout = QVBoxLayout()  # Base layout
        self.viewer = napari_viewer
        self.widget.setLayout(self.layout)
        self.movement_layer = active_movement_points_layer(self.viewer)
        self.viewer.layers.selection.events.changed.connect(
            self._update_core_widgets
        )

        # Subwidgets should be instantiates in the constructor
        self.start_frame_validator = QIntValidator(0, 2147483647)
        self.start_frame_textbox = QLineEdit()
        self.start_frame_textbox.setValidator(self.start_frame_validator)
        self.end_frame_validator = QIntValidator(0, 2147483647)
        self.end_frame_textbox = QLineEdit()
        self.end_frame_textbox.setValidator(self.end_frame_validator)
        self.apply_changes_button = QPushButton("Apply changes")
        self.apply_changes_button.clicked.connect(self.apply_changes_clicked)
        self._add_core_widgets()

    @property
    @abstractmethod
    def combo_box_label(self) -> str:
        """Read-only property holding the dropdown label.

        Subclasses must supply a string.
        """
        raise NotImplementedError(
            "Subclass must implement abstract property of superclass"
        )

    def _update_core_widgets(self):
        """Update widgets when the active movement points layer changes.

        Calls ``update_widgets()``, which is the intended way to update
        ``EditActionContainer`` subclass widgets when the movement points.
        layer changes. Also maintains the reference to the active movement
        points layer and shows/hides the widget stack when there is/isn't
        an active movement points layer respectively.
        """
        if self.movement_layer != active_movement_points_layer(self.viewer):
            self.movement_layer = active_movement_points_layer(self.viewer)

            if self.movement_layer is not None:
                if not self.widget.parentWidget().isVisible():
                    self.widget.parentWidget().setVisible(True)
            elif self.widget.parentWidget().isVisible():
                self.widget.parentWidget().setVisible(False)

            self.update_widgets()

    def _add_core_widgets(self):
        """Add subwidgets to an EditActionContainer.

        This adds frame selection widgets, any action-specific
        widgets as defined by ``add_edit_action_widgets()``, and
        the Apply Changes button widget (in that order).
        """
        # Add frame selection widgets and input validators
        frame_select_layout = QFormLayout()
        frame_select_layout.addRow("Start frame:", self.start_frame_textbox)
        frame_select_layout.addRow("End frame:", self.end_frame_textbox)
        self.layout.addLayout(frame_select_layout)

        # Add custom widgets
        self.add_edit_action_widgets()

        # Add Apply Changes button widget
        self.layout.addWidget(self.apply_changes_button)

    def get_property_list(self, property: str):
        """Get a list of distinct individuals or keypoints in a dataset.

        ``property`` should be one of "individual" or "keypoint".

        Return None if there is no active movement points layer or there
        is no properties metadata.

        Useful for subclasses that need to generate a list of checkbox
        widgets.
        """
        if property not in ["individual", "keypoint"]:
            raise ValueError(
                "get_property_list() accepts one of 'individual' or 'keypoint'"
            )
        if self.movement_layer is None:
            return None
        else:
            if POINTS_PROPERTIES_KEY not in self.movement_layer.metadata:
                return None
            properties = self.movement_layer.metadata.get(
                POINTS_PROPERTIES_KEY, 0
            )
            return list(properties[property].unique())

    def get_frame_range(self):
        """Return the widget's frame range as a tuple of (min,max).

        Return None if there is no minimum or maximum range set.
        """
        # Validate that both start and end frame textboxes have entries
        if self.start_frame_textbox.text() == "":
            return None
        if self.end_frame_textbox.text() == "":
            return None
        return (
            int(self.start_frame_textbox.text()),
            int(self.end_frame_textbox.text()),
        )

    @abstractmethod
    def update_widgets(self):
        """Define widget updates when the active movement points layer changes.

        Called from ``_update_core_widgets()``. Subclasses must implement
        this method to define how they regenerate their layer-dependent
        widgets when the active movement points layer changes.
        """
        raise NotImplementedError(
            "Subclass implement abstract method of superclass"
        )

    @abstractmethod
    def add_edit_action_widgets(self):
        """Add subwidgets for an edit action to an ``EditActionContainer``.

        A subclass must implement this method to add custom ``QWidget``s
        for the UI to ``self.layout``. These widgets should be instantiated
        at the start of the constructor, before calling ``super.__init__()``.
        """
        raise NotImplementedError(
            "Subclass implement abstract method of superclass"
        )

    @abstractmethod
    def apply_changes_clicked(self):
        """Call functions when the Apply Changes button is clicked.

        Subclasses must implement this to define what actions to perform to the
        active ``movement`` points layer based on the user's selections on the
        edit action GUI.
        """
        raise NotImplementedError(
            "Subclass implement abstract method of superclass"
        )


class DropPointsEditActionContainer(EditActionContainer):
    """Subwidget for dropping points based on user criteria.

    Implements subwidgets to allow the selection of a range of frames,
    individuals, and keypoints to drop points over.
    """

    def __init__(self, napari_viewer: Viewer):
        """Instantiate widgets, then call superclass constructor."""
        # Instantiate subwidgets before calling the superclass constructor
        self.individual_checkbox_container = QWidget()
        self.individual_checkbox_layout = QVBoxLayout()
        self.individual_checkbox_container.setLayout(
            self.individual_checkbox_layout
        )
        self.individual_checkbox_select_all = QPushButton(
            "Select/deselect all individuals"
        )
        self.individual_checkbox_select_all.clicked.connect(
            lambda: self.check_or_uncheck_all(self.individual_checkbox_layout)
        )

        self.keypoint_checkbox_container = QWidget()
        self.keypoint_checkbox_layout = QVBoxLayout()
        self.keypoint_checkbox_container.setLayout(
            self.keypoint_checkbox_layout
        )
        self.keypoint_checkbox_select_all = QPushButton(
            "Select/deselect all keypoints"
        )
        self.keypoint_checkbox_select_all.clicked.connect(
            lambda: self.check_or_uncheck_all(self.keypoint_checkbox_layout)
        )

        super().__init__(napari_viewer=napari_viewer)

    @property
    def combo_box_label(self):
        """The label in the sidebar's edit action dropdown selector."""
        return "Drop Points"

    def check_or_uncheck_all(self, checkbox_layout: QLayout):
        """Given a layout, checks all ``QCheckBox`` children of that layout.

        If all QCheckBoxes were already checked, unchecks them instead.
        """
        # Iterate over all subwidgets and check unchecked widgets
        # Mark whether any widgets were checked during this
        # Only check/uncheck if there is more than one checkbox
        # as single-item layouts are always checked by default
        if checkbox_layout.count() > 1:
            subwidget_checked = False
            for i in range(0, checkbox_layout.count()):
                widget = checkbox_layout.itemAt(i).widget()
                if (
                    isinstance(widget, QCheckBox)
                    and widget.checkState() == Qt.Unchecked
                ):
                    widget.setCheckState(Qt.Checked)
                    subwidget_checked = True

            # If no subwidgets were checked, then uncheck them all
            if not subwidget_checked:
                for i in range(0, checkbox_layout.count()):
                    widget = checkbox_layout.itemAt(i).widget()
                    if isinstance(widget, QCheckBox):
                        widget.setCheckState(Qt.Unchecked)

    def _clear_checkboxes(self, layout: QLayout):
        """Clear checkboxes from a layout.

        Used to reset the list of checkboxes for individuals and
        keypoints when the active layer changes.
        """
        # Iterate
        for i in range(layout.count() - 1, -1, -1):
            widget = layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                layout.removeWidget(widget)
                widget.setParent(None)

    def update_widgets(self):
        """Refresh the contents of the checkbox ``QListWidget``s.

        Called when the active movement points layer is changed. Regenerates
        checkbox widgets to match the individuals and keypoints of the new
        layer. Defaults to unchecked boxes if there are multiple of an
        individual/keypoint or one checked uninteractable one if there is
        only one individual or keypoint.
        """
        # Clear contents of widget lists when updating and make sure
        # widget is enabled
        self._clear_checkboxes(self.individual_checkbox_layout)
        self._clear_checkboxes(self.keypoint_checkbox_layout)

        # Add items to widget lists if there is a points layer
        if self.movement_layer is not None:
            individual_list = self.get_property_list("individual")
            keypoint_list = self.get_property_list("keypoint")
            if individual_list is not None and keypoint_list is not None:
                for i in individual_list:
                    item = QCheckBox(i)
                    if len(individual_list) > 1:
                        item.setCheckState(Qt.Unchecked)
                        self.individual_checkbox_layout.addWidget(item)
                    else:
                        item.setCheckState(Qt.Checked)
                        item.setDisabled(True)
                        self.individual_checkbox_layout.addWidget(item)
                for i in keypoint_list:
                    item = QCheckBox(i)
                    if len(keypoint_list) > 1:
                        item.setCheckState(Qt.Unchecked)
                        self.keypoint_checkbox_layout.addWidget(item)
                    else:
                        item.setCheckState(Qt.Checked)
                        item.setDisabled(True)
                        self.keypoint_checkbox_layout.addWidget(item)

    def add_edit_action_widgets(self):
        """Add checkbox selection widgets and labels to the sidebar."""
        self.layout.addWidget(QLabel("Select individuals"))
        self.layout.addWidget(self.individual_checkbox_container)
        self.layout.addWidget(self.individual_checkbox_select_all)
        self.layout.addWidget(QLabel("Select keypoints"))
        self.layout.addWidget(self.keypoint_checkbox_container)
        self.layout.addWidget(self.keypoint_checkbox_select_all)

    def _validate_frame_range(self, range: tuple | None):
        frame_range = self.get_frame_range()
        if frame_range is None:
            show_warning(
                "Start or end frame textbox is empty, no action performed"
            )
            return False
        if frame_range[0] > frame_range[1]:
            show_warning(
                "Start frame cannot be after end frame, no action performed"
            )
            return False
        if frame_range[1] > self.movement_layer.metadata.get(
            MAX_FRAME_IDX_KEY, 0
        ):
            show_warning(
                "End frame must precede layer's max frame, no action performed"
            )
            return False
        return True

    def apply_changes_clicked(self):
        """Remove points from the ``napari`` editor based on sidebar inputs.

        Specifically, generates a list of point indexes to remove based on
        the start/end frame textboxes and the individual/keypoint checkboxes,
        then calls ``remove()`` on the active movement points layer using
        this list.
        """
        # Make sure there's a valid movement points layer
        if self.movement_layer is None:
            show_warning(
                "No active movement points layer, no action performed"
            )
            return

        # Get and validate frame range
        # QIntValidator already makes sure the user can only input
        # positive integers, but more checks are needed
        frame_range = self.get_frame_range()
        if not self._validate_frame_range(frame_range):
            return

        # Get list of individuals to drop
        individuals = []
        for i in range(0, self.individual_checkbox_layout.count()):
            widget = self.individual_checkbox_layout.itemAt(i).widget()
            if (
                isinstance(widget, QCheckBox)
                and widget.checkState() == Qt.Checked
            ):
                individuals.append(widget.text())
        if len(individuals) == 0:
            show_warning("No individuals selected, no action performed")
            return

        # Get list of keypoints to drop
        keypoints = []
        for i in range(0, self.keypoint_checkbox_layout.count()):
            widget = self.keypoint_checkbox_layout.itemAt(i).widget()
            if (
                isinstance(widget, QCheckBox)
                and widget.checkState() == Qt.Checked
            ):
                keypoints.append(widget.text())
        if len(keypoints) == 0:
            show_warning("No keypoints selected, no action performed")
            return

        # Get a list of all indices which match the frame, individual,
        # and keypoint to drop
        frame_mask = self.movement_layer.properties["time"]
        frame_mask = (frame_mask >= frame_range[0]) & (
            frame_mask <= frame_range[1]
        )
        individual_mask = np.isin(
            self.movement_layer.properties["individual"], individuals
        )
        keypoint_mask = np.isin(
            self.movement_layer.properties["keypoint"], keypoints
        )
        idx_mask = np.where(
            np.logical_and(
                np.logical_and(frame_mask, individual_mask), keypoint_mask
            )
        )[0].tolist()

        # Warn the user if there are no points matching the frames,
        # individuals, and keypoints.
        if len(idx_mask) == 0:
            show_warning(
                "No points match selection criteria, no action performed"
            )
            return

        # Select and drop qualifying points
        self.movement_layer.remove(idx_mask)
