"""Keybindings for ``movement`` widgets."""

import napari

from movement.napari.edit_timeline_widget import EditControlsWidget

viewer = napari.current_viewer()


def init_keybindings(plugin_context):
    """Initialize keybinds and set method for de-initializing them.

    Called from ``napari.yaml`` ``on_activate`` entrypoint.
    """
    viewer.bind_key("X", lambda x: select_min_frame())
    viewer.bind_key("Shift-X", lambda x: select_max_frame())

    plugin_context.register_disposable(clear_keybindings)


def clear_keybindings():
    """Clear keybinds when the ``movement`` plugin is closed.

    Note, closing the ``movement`` meta-widget does not close the plugin
    itself.
    """
    viewer.bind_key("x", None, overwrite=True)


def select_min_frame():
    """Set the end frame of an ``EditActionWidget``'s open edit action."""
    if not _is_movement_open():
        return
    _set_edit_action_frame(_get_edit_widget(), True)


def select_max_frame():
    """Set the end frame of an ``EditActionWidget``'s open edit action."""
    if not _is_movement_open():
        return
    _set_edit_action_frame(_get_edit_widget(), False)


def _is_movement_open():
    """Return whether or not the movement metawidget is open."""
    try:
        viewer.window.dock_widgets["movement (movement)"]
    except KeyError:
        return False
    return True


def _get_edit_widget():
    if not _is_movement_open():
        return None

    movement_widget = viewer.window.dock_widgets["movement (movement)"]
    return movement_widget.findChild(EditControlsWidget)


def _set_edit_action_frame(
    edit_controls: EditControlsWidget,
    start_frame: bool = True,
) -> None:
    """Set the selected edit action's start/end frame to the current frame.

    Aborts if there is no selected edit action or it has no associated
    ``movement`` points layer.
    """
    # Make sure an edit action is actually selected
    if edit_controls.edit_action_stack.currentIndex() == 0:
        return
    # Get that edit action's container if one is
    container = edit_controls.valid_edit_actions[
        edit_controls.edit_action_stack.currentIndex() - 1
    ]
    # Terminate early if it isn't associated with a movement layer
    if container.movement_layer is None:
        return
    # Set start frame by default to current frame
    # or end frame if start_frame == False
    else:
        idx = viewer.dims.current_step[0]
        if start_frame:
            container.start_frame_textbox.setText(str(idx))
        else:
            container.end_frame_textbox.setText(str(idx))
