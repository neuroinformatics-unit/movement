"""Unit tests for loader widgets in the napari plugin.

We instantiate the DataLoader widget in each test instead of using a fixture.
This is because mocking widget methods would not work after the widget is
instantiated (the methods would have already been connected to signals).
"""

from contextlib import nullcontext as does_not_raise

import pytest
from napari.components.dims import RangeTuple
from napari.layers import (
    Image,
    Labels,
    Points,
    Shapes,
    Surface,
    Tracks,
    Vectors,
)
from napari.settings import get_settings
from pytest import DATA_PATHS
from qtpy.QtWidgets import QComboBox, QDoubleSpinBox, QLineEdit, QPushButton

from movement.napari.loader_widgets import DataLoader


# ------------------- tests for widget instantiation--------------------------#
def test_data_loader_widget_instantiation(make_napari_viewer_proxy):
    """Test that the loader widget is properly instantiated."""
    # Instantiate the data loader widget
    data_loader_widget = DataLoader(make_napari_viewer_proxy())

    # Check that the widget has the expected number of rows
    assert data_loader_widget.layout().rowCount() == 4

    # Check that the expected widgets are present in the layout
    expected_widgets = [
        (QComboBox, "source_software_combo"),
        (QDoubleSpinBox, "fps_spinbox"),
        (QLineEdit, "file_path_edit"),
        (QPushButton, "load_button"),
        (QPushButton, "browse_button"),
    ]
    assert all(
        data_loader_widget.findChild(widget_type, widget_name) is not None
        for widget_type, widget_name in expected_widgets
    ), "Some widgets are missing."

    # Make sure that layer tooltips are enabled
    assert get_settings().appearance.layer_tooltip_visibility is True

    # Test methods are connected to layer events
    assert all(
        [
            data_loader_widget._update_frame_slider_range.__name__
            in event.callback_refs
            for event in [
                data_loader_widget.viewer.layers.events.inserted,
                data_loader_widget.viewer.layers.events.removed,
            ]
        ]
    )


# --------test connection between widget methods and buttons/events----------#
@pytest.mark.parametrize("button", ["browse", "load"])
def test_button_connected_to_on_clicked(
    make_napari_viewer_proxy, mocker, button
):
    """Test that clicking a button calls the right function."""
    mock_method = mocker.patch(
        f"movement.napari.loader_widgets.DataLoader._on_{button}_clicked"
    )
    data_loader_widget = DataLoader(make_napari_viewer_proxy())
    button = data_loader_widget.findChild(QPushButton, f"{button}_button")
    button.click()
    mock_method.assert_called_once()


@pytest.mark.parametrize(
    "layer_type",
    [
        Points,
        Image,
        Tracks,
        Labels,
        Shapes,
        Surface,
        Vectors,
    ],
)
def test_on_layer_added_and_deleted(
    layer_type, sample_layer_data, make_napari_viewer_proxy, mocker
):
    """Test the frame slider update is called when a layer is added/removed."""
    # Create a mock napari viewer
    data_loader_widget = DataLoader(make_napari_viewer_proxy())

    # Mock the frame slider check method
    mock_frame_slider_check = mocker.patch(
        "movement.napari.loader_widgets.DataLoader._update_frame_slider_range"
    )

    # Add a sample layer to the viewer
    mock_layer = layer_type(
        data=sample_layer_data[layer_type.__name__],
        name="mock_layer",
    )
    data_loader_widget.viewer.add_layer(mock_layer)

    # Check that the slider check method was called once
    mock_frame_slider_check.assert_called_once()
    mock_frame_slider_check.reset_mock()

    # Delete the layer
    data_loader_widget.viewer.layers.remove(mock_layer)

    # Check that the slider check method was called once
    mock_frame_slider_check.assert_called_once()


# ------------------- tests for widget methods--------------------------------#
# In these tests we check if calling a widget method has the expected effects


@pytest.mark.parametrize(
    "file_path",
    [
        str(
            DATA_PATHS.get("DLC_single-wasp.predictions.h5").parent
        ),  # valid file path poses
        str(
            DATA_PATHS.get("VIA_single-crab_MOCA-crab-1.csv").parent
        ),  # valid file path bboxes
        "",  # empty string, simulate user canceling the dialog
    ],
)
def test_on_browse_clicked(file_path, make_napari_viewer_proxy, mocker):
    """Test that the _on_browse_clicked method correctly sets the
    file path in the QLineEdit widget (file_path_edit).
    The file path is provided by mocking the return of the
    QFileDialog.getOpenFileName method.
    """
    # Instantiate the napari viewer and the data loader widget
    viewer = make_napari_viewer_proxy()
    data_loader_widget = DataLoader(viewer)

    # Mock the QFileDialog.getOpenFileName method to return the file path
    mocker.patch(
        "movement.napari.loader_widgets.QFileDialog.getOpenFileName",
        return_value=(file_path, None),  # tuple(file_path, filter)
    )
    # Simulate the user clicking the 'Browse' button
    data_loader_widget._on_browse_clicked()
    # Check that the file path edit text has been updated
    assert data_loader_widget.file_path_edit.text() == file_path


@pytest.mark.parametrize(
    "source_software, expected_file_filter",
    [
        ("DeepLabCut", "*.h5 *.csv"),
        ("SLEAP", "*.h5 *.slp"),
        ("LightningPose", "*.csv"),
        ("VIA-tracks", "*.csv"),
    ],
)
def test_file_filters_per_source_software(
    source_software, expected_file_filter, make_napari_viewer_proxy, mocker
):
    """Test that the file dialog is opened with the correct filters."""
    data_loader_widget = DataLoader(make_napari_viewer_proxy())
    data_loader_widget.source_software_combo.setCurrentText(source_software)
    mock_file_dialog = mocker.patch(
        "movement.napari.loader_widgets.QFileDialog.getOpenFileName",
        return_value=("", None),
    )
    data_loader_widget._on_browse_clicked()
    mock_file_dialog.assert_called_once_with(
        data_loader_widget,
        caption="Open file containing tracked data",
        filter=f"Valid data files ({expected_file_filter})",
    )


def test_on_load_clicked_without_file_path(make_napari_viewer_proxy, capsys):
    """Test that clicking 'Load' without a file path shows a warning."""
    # Instantiate the napari viewer and the data loader widget
    viewer = make_napari_viewer_proxy()
    data_loader_widget = DataLoader(viewer)
    # Call the _on_load_clicked method (pretend the user clicked "Load")
    data_loader_widget._on_load_clicked()
    captured = capsys.readouterr()
    assert "No file path specified." in captured.out


@pytest.mark.parametrize(
    "filename, source_software, tracks_array_shape",
    [
        ("DLC_single-wasp.predictions.h5", "DeepLabCut", (2170, 4)),
        ("VIA_single-crab_MOCA-crab-1.csv", "VIA-tracks", (35, 4)),
    ],
)
def test_on_load_clicked_with_valid_file_path(
    filename,
    source_software,
    tracks_array_shape,
    make_napari_viewer_proxy,
    caplog,
):
    """Test clicking 'Load' with a valid file path.

    This test checks that the `_on_load_clicked` method causes the following:
    - creates the `data`, `props`, and `file_name` attributes
    - emits a series of expected log messages
    - adds a Points layer to the viewer (with the expected name)
    - sets the playback fps to the specified value
    """
    # Instantiate the napari viewer and the data loader widget
    viewer = make_napari_viewer_proxy()
    data_loader_widget = DataLoader(viewer)

    # Set the file path to a valid file
    file_path = pytest.DATA_PATHS.get(filename)
    data_loader_widget.file_path_edit.setText(file_path.as_posix())

    # Set the source software
    data_loader_widget.source_software_combo.setCurrentText(source_software)

    # Set the fps to 60
    data_loader_widget.fps_spinbox.setValue(60)

    # Call the _on_load_clicked method (pretend the user clicked "Load")
    data_loader_widget._on_load_clicked()

    # Check that class attributes have been created
    assert data_loader_widget.file_name == file_path.name
    assert data_loader_widget.data is not None
    assert data_loader_widget.props is not None

    # Check that the expected log messages were emitted
    expected_log_messages = {
        "Converted dataset to a napari Tracks array.",
        f"Tracks array shape: {tracks_array_shape}",
        "Added tracked dataset as a napari Points layer.",
    }
    log_messages = {record.getMessage() for record in caplog.records}
    assert expected_log_messages <= log_messages

    # Check that a Points layer was added to the viewer
    points_layer = data_loader_widget.viewer.layers[0]
    assert points_layer.name == f"data: {file_path.name}"


# ------------------- tests for dimension slider ----------------------------#
# These tests check that the frame slider is set to the expected range
@pytest.mark.parametrize(
    "nan_time_location",
    ["start", "middle", "end"],
)
@pytest.mark.parametrize(
    "nan_individuals",
    [["id_0"], ["id_0", "id_1"]],
    ids=["one_individual", "all_individuals"],
)
@pytest.mark.parametrize(
    "nan_keypoints",
    [["centroid"], ["centroid", "left", "right"]],
    ids=["one_keypoint", "all_keypoints"],
)
def test_dimension_slider_with_nans(
    valid_poses_path_and_ds_with_localised_nans,
    nan_time_location,
    nan_individuals,
    nan_keypoints,
    make_napari_viewer_proxy,
):
    """Test that the dimension slider is set to the total number of frames
    when data layers with NaNs are loaded.
    """
    # Get data with nans at the expected locations
    nan_location = {
        "time": nan_time_location,
        "individuals": nan_individuals,
        "keypoints": nan_keypoints,
    }
    file_path, ds = valid_poses_path_and_ds_with_localised_nans(nan_location)

    # Define the expected frame index with the NaN value
    if nan_location["time"] == "start":
        expected_frame = ds.coords["time"][0]
    elif nan_location["time"] == "middle":
        expected_frame = ds.coords["time"][ds.coords["time"].shape[0] // 2]
    elif nan_location["time"] == "end":
        expected_frame = ds.coords["time"][-1]

    # Load the data loader widget
    viewer = make_napari_viewer_proxy()
    data_loader_widget = DataLoader(viewer)

    # Read sample data with a NaN at the specified
    # location (start, middle, or end)
    data_loader_widget.file_path_edit.setText(file_path.as_posix())
    data_loader_widget.source_software_combo.setCurrentText("DeepLabCut")

    # Check the data contains nans where expected
    assert (
        ds.position.sel(
            individuals=nan_location["individuals"],
            keypoints=nan_location["keypoints"],
            time=expected_frame,
        )
        .isnull()
        .all()
    )

    # Call the _on_load_clicked method
    # (to pretend the user clicked "Load")
    data_loader_widget._on_load_clicked()

    # Check the frame slider is set to the full range of frames
    assert viewer.dims.range[0] == RangeTuple(
        start=0.0, stop=ds.position.shape[0] - 1, step=1.0
    )


@pytest.mark.parametrize(
    "list_input_data_files",
    [
        ["valid_poses_path_and_ds", "valid_poses_path_and_ds_short"],
        ["valid_poses_path_and_ds_short", "valid_poses_path_and_ds"],
    ],
    ids=["long_first", "short_first"],
)
def test_dimension_slider_multiple_files(
    list_input_data_files, make_napari_viewer_proxy, request
):
    """Test that the dimension slider is set to the maximum number of frames
    when multiple data layers are loaded.
    """
    # Get the datasets to load (paths and ds)
    list_paths, list_datasets = [
        [
            request.getfixturevalue(file_name)[j]
            for file_name in list_input_data_files
        ]
        for j in range(len(list_input_data_files))
    ]

    # Get the maximum number of frames from all datasets
    max_frames = max(ds.sizes["time"] for ds in list_datasets)

    # Load the data loader widget
    viewer = make_napari_viewer_proxy()
    data_loader_widget = DataLoader(viewer)

    # Load each dataset in order
    for file_path in list_paths:
        data_loader_widget.file_path_edit.setText(file_path.as_posix())
        data_loader_widget.source_software_combo.setCurrentText("DeepLabCut")
        data_loader_widget._on_load_clicked()

    # Check the frame slider is as expected
    assert viewer.dims.range[0] == RangeTuple(
        start=0.0, stop=max_frames - 1, step=1.0
    )

    # Check the maximum number of frames is the number of frames
    # in the longest dataset
    _, ds_long = request.getfixturevalue("valid_poses_path_and_ds")
    assert max_frames == ds_long.sizes["time"]


@pytest.mark.parametrize(
    "list_input_data_files",
    [
        [
            "valid_poses_path_and_ds",
            "valid_poses_path_and_ds_nan_start",
        ],  # one with NaNs at the start remains after deletion
        [
            "valid_poses_path_and_ds",
            "valid_poses_path_and_ds_nan_end",
        ],  # one with NaNs at the end remains after deletion
        [
            "valid_poses_path_and_ds",
            "valid_poses_path_and_ds_nan_start",
            "valid_poses_path_and_ds_nan_end",
        ],  # two remain after deletion, with NaNs at the start and end
        [
            "valid_poses_path_and_ds",
            "valid_poses_path_and_ds_short",
            "valid_poses_path_and_ds_nan_start",
        ],  # two remain after deletion, the longest one with NaNs at the start
        [
            "valid_poses_path_and_ds",
            "valid_poses_path_and_ds_short",
            "valid_poses_path_and_ds_nan_end",
        ],  # two remain after deletion, the longest one with NaNs at the end
    ],
)
def test_dimension_slider_with_deletion(
    list_input_data_files,
    make_napari_viewer_proxy,
    request,
):
    """Test that the dimension slider is set to the correct range of frames
    when loading two point layers, deleting one, and the remaining layer(s)
    have all NaN values at the start or end.
    """
    # Get the input data to load (paths and ds)
    list_paths, list_datasets = [
        [
            request.getfixturevalue(file_name)[j]
            for file_name in list_input_data_files
        ]
        for j in range(2)
    ]

    # Check the expected number of datasets have NaN values
    # at the start or end
    expected_datasets_with_nans = sum(
        ["nan" in file_name for file_name in list_input_data_files]
    )
    actual_datasets_with_nans = sum(
        [
            any(
                [
                    ds.position.sel(time=ds.coords["time"][0])
                    .isnull()
                    .all()
                    .values,
                    ds.position.sel(time=ds.coords["time"][-1])
                    .isnull()
                    .all()
                    .values,
                ]
            )
            for ds in list_datasets
        ]
    )
    assert actual_datasets_with_nans == expected_datasets_with_nans

    # Get the maximum number of frames from all datasets
    max_frames = max(ds.sizes["time"] for ds in list_datasets)

    # Load each dataset as a points layer in napari
    viewer = make_napari_viewer_proxy()
    data_loader_widget = DataLoader(viewer)
    for file_path in list_paths:
        data_loader_widget.file_path_edit.setText(file_path.as_posix())
        data_loader_widget.source_software_combo.setCurrentText("DeepLabCut")
        data_loader_widget._on_load_clicked()

    # Check the frame slider after loading all data
    assert viewer.dims.range[0] == RangeTuple(
        start=0.0, stop=max_frames - 1, step=1.0
    )

    # Remove the first loaded layer
    viewer.layers.remove(viewer.layers[0])

    # Update maximum number of frames from the remaining layers
    max_frames = max(ds.sizes["time"] for ds in list_datasets[1:])

    # Check the frame slider after deletion
    assert viewer.dims.range[0] == RangeTuple(
        start=0.0, stop=max_frames - 1, step=1.0
    )


@pytest.mark.parametrize(
    "layer_type",
    [
        Points,
        Image,
        Tracks,
    ],
)
def test_dimension_slider_with_layer_types(
    layer_type, sample_layer_data, make_napari_viewer_proxy
):
    """Test the slider update attends to all the expected layer types."""
    # Create a mock napari viewer
    viewer = make_napari_viewer_proxy()
    data_loader_widget = DataLoader(viewer)

    # Load a sample dataset as a points layer
    file_path = pytest.DATA_PATHS.get("DLC_single-wasp.predictions.h5")
    data_loader_widget.file_path_edit.setText(file_path.as_posix())
    data_loader_widget.source_software_combo.setCurrentText("DeepLabCut")
    data_loader_widget._on_load_clicked()

    # Get number of frames in pose data
    n_frames_data = viewer.layers[0].metadata["max_frame_idx"]

    # Load mock data as the relevant layer type
    mock_layer = layer_type(
        data=sample_layer_data[layer_type.__name__],
        name="mock_layer",
    )
    viewer.add_layer(mock_layer)

    # Check mock data is defined for more frames than the pose data
    assert sample_layer_data["n_frames"] > n_frames_data

    # Check the frame slider is set to the max number of frames of the
    # mock data
    assert viewer.dims.range[0] == RangeTuple(
        start=0.0, stop=sample_layer_data["n_frames"] - 1, step=1.0
    )


def test_deletion_all_layers(make_napari_viewer_proxy):
    """Test there are no errors when all layers are deleted."""
    # Load the data loader widget
    viewer = make_napari_viewer_proxy()
    data_loader_widget = DataLoader(viewer)

    # Load a dataset
    file_path = pytest.DATA_PATHS.get("DLC_single-wasp.predictions.h5")
    data_loader_widget.file_path_edit.setText(file_path.as_posix())
    data_loader_widget.source_software_combo.setCurrentText("DeepLabCut")
    data_loader_widget._on_load_clicked()

    # Delete all layers
    with does_not_raise():
        viewer.layers.clear()


# ------------------- tests for layers style ----------------------------#
@pytest.mark.parametrize(
    (
        "filename, source_software, "
        "expected_text_property, expected_color_property"
    ),
    [
        (
            "VIA_multiple-crabs_5-frames_labels.csv",
            "VIA-tracks",
            "individual",
            "individual",
        ),
        (
            "SLEAP_single-mouse_EPM.predictions.slp",
            "SLEAP",
            "keypoint",
            "keypoint",
        ),
        (
            "DLC_two-mice.predictions.csv",
            "DeepLabCut",
            "keypoint",
            "individual",
        ),
        (
            "SLEAP_three-mice_Aeon_mixed-labels.analysis.h5",
            "SLEAP",
            "individual",
            "individual",
        ),
    ],
    ids=[
        "multiple individuals, no keypoints",
        "single individual, multiple keypoints",
        "multiple individuals, multiple keypoints",
        "multiple individuals, one keypoint",
    ],
)
def test_add_points_layer_style(
    filename,
    source_software,
    make_napari_viewer_proxy,
    expected_text_property,
    expected_color_property,
    caplog,
):
    """Test that the Points layer is added to the viewer with the markers
    and text following the expected properties.
    """
    # Instantiate the napari viewer and the data loader widget
    viewer = make_napari_viewer_proxy()
    data_loader_widget = DataLoader(viewer)

    # Load data as a points layer
    file_path = pytest.DATA_PATHS.get(filename)
    data_loader_widget.file_path_edit.setText(file_path.as_posix())
    data_loader_widget.source_software_combo.setCurrentText(source_software)
    data_loader_widget._on_load_clicked()

    # Check no warnings were emitted
    log_messages = {record.getMessage() for record in caplog.records}
    assert not any("Warning" in message for message in log_messages)

    # Get the points layer
    points_layer = next(
        layer for layer in viewer.layers if isinstance(layer, Points)
    )

    # Check the color of markers and text follows the expected property
    assert points_layer._face.color_properties.name == expected_color_property
    assert points_layer.text.color.feature == expected_color_property

    # Check the text follows the expected property
    assert points_layer.text.string.feature == expected_text_property
