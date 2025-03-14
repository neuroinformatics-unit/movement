"""Unit tests for loader widgets in the napari plugin.

We instantiate the DataLoader widget in each test instead of using a fixture.
This is because mocking widget methods would not work after the widget is
instantiated (the methods would have already been connected to signals).
"""

import pytest
from napari.components.dims import RangeTuple
from napari.layers.points.points import Points
from napari.settings import get_settings
from pytest import DATA_PATHS
from qtpy.QtWidgets import QComboBox, QDoubleSpinBox, QLineEdit, QPushButton

from movement.napari.loader_widgets import DataLoader


# ------------------- tests for widget instantiation--------------------------#
def test_data_loader_widget_instantiation(make_napari_viewer_proxy):
    """Test that the loader widget is properly instantiated."""
    # Instantiate the data loader widget
    data_loader_widget = DataLoader(make_napari_viewer_proxy)

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


# --------test connection between widget buttons and methods------------------#
@pytest.mark.parametrize("button", ["browse", "load"])
def test_button_connected_to_on_clicked(
    make_napari_viewer_proxy, mocker, button
):
    """Test that clicking a button calls the right function."""
    mock_method = mocker.patch(
        f"movement.napari.loader_widgets.DataLoader._on_{button}_clicked"
    )
    data_loader_widget = DataLoader(make_napari_viewer_proxy)
    button = data_loader_widget.findChild(QPushButton, f"{button}_button")
    button.click()
    mock_method.assert_called_once()


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
    data_loader_widget = DataLoader(make_napari_viewer_proxy)
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
def test_dimension_slider_matches_frames(
    valid_dataset_with_localised_nans,
    nan_time_location,
    nan_individuals,
    nan_keypoints,
    make_napari_viewer_proxy,
):
    """Test that the dimension slider is set to the total number of frames
    when data with NaNs is loaded.
    """
    # Get data with nans at the expected locations
    nan_location = {
        "time": nan_time_location,
        "individuals": nan_individuals,
        "keypoints": nan_keypoints,
    }
    file_path, ds = valid_dataset_with_localised_nans(nan_location)

    # Define the expected frame index with the NaN value
    if nan_location["time"] == "start":
        expected_frame = ds.coords["time"][0]
    elif nan_location["time"] == "middle":
        expected_frame = ds.coords["time"][ds.coords["time"].shape[0] // 2]
    elif nan_location["time"] == "end":
        expected_frame = ds.coords["time"][-1]

    # Load the poses loader widget
    viewer = make_napari_viewer_proxy()
    poses_loader_widget = DataLoader(viewer)

    # Read sample data with a NaN at the specified
    # location (start, middle, or end)
    poses_loader_widget.file_path_edit.setText(file_path.as_posix())
    poses_loader_widget.source_software_combo.setCurrentText("DeepLabCut")

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
    poses_loader_widget._on_load_clicked()

    # Check the frame slider is set to the full range of frames
    assert viewer.dims.range[0] == RangeTuple(
        start=0.0, stop=ds.position.shape[0] - 1, step=1.0
    )


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
    loader_widget = DataLoader(viewer)

    # Load data as a points layer
    file_path = pytest.DATA_PATHS.get(filename)
    loader_widget.file_path_edit.setText(file_path.as_posix())
    loader_widget.source_software_combo.setCurrentText(source_software)
    loader_widget._on_load_clicked()

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
