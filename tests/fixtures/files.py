"""Valid and invalid file fixtures.

Note: NWB file fixtures are in tests/fixtures/nwb.py.
"""

import os
from pathlib import Path
from unittest.mock import mock_open, patch

import h5py
import pytest
import xarray as xr
from sleap_io.io.slp import read_labels, write_labels
from sleap_io.model.labels import LabeledFrame, Labels


# ------------------ Generic file fixtures ----------------------
@pytest.fixture
def unreadable_file(tmp_path):
    """Return the path to an unreadable .h5 file."""
    file_path = tmp_path / "unreadable.h5"
    file_mock = mock_open()
    file_mock.return_value.read.side_effect = PermissionError
    with (
        patch("builtins.open", side_effect=file_mock),
        patch.object(Path, "exists", return_value=True),
    ):
        yield file_path


@pytest.fixture
def unwriteable_file(tmp_path):
    """Return the path to an unwriteable .h5 file."""
    unwriteable_dir = tmp_path / "no_write"
    unwriteable_dir.mkdir()
    original_access = os.access

    def mock_access(path, mode):
        if path == unwriteable_dir and mode == os.W_OK:
            return False
        # Ensure that the original access function is called
        # for all other cases
        return original_access(path, mode)

    with patch("os.access", side_effect=mock_access):
        file_path = unwriteable_dir / "unwriteable.h5"
        yield file_path


@pytest.fixture
def wrong_extension_file(tmp_path):
    """Return the path to a file with unsupported extension."""
    file_path = tmp_path / "wrong_extension.txt"
    with open(file_path, "w") as f:
        f.write("")
    return file_path


@pytest.fixture
def nonexistent_file(tmp_path):
    """Return the path to a nonexistent file."""
    file_path = tmp_path / "nonexistent.h5"
    return file_path


@pytest.fixture
def data_as_list_h5_file(tmp_path):
    """Return the path to a .h5 file with "data_as_list" dataset."""
    file_path = tmp_path / "no_dataframe.h5"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data_as_list", data=[1, 2, 3])
    return file_path


@pytest.fixture
def fake_h5_file(tmp_path):
    """Return the path to a .h5 file that is not in HDF5 format."""
    file_path = tmp_path / "fake.h5"
    with open(file_path, "w") as f:
        f.write("")
    return file_path


@pytest.fixture
def invalid_single_individual_csv_file(tmp_path):
    """Return the path to a fake single-individual .csv file."""
    file_path = tmp_path / "fake_single_individual.csv"
    with open(file_path, "w") as f:
        f.write("scorer,columns\nsome,columns\ncoords,columns\n")
        f.write("1,2")
    return file_path


@pytest.fixture
def invalid_multi_individual_csv_file(tmp_path):
    """Return the path to a fake multi-individual .csv file."""
    file_path = tmp_path / "fake_multi_individual.csv"
    with open(file_path, "w") as f:
        f.write(
            "scorer,columns\nindividuals,columns\nbodyparts,columns\nsome,columns\n"
        )
        f.write("1,2")
    return file_path


@pytest.fixture
def wrong_extension_new_file(tmp_path):
    """Return the path to a new file with unsupported extension."""
    return tmp_path / "wrong_extension_new_file.txt"


@pytest.fixture
def directory(tmp_path):
    """Return the path to a directory."""
    file_path = tmp_path / "directory"
    file_path.mkdir()
    return file_path


@pytest.fixture
def readable_csv_file(tmp_path):
    """Return the path to a readable .csv file."""
    file_path = tmp_path / "readable.csv"
    with open(file_path, "w") as f:
        f.write("header1,header2\n1,2\n3,4")
    return file_path


@pytest.fixture
def new_h5_file(tmp_path):
    """Return the path to a new .h5 file."""
    return tmp_path / "new_file.h5"


@pytest.fixture
def new_csv_file(tmp_path):
    """Return the path to a new .csv file."""
    return tmp_path / "new_file.csv"


# ---------------- Anipose file fixtures ----------------------------
@pytest.fixture
def missing_keypoint_columns_anipose_csv_file(tmp_path):
    """Return the path to a single-individual anipose .csv file
    missing the z-coordinate of keypoint kp0 "kp0_z".
    """
    file_path = tmp_path / "missing_keypoint_columns.csv"
    columns = [
        "fnum",
        "center_0",
        "center_1",
        "center_2",
        "M_00",
        "M_01",
        "M_02",
        "M_10",
        "M_11",
        "M_12",
        "M_20",
        "M_21",
        "M_22",
    ]
    # Here we are missing kp0_z:
    columns.extend(["kp0_x", "kp0_y", "kp0_score", "kp0_error", "kp0_ncams"])
    with open(file_path, "w") as f:
        f.write(",".join(columns))
        f.write("\n")
        f.write(",".join(["1"] * len(columns)))
    return file_path


@pytest.fixture
def spurious_column_anipose_csv_file(tmp_path):
    """Return the path to a single-individual anipose .csv file
    with an unexpected column.
    """
    file_path = tmp_path / "spurious_column.csv"
    columns = [
        "fnum",
        "center_0",
        "center_1",
        "center_2",
        "M_00",
        "M_01",
        "M_02",
        "M_10",
        "M_11",
        "M_12",
        "M_20",
        "M_21",
        "M_22",
    ]
    columns.extend(["funny_column"])
    with open(file_path, "w") as f:
        f.write(",".join(columns))
        f.write("\n")
        f.write(",".join(["1"] * len(columns)))
    return file_path


# ---------------- DeepLabCut file fixtures ----------------------------
@pytest.fixture
def dlc_h5_file():
    """Return the path to a DeepLabCut .h5 file."""
    return pytest.DATA_PATHS.get("DLC_single-wasp.predictions.h5")


@pytest.fixture
def dlc_csv_file():
    """Return the path to a DeepLabCut .csv file."""
    return pytest.DATA_PATHS.get("DLC_single-wasp.predictions.csv")


# ---------------- SLEAP file fixtures ----------------------------
@pytest.fixture(
    params=[
        "SLEAP_single-mouse_EPM.analysis.h5",
        "SLEAP_single-mouse_EPM.predictions.slp",
        "SLEAP_three-mice_Aeon_proofread.analysis.h5",
        "SLEAP_three-mice_Aeon_proofread.predictions.slp",
        "SLEAP_three-mice_Aeon_mixed-labels.analysis.h5",
        "SLEAP_three-mice_Aeon_mixed-labels.predictions.slp",
    ]
)
def sleap_file(request):
    """Return the path to a SLEAP .h5 or .slp file."""
    return pytest.DATA_PATHS.get(request.param)


@pytest.fixture
def sleap_slp_file():
    """Return the path to a SLEAP .slp file."""
    return pytest.DATA_PATHS.get(
        "SLEAP_three-mice_Aeon_proofread.predictions.slp"
    )


@pytest.fixture
def sleap_analysis_file():
    """Return the path to a SLEAP analysis .h5 file."""
    return pytest.DATA_PATHS.get("SLEAP_three-mice_Aeon_proofread.analysis.h5")


@pytest.fixture
def sleap_slp_file_without_tracks(tmp_path):
    """Mock and return the path to a SLEAP .slp file without tracks."""
    sleap_file = pytest.DATA_PATHS.get(
        "SLEAP_single-mouse_EPM.predictions.slp"
    )
    labels = read_labels(sleap_file)
    file_path = tmp_path / "track_is_none.slp"
    lfs = []
    for lf in labels.labeled_frames:
        instances = []
        for inst in lf.instances:
            inst.track = None
            inst.tracking_score = 0
            instances.append(inst)
        lfs.append(
            LabeledFrame(
                video=lf.video, frame_idx=lf.frame_idx, instances=instances
            )
        )
    write_labels(
        file_path,
        Labels(
            labeled_frames=lfs,
            videos=labels.videos,
            skeletons=labels.skeletons,
        ),
    )
    return file_path


@pytest.fixture
def sleap_h5_file_without_tracks(tmp_path):
    """Mock and return the path to a SLEAP .h5 file without tracks."""
    sleap_file = pytest.DATA_PATHS.get("SLEAP_single-mouse_EPM.analysis.h5")
    file_path = tmp_path / "track_is_none.h5"
    with h5py.File(sleap_file, "r") as f1, h5py.File(file_path, "w") as f2:
        for key in list(f1.keys()):
            if key == "track_names":
                f2.create_dataset(key, data=[])
            else:
                f1.copy(key, f2, name=key)
    return file_path


@pytest.fixture(
    params=[
        "sleap_h5_file_without_tracks",
        "sleap_slp_file_without_tracks",
    ]
)
def sleap_file_without_tracks(request):
    """Fixture to parametrize the SLEAP files without tracks."""
    return request.getfixturevalue(request.param)


# ---------------- VIA tracks CSV file fixtures ----------------------------
via_tracks_csv_file_valid_header = (
    "filename,file_size,file_attributes,region_count,"
    "region_id,region_shape_attributes,region_attributes\n"
)


@pytest.fixture
def via_tracks_csv():
    """Return the path to a VIA tracks .csv file."""
    return pytest.DATA_PATHS.get("VIA_single-crab_MOCA-crab-1.csv")


@pytest.fixture
def invalid_via_tracks_csv_file(tmp_path, request):
    """Return the path to an invalid VIA tracks .csv file."""

    def _invalid_via_tracks_csv_file(invalid_content):
        file_path = tmp_path / "invalid_via_tracks.csv"
        with open(file_path, "w") as f:
            f.write(request.getfixturevalue(invalid_content))
        return file_path

    return _invalid_via_tracks_csv_file


@pytest.fixture
def via_invalid_header():
    """Return the content of a VIA tracks .csv file with invalid header."""
    return "filename,file_size,file_attributes\n1,2,3"


@pytest.fixture
def via_frame_number_in_file_attribute_not_integer():
    """Return the content of a VIA tracks .csv file with invalid frame
    number defined as file_attribute.
    """
    return (
        via_tracks_csv_file_valid_header
        + "04.09.2023-04-Right_RE_test_frame_A.png,"
        "26542080,"
        '"{""clip"":123, ""frame"":""FOO""}",'  # frame number is a string
        "1,"
        "0,"
        '"{""name"":""rect"",""x"":526.236,""y"":393.281,""width"":46,""height"":38}",'
        '"{""track"":""71""}"'
    )


@pytest.fixture
def via_frame_number_in_filename_wrong_pattern():
    """Return the content of a VIA tracks .csv file with invalid frame
    number defined in the frame's filename.
    """
    return (
        via_tracks_csv_file_valid_header
        + "04.09.2023-04-Right_RE_test_frame_1.png,"  # frame not zero-padded
        "26542080,"
        '"{""clip"":123}",'
        "1,"
        "0,"
        '"{""name"":""rect"",""x"":526.236,""y"":393.281,""width"":46,""height"":38}",'
        '"{""track"":""71""}"'
    )


@pytest.fixture
def via_more_frame_numbers_than_filenames():
    """Return the content of a VIA tracks .csv file with more
    frame numbers than filenames.
    """
    return (
        via_tracks_csv_file_valid_header + "04.09.2023-04-Right_RE_test.png,"
        "26542080,"
        '"{""clip"":123, ""frame"":24}",'
        "1,"
        "0,"
        '"{""name"":""rect"",""x"":526.236,""y"":393.281,""width"":46,""height"":38}",'
        '"{""track"":""71""}"'
        "\n"
        "04.09.2023-04-Right_RE_test.png,"  # same filename as previous row
        "26542080,"
        '"{""clip"":123, ""frame"":25}",'  # different frame number
        "1,"
        "0,"
        '"{""name"":""rect"",""x"":526.236,""y"":393.281,""width"":46,""height"":38}",'
        '"{""track"":""71""}"'
    )


@pytest.fixture
def via_less_frame_numbers_than_filenames():
    """Return the content of a VIA tracks .csv file with with less
    frame numbers than filenames.
    """
    return (
        via_tracks_csv_file_valid_header + "04.09.2023-04-Right_RE_test_A.png,"
        "26542080,"
        '"{""clip"":123, ""frame"":24}",'
        "1,"
        "0,"
        '"{""name"":""rect"",""x"":526.236,""y"":393.281,""width"":46,""height"":38}",'
        '"{""track"":""71""}"'
        "\n"
        "04.09.2023-04-Right_RE_test_B.png,"  # different filename
        "26542080,"
        '"{""clip"":123, ""frame"":24}",'  # same frame as previous row
        "1,"
        "0,"
        '"{""name"":""rect"",""x"":526.236,""y"":393.281,""width"":46,""height"":38}",'
        '"{""track"":""71""}"'
    )


@pytest.fixture
def via_region_shape_attribute_not_rect():
    """Return the content of a VIA tracks .csv file with invalid shape in
    region_shape_attributes.
    """
    return (
        via_tracks_csv_file_valid_header
        + "04.09.2023-04-Right_RE_test_frame_01.png,"
        "26542080,"
        '"{""clip"":123}",'
        "1,"
        "0,"
        '"{""name"":""circle"",""cx"":1049,""cy"":1006,""r"":125}",'
        '"{""track"":""71""}"'
    )  # annotation of circular shape


@pytest.fixture
def via_region_shape_attribute_missing_x():
    """Return the content of a VIA tracks .csv file with missing `x` key in
    region_shape_attributes.
    """
    return (
        via_tracks_csv_file_valid_header
        + "04.09.2023-04-Right_RE_test_frame_01.png,"
        "26542080,"
        '"{""clip"":123}",'
        "1,"
        "0,"
        '"{""name"":""rect"",""y"":393.281,""width"":46,""height"":38}",'
        '"{""track"":""71""}"'
    )  # region_shape_attributes is missing ""x"" key


@pytest.fixture
def via_region_attribute_missing_track():
    """Return the content of a VIA tracks .csv file with missing track
    attribute in region_attributes.
    """
    return (
        via_tracks_csv_file_valid_header
        + "04.09.2023-04-Right_RE_test_frame_01.png,"
        "26542080,"
        '"{""clip"":123}",'
        "1,"
        "0,"
        '"{""name"":""rect"",""x"":526.236,""y"":393.281,""width"":46,""height"":38}",'
        '"{""foo"":""71""}"'  # missing ""track""
    )


@pytest.fixture
def via_track_id_not_castable_as_int():
    """Return the content of a VIA tracks .csv file with a track ID
    attribute not castable as an integer.
    """
    return (
        via_tracks_csv_file_valid_header
        + "04.09.2023-04-Right_RE_test_frame_01.png,"
        "26542080,"
        '"{""clip"":123}",'
        "1,"
        "0,"
        '"{""name"":""rect"",""x"":526.236,""y"":393.281,""width"":46,""height"":38}",'
        '"{""track"":""FOO""}"'  # ""track"" not castable as int
    )


@pytest.fixture
def via_track_ids_not_unique_per_frame():
    """Return the content of a VIA tracks .csv file with a track ID
    that appears twice in the same frame.
    """
    return (
        via_tracks_csv_file_valid_header
        + "04.09.2023-04-Right_RE_test_frame_01.png,"
        "26542080,"
        '"{""clip"":123}",'
        "1,"
        "0,"
        '"{""name"":""rect"",""x"":526.236,""y"":393.281,""width"":46,""height"":38}",'
        '"{""track"":""71""}"'
        "\n"
        "04.09.2023-04-Right_RE_test_frame_01.png,"
        "26542080,"
        '"{""clip"":123}",'
        "1,"
        "0,"
        '"{""name"":""rect"",""x"":2567.627,""y"":466.888,""width"":40,""height"":37}",'
        '"{""track"":""71""}"'  # same track ID as the previous row
    )


# ---------------- Anipose file fixtures ----------------------------
@pytest.fixture
def anipose_csv_file():
    """Return the path to an Anipose .csv file."""
    return pytest.DATA_PATHS.get(
        "anipose_mouse-paw_anipose-paper.triangulation.csv"
    )


# ---------------- netCDF file fixtures ----------------------------
@pytest.fixture(scope="session")
def invalid_netcdf_file_missing_confidence(tmp_path_factory):
    """Create an invalid 'poses' netCDF file missing the
    'confidence' variable.
    """
    valid_file = pytest.DATA_PATHS.get("MOVE_two-mice_octagon.analysis.nc")
    ds = xr.open_dataset(valid_file)
    del ds["confidence"]

    temp_dir = tmp_path_factory.mktemp("invalid_netcdf")
    invalid_path = temp_dir / "invalid_file_missing_confidence.nc"
    ds.to_netcdf(invalid_path)
    yield str(invalid_path)


@pytest.fixture(scope="session")
def unopenable_netcdf_file(tmp_path_factory):
    """Create a fake .nc file that is just text, causing
    xr.open_dataset to fail.
    """
    temp_dir = tmp_path_factory.mktemp("invalid_netcdf")
    invalid_path = temp_dir / "unopenable_file.nc"
    with open(invalid_path, "w") as f:
        f.write("This is not a real netCDF file")
    yield str(invalid_path)


@pytest.fixture(scope="session")
def invalid_dstype_netcdf_file(tmp_path_factory):
    """Create a valid netCDF file but with an invalid 'ds_type' attribute."""
    valid_file = pytest.DATA_PATHS.get("MOVE_two-mice_octagon.analysis.nc")
    ds = xr.open_dataset(valid_file)

    ds.attrs["ds_type"] = "not_a_valid_type"

    temp_dir = tmp_path_factory.mktemp("invalid_netcdf")
    invalid_path = temp_dir / "invalid_dstype_file.nc"
    ds.to_netcdf(invalid_path)

    yield str(invalid_path)


# ---------------- COCO JSON file fixtures ----------------------------


# -------------- Motion-BIDS file fixtures ----------------------------

_MOTION_BIDS_PREFIX = "sub-01_task-tracking_tracksys-pose"
_CHANNELS_HEADER = ["name", "component", "type", "tracked_point", "units"]


def _write_motion_bids_files(
    tmp_path,
    *,
    prefix=_MOTION_BIDS_PREFIX,
    motion_data=None,
    channels_data=None,
    metadata=None,
):
    """Write a set of Motion-BIDS files and return the motion.tsv path.

    This is a helper function for creating Motion-BIDS fixture files.
    """
    import json

    parent = tmp_path / "motion_bids"
    parent.mkdir(exist_ok=True)

    motion_path = parent / f"{prefix}_motion.tsv"
    channels_path = parent / f"{prefix}_channels.tsv"
    json_path = parent / f"{prefix}_motion.json"

    if motion_data is not None:
        with open(motion_path, "w") as f:
            for row in motion_data:
                f.write("\t".join(str(v) for v in row) + "\n")

    if channels_data is not None:
        with open(channels_path, "w") as f:
            f.write("\t".join(channels_data[0]) + "\n")
            for row in channels_data[1:]:
                f.write("\t".join(str(v) for v in row) + "\n")

    if metadata is not None:
        with open(json_path, "w") as f:
            json.dump(metadata, f)

    return motion_path


@pytest.fixture
def valid_motion_bids_2d(tmp_path):
    """Return paths for a valid 2D Motion-BIDS dataset.

    3 frames, 1 individual, 2 keypoints (nose, tail), 2D (x, y).
    """
    motion_data = [
        [1.0, 2.0, 5.0, 6.0],
        [1.1, 2.1, 5.1, 6.1],
        [1.2, 2.2, 5.2, 6.2],
    ]
    channels_data = [
        _CHANNELS_HEADER,
        ["nose_x", "x", "POS", "nose", "px"],
        ["nose_y", "y", "POS", "nose", "px"],
        ["tail_x", "x", "POS", "tail", "px"],
        ["tail_y", "y", "POS", "tail", "px"],
    ]
    metadata = {"SamplingFrequency": 30}
    return _write_motion_bids_files(
        tmp_path,
        motion_data=motion_data,
        channels_data=channels_data,
        metadata=metadata,
    )


@pytest.fixture
def valid_motion_bids_3d(tmp_path):
    """Return paths for a valid 3D Motion-BIDS dataset.

    2 frames, 1 individual, 2 keypoints (nose, tail), 3D (x, y, z).
    """
    motion_data = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
    ]
    channels_data = [
        _CHANNELS_HEADER,
        ["nose_x", "x", "POS", "nose", "m"],
        ["nose_y", "y", "POS", "nose", "m"],
        ["nose_z", "z", "POS", "nose", "m"],
        ["tail_x", "x", "POS", "tail", "m"],
        ["tail_y", "y", "POS", "tail", "m"],
        ["tail_z", "z", "POS", "tail", "m"],
    ]
    metadata = {"SamplingFrequency": 60}
    return _write_motion_bids_files(
        tmp_path,
        motion_data=motion_data,
        channels_data=channels_data,
        metadata=metadata,
    )


@pytest.fixture
def valid_motion_bids_multi_individual(tmp_path):
    """Return paths for a valid Motion-BIDS dataset with 2 individuals.

    2 frames, 2 individuals (Alice, Bob), 1 keypoint (nose), 2D (x, y).
    """
    motion_data = [
        [10.0, 20.0, 30.0, 40.0],
        [10.1, 20.1, 30.1, 40.1],
    ]
    channels_data = [
        _CHANNELS_HEADER + ["individual"],
        ["Alice_nose_x", "x", "POS", "nose", "px", "Alice"],
        ["Alice_nose_y", "y", "POS", "nose", "px", "Alice"],
        ["Bob_nose_x", "x", "POS", "nose", "px", "Bob"],
        ["Bob_nose_y", "y", "POS", "nose", "px", "Bob"],
    ]
    metadata = {"SamplingFrequency": 30}
    return _write_motion_bids_files(
        tmp_path,
        motion_data=motion_data,
        channels_data=channels_data,
        metadata=metadata,
    )


@pytest.fixture
def motion_bids_missing_channels(tmp_path):
    """Return path for Motion-BIDS files missing _channels.tsv."""
    motion_data = [[1.0, 2.0]]
    metadata = {"SamplingFrequency": 30}
    return _write_motion_bids_files(
        tmp_path,
        motion_data=motion_data,
        channels_data=None,
        metadata=metadata,
    )


@pytest.fixture
def motion_bids_missing_json(tmp_path):
    """Return path for Motion-BIDS files missing _motion.json."""
    motion_data = [[1.0, 2.0]]
    channels_data = [
        _CHANNELS_HEADER,
        ["nose_x", "x", "POS", "nose", "px"],
        ["nose_y", "y", "POS", "nose", "px"],
    ]
    return _write_motion_bids_files(
        tmp_path,
        motion_data=motion_data,
        channels_data=channels_data,
        metadata=None,
    )


@pytest.fixture
def motion_bids_missing_sampling_freq(tmp_path):
    """Return path for Motion-BIDS files missing SamplingFrequency."""
    motion_data = [[1.0, 2.0]]
    channels_data = [
        _CHANNELS_HEADER,
        ["nose_x", "x", "POS", "nose", "px"],
        ["nose_y", "y", "POS", "nose", "px"],
    ]
    metadata = {"TaskName": "walking"}
    return _write_motion_bids_files(
        tmp_path,
        motion_data=motion_data,
        channels_data=channels_data,
        metadata=metadata,
    )


@pytest.fixture
def motion_bids_missing_channels_columns(tmp_path):
    """Return path for Motion-BIDS files where _channels.tsv is
    missing required columns.
    """
    motion_data = [[1.0, 2.0]]
    channels_data = [
        ["name", "component"],
        ["nose_x", "x"],
        ["nose_y", "y"],
    ]
    metadata = {"SamplingFrequency": 30}
    return _write_motion_bids_files(
        tmp_path,
        motion_data=motion_data,
        channels_data=channels_data,
        metadata=metadata,
    )


@pytest.fixture
def motion_bids_no_pos_channels(tmp_path):
    """Return path for Motion-BIDS files where _channels.tsv has
    no POS-type channels.
    """
    motion_data = [[1.0, 2.0]]
    channels_data = [
        _CHANNELS_HEADER,
        ["accel_x", "x", "ACCEL", "wrist", "m/s^2"],
        ["accel_y", "y", "ACCEL", "wrist", "m/s^2"],
    ]
    metadata = {"SamplingFrequency": 30}
    return _write_motion_bids_files(
        tmp_path,
        motion_data=motion_data,
        channels_data=channels_data,
        metadata=metadata,
    )


@pytest.fixture
def motion_bids_with_header(tmp_path):
    """Return path for a Motion-BIDS _motion.tsv with a non-numeric
    header row (invalid).
    """
    motion_path = _write_motion_bids_files(
        tmp_path,
        motion_data=None,
        channels_data=[
            _CHANNELS_HEADER,
            ["nose_x", "x", "POS", "nose", "px"],
            ["nose_y", "y", "POS", "nose", "px"],
        ],
        metadata={"SamplingFrequency": 30},
    )
    # Overwrite motion tsv with a header row (invalid)
    with open(motion_path, "w") as f:
        f.write("col_a\tcol_b\n1.0\t2.0\n")
    return motion_path


@pytest.fixture
def motion_bids_empty_tsv(tmp_path):
    """Return path for an empty Motion-BIDS _motion.tsv file."""
    motion_path = _write_motion_bids_files(
        tmp_path,
        motion_data=None,
        channels_data=[
            _CHANNELS_HEADER,
            ["nose_x", "x", "POS", "nose", "px"],
        ],
        metadata={"SamplingFrequency": 30},
    )
    # Overwrite motion tsv as empty (invalid)
    motion_path.write_text("")
    return motion_path


@pytest.fixture
def motion_bids_invalid_json(tmp_path):
    """Return path for Motion-BIDS files with an invalid JSON file."""
    motion_path = _write_motion_bids_files(
        tmp_path,
        motion_data=[[1.0, 2.0]],
        channels_data=[
            _CHANNELS_HEADER,
            ["nose_x", "x", "POS", "nose", "px"],
            ["nose_y", "y", "POS", "nose", "px"],
        ],
        metadata={"SamplingFrequency": 30},
    )
    # Overwrite JSON with invalid content
    json_path = motion_path.parent / (
        motion_path.name.replace("_motion.tsv", "_motion.json")
    )
    with open(json_path, "w") as f:
        f.write("{bad json content")
    return motion_path


@pytest.fixture
def motion_bids_wrong_filename(tmp_path):
    """Return path for a .tsv file that doesn't end with _motion.tsv."""
    parent = tmp_path / "motion_bids_wrong_name"
    parent.mkdir(exist_ok=True)
    file_path = parent / "sub-01_task-tracking_data.tsv"
    with open(file_path, "w") as f:
        f.write("1.0\t2.0\n")
    return file_path


@pytest.fixture
def motion_bids_corrupt_channels(tmp_path):
    """Return path for Motion-BIDS with a corrupt channels TSV file."""
    motion_path = _write_motion_bids_files(
        tmp_path,
        motion_data=[[1.0, 2.0]],
        channels_data=[
            _CHANNELS_HEADER,
            ["nose_x", "x", "POS", "nose", "px"],
            ["nose_y", "y", "POS", "nose", "px"],
        ],
        metadata={"SamplingFrequency": 30},
    )
    # Overwrite channels TSV with binary/corrupt data that pandas can't parse
    channels_path = motion_path.parent / (
        motion_path.name.replace("_motion.tsv", "_channels.tsv")
    )
    with open(channels_path, "wb") as f:
        # Write binary data that will cause pandas to fail
        f.write(b"\x00\x01\x02\xff\xfe\xfd")
    return motion_path


@pytest.fixture
def motion_bids_wrong_component_order(tmp_path):
    """Return path for Motion-BIDS where components are y, x (wrong order)."""
    motion_data = [
        [1.0, 2.0, 5.0, 6.0],
        [1.1, 2.1, 5.1, 6.1],
    ]
    channels_data = [
        _CHANNELS_HEADER,
        ["nose_y", "y", "POS", "nose", "px"],
        ["nose_x", "x", "POS", "nose", "px"],
        ["tail_y", "y", "POS", "tail", "px"],
        ["tail_x", "x", "POS", "tail", "px"],
    ]
    metadata = {"SamplingFrequency": 30}
    return _write_motion_bids_files(
        tmp_path,
        motion_data=motion_data,
        channels_data=channels_data,
        metadata=metadata,
    )


@pytest.fixture
def motion_bids_column_count_mismatch(tmp_path):
    """Return path for Motion-BIDS where motion.tsv has more columns
    than channels.tsv defines.
    """
    motion_data = [
        [1.0, 2.0, 3.0, 4.0, 5.0],  # 5 columns
        [1.1, 2.1, 3.1, 4.1, 5.1],
    ]
    channels_data = [
        _CHANNELS_HEADER,
        ["nose_x", "x", "POS", "nose", "px"],
        ["nose_y", "y", "POS", "nose", "px"],
    ]  # only 2 channels
    metadata = {"SamplingFrequency": 30}
    return _write_motion_bids_files(
        tmp_path,
        motion_data=motion_data,
        channels_data=channels_data,
        metadata=metadata,
    )


@pytest.fixture
def motion_bids_invalid_components(tmp_path):
    """Return path for Motion-BIDS with invalid component values (a, b)."""
    motion_data = [
        [1.0, 2.0],
        [1.1, 2.1],
    ]
    channels_data = [
        _CHANNELS_HEADER,
        ["nose_a", "a", "POS", "nose", "px"],
        ["nose_b", "b", "POS", "nose", "px"],
    ]
    metadata = {"SamplingFrequency": 30}
    return _write_motion_bids_files(
        tmp_path,
        motion_data=motion_data,
        channels_data=channels_data,
        metadata=metadata,
    )
