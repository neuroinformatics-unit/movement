"""Valid and invalid file fixtures.

Note: NWB file fixtures are in tests/fixtures/nwb.py.
"""

import os
from pathlib import Path
from unittest.mock import mock_open, patch

import h5py
import pytest
from sleap_io.io.slp import read_labels, write_labels
from sleap_io.model.labels import LabeledFrame, Labels


# ------------------ Generic file fixtures ----------------------
@pytest.fixture
def unreadable_file(tmp_path):
    """Return a dictionary containing the file path and
    expected permission for an unreadable .h5 file.
    """
    file_path = tmp_path / "unreadable.h5"
    file_mock = mock_open()
    file_mock.return_value.read.side_effect = PermissionError
    with (
        patch("builtins.open", side_effect=file_mock),
        patch.object(Path, "exists", return_value=True),
    ):
        yield {
            "file_path": file_path,
            "expected_permission": "r",
        }


@pytest.fixture
def unwriteable_file(tmp_path):
    """Return a dictionary containing the file path and
    expected permission for an unwriteable .h5 file.
    """
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
        yield {
            "file_path": file_path,
            "expected_permission": "w",
        }


@pytest.fixture
def wrong_extension_file(tmp_path):
    """Return a dictionary containing the file path,
    expected permission, and expected suffix for a file
    with unsupported extension.
    """
    file_path = tmp_path / "wrong_extension.txt"
    with open(file_path, "w") as f:
        f.write("")
    return {
        "file_path": file_path,
        "expected_permission": "r",
        "expected_suffix": ["h5", "csv"],
    }


@pytest.fixture
def nonexistent_file(tmp_path):
    """Return a dictionary containing the file path and
    expected permission for a nonexistent file.
    """
    file_path = tmp_path / "nonexistent.h5"
    return {
        "file_path": file_path,
        "expected_permission": "r",
    }


@pytest.fixture
def no_dataframe_h5_file(tmp_path):
    """Return a dictionary containing the file path and
    expected datasets for a .h5 file that lacks the
    dataset "dataframe".
    """
    file_path = tmp_path / "no_dataframe.h5"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data_in_list", data=[1, 2, 3])
    return {
        "file_path": file_path,
        "expected_datasets": ["dataframe"],
    }


@pytest.fixture
def fake_h5_file(tmp_path):
    """Return a dictionary containing the file path,
    expected permission, and expected datasets for
    a file with .h5 extension that is not in HDF5 format.
    """
    file_path = tmp_path / "fake.h5"
    with open(file_path, "w") as f:
        f.write("")
    return {
        "file_path": file_path,
        "expected_datasets": ["dataframe"],
        "expected_permission": "w",
    }


@pytest.fixture
def invalid_single_individual_csv_file(tmp_path):
    """Return the file path for a fake single-individual .csv file."""
    file_path = tmp_path / "fake_single_individual.csv"
    with open(file_path, "w") as f:
        f.write("scorer,columns\nsome,columns\ncoords,columns\n")
        f.write("1,2")
    return file_path


@pytest.fixture
def invalid_multi_individual_csv_file(tmp_path):
    """Return the file path for a fake multi-individual .csv file."""
    file_path = tmp_path / "fake_multi_individual.csv"
    with open(file_path, "w") as f:
        f.write(
            "scorer,columns\nindividuals,columns\nbodyparts,columns\nsome,columns\n"
        )
        f.write("1,2")
    return file_path


@pytest.fixture
def wrong_extension_new_file(tmp_path):
    """Return the file path for a new file with unsupported extension."""
    return tmp_path / "wrong_extension_new_file.txt"


@pytest.fixture
def directory(tmp_path):
    """Return a dictionary containing the file path and
    expected permission for a directory.
    """
    file_path = tmp_path / "directory"
    file_path.mkdir()
    return {
        "file_path": file_path,
        "expected_permission": "r",
    }


@pytest.fixture
def new_h5_file(tmp_path):
    """Return the file path for a new .h5 file."""
    return tmp_path / "new_file.h5"


@pytest.fixture
def new_csv_file(tmp_path):
    """Return the file path for a new .csv file."""
    return tmp_path / "new_file.csv"


# ---------------- Anipose file fixtures ----------------------------
@pytest.fixture
def missing_keypoint_columns_anipose_csv_file(tmp_path):
    """Return the file path for a single-individual anipose .csv file
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
    """Return the file path for a single-individual anipose .csv file
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
    """Return the file path for a SLEAP .h5 or .slp file."""
    return pytest.DATA_PATHS.get(request.param)


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
def invalid_via_tracks_csv_file(tmp_path, request):
    """Return the file path for an invalid VIA tracks .csv file."""

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
