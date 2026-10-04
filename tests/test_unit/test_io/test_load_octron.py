"""Test OCTRON tracking CSV loading without installing OCTRON."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from movement.io import load_bboxes, load_dataset
from movement.io.load import infer_source_software
from movement.validators.files import ValidOCTRONCSV


@pytest.fixture
def octron_csv(tmp_path):
    """Write the metadata and indexed table emitted by OCTRON's CSV writer.

    The layout follows AnalysisOctron.predict_batch: six metadata lines,
    a blank line, then DataFrame.to_csv with its three-level index.
    Writer reference: OCTRON-GUI revision
    54db1e19ae01b86c6f9fd521ea5c4e5fb7519b79, analysis_octron.py.
    """

    def write(name="bird_track_7.csv", track_id=7, video_name="birds.mp4"):
        path = tmp_path / name
        data = pd.DataFrame(
            {
                "frame_counter": [1, 3],
                "frame_idx": [2, 6],
                "track_id": [track_id, track_id],
                "label": ["bird", "bird"],
                "confidence": [0.9, 0.7],
                "pos_x": [999, 999],
                "pos_y": [999, 999],
                "bbox_x_min": [10, 20],
                "bbox_x_max": [30, 50],
                "bbox_y_min": [40, 50],
                "bbox_y_max": [60, 90],
            }
        ).set_index(["frame_counter", "frame_idx", "track_id"])
        with path.open("w") as stream:
            stream.write(
                f"video_name: {video_name}\nframe_count: 8\n"
                "frame_count_analyzed: 4\nvideo_height: 100\n"
                "video_width: 200\ncreated_at: 2026-09-16 12:00:00\n\n"
            )
            data.to_csv(stream, na_rep="NaN")
        return path

    return write


@pytest.mark.parametrize("fps", [None, 20])
def test_load_octron_coordinates(octron_csv, fps):
    """Preserve original frames, confidences, gaps and bounding box centres."""
    path = octron_csv()
    ds = load_dataset(path, source_software="OCTRON", fps=fps)
    np.testing.assert_allclose(ds.time, np.arange(8) / (fps or 1))
    assert ds.individual.values.tolist() == ["id_7"]
    np.testing.assert_allclose(
        ds.position.isel(time=[2, 6], individual=0), [[20, 50], [35, 70]]
    )
    np.testing.assert_allclose(
        ds.shape.isel(time=[2, 6], individual=0), [[20, 20], [30, 40]]
    )
    np.testing.assert_allclose(
        ds.confidence.isel(time=[2, 6], individual=0), [0.9, 0.7]
    )
    assert ds.position.isel(time=[0, 1, 3, 4, 5, 7]).isnull().all()
    assert ds.shape.isel(time=[0, 1, 3, 4, 5, 7]).isnull().all()
    assert ds.confidence.isel(time=[0, 1, 3, 4, 5, 7]).isnull().all()
    assert ds.attrs["source_software"] == "OCTRON"
    assert ds.attrs["source_file"] == path.as_posix()


def test_octron_auto_detection(octron_csv):
    """Detect OCTRON and dispatch through the unified loading interface."""
    path = octron_csv()
    assert infer_source_software(path) == "OCTRON"
    xr.testing.assert_identical(
        load_dataset(path), load_bboxes.from_octron_file(path)
    )


def test_combine_octron_tracks(octron_csv):
    """Combine distinct IDs even when class labels are identical."""
    first = octron_csv()
    second = octron_csv("bird_track_2.csv", track_id=2)
    ds = load_dataset(first, "OCTRON", additional_files=[second])
    assert ds.individual.values.tolist() == ["id_2", "id_7"]
    assert ds.attrs["source_files"] == [str(first), str(second)]
    np.testing.assert_allclose(ds.confidence.isel(time=2), [0.9, 0.9])


def test_reject_mixed_videos(octron_csv):
    """Reject files from distinct videos instead of silently mixing tracks."""
    first = octron_csv()
    second = octron_csv("other.csv", track_id=2, video_name="other.mp4")
    with pytest.raises(ValueError, match="same video"):
        load_bboxes.from_octron_file(first, additional_files=[second])


def test_reject_duplicate_tracks(octron_csv):
    """Reject overlapping observations across input files."""
    path = octron_csv()
    with pytest.raises(ValueError, match="Duplicate"):
        load_bboxes.from_octron_file(path, additional_files=[path])


@pytest.mark.parametrize(
    "old,new,match",
    [
        ("video_name:", "wrong_name:", "metadata"),
        ("frame_count: 8", "frame_count: 0", "positive"),
        ("bbox_x_min", "wrong_column", "columns"),
        ("1,2,7,", "1,-2,7,", "non-negative"),
        ("1,2,7,", "1,2.5,7,", "non-negative"),
        ("1,2,7,", "1,8,7,", "exceeds"),
        ("10,30,40,60", "40,30,40,60", "extents"),
    ],
)
def test_invalid_octron_data(octron_csv, old, new, match):
    """Give actionable errors for malformed tracking files."""
    path = octron_csv()
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ValueError, match=match):
        ValidOCTRONCSV(path)


def test_empty_octron_table(octron_csv):
    """Reject metadata-only files with no track observations."""
    path = octron_csv()
    path.write_text("\n".join(path.read_text().splitlines()[:8]) + "\n")
    with pytest.raises(ValueError, match="no observations"):
        ValidOCTRONCSV(path)


def test_missing_octron_file(tmp_path):
    """Use the standard file validator for nonexistent inputs."""
    with pytest.raises(FileNotFoundError):
        load_bboxes.from_octron_file(tmp_path / "missing.csv")


@pytest.mark.parametrize(
    "old,new,match",
    [
        ("frame_count: 8", "frame_count: unknown", "frame_count"),
        ("1,2,7,", "1,2,NaN,", "non-negative"),
        ("0.9", "inf", "infinite"),
        ("40,60", "90,60", "extents"),
        ("\n\nframe_counter", "\nnot blank\nframe_counter", "blank line"),
    ],
)
def test_invalid_octron_metadata_and_values(octron_csv, old, new, match):
    """Reject malformed metadata, track IDs and numeric values."""
    path = octron_csv()
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ValueError, match=match):
        ValidOCTRONCSV(path)


def test_duplicate_octron_rows(octron_csv):
    """Reject repeated observations within a tracking CSV."""
    path = octron_csv()
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines + [lines[-1]]) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        ValidOCTRONCSV(path)


def test_unsorted_octron_rows_and_nan(octron_csv):
    """Unsorted input and absent confidence retain original frame alignment."""
    path = octron_csv()
    lines = path.read_text().splitlines()
    lines[-2:] = reversed(lines[-2:])
    path.write_text("\n".join(lines).replace("0.9", "NaN") + "\n")
    ds = load_dataset(path, "OCTRON")
    assert np.isnan(ds.confidence.sel(time=2, individual="id_7"))
    np.testing.assert_allclose(
        ds.position.sel(time=2, individual="id_7"), [20, 50]
    )
