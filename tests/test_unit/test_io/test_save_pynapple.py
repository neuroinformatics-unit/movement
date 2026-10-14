"""Tests for saving movement datasets in pynapple ``.npz`` format."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from movement.io import load_bboxes, load_poses, save_dataset, save_pynapple


def _poses_ds(multi_individual: bool = True) -> xr.Dataset:
    """Return a 2D poses dataset with time in seconds."""
    n_individuals = 2 if multi_individual else 1
    individual_names = ["Alice", "Bob"][:n_individuals]
    return load_poses.from_numpy(
        position_array=np.arange(48, dtype=float).reshape(4, 2, 3, 2)[
            :, :, :, :n_individuals
        ],
        confidence_array=np.full((4, 3, n_individuals), 0.9),
        individual_names=individual_names,
        keypoint_names=["snout", "centre", "tail"],
        fps=30.0,
    )


def _bboxes_ds() -> xr.Dataset:
    """Return a 2D bounding boxes dataset with time in seconds."""
    return load_bboxes.from_numpy(
        position_array=np.arange(16, dtype=float).reshape(4, 2, 2),
        shape_array=np.full((4, 2, 2), 5.0),
        confidence_array=np.full((4, 2), 0.8),
        individual_names=["Alice", "Bob"],
        fps=30.0,
    )


def _load_npz(file: Path) -> dict:
    """Load the raw npz file, allowing pickled entries."""
    with np.load(file, allow_pickle=True) as npz:
        return {key: npz[key] for key in npz.files}


def test_poses_npz_structure_single_individual(tmp_path):
    """Test npz keys, column names and values for one individual."""
    ds = _poses_ds(multi_individual=False)
    file = tmp_path / "poses.npz"
    save_pynapple.to_pynapple_file(ds, file)

    npz = _load_npz(file)
    assert set(npz) == {"t", "d", "start", "end", "columns", "type"}
    assert npz["type"].tolist() == ["TsdFrame"]
    expected_columns = [
        f"{kp}_{space}"
        for kp in ["snout", "centre", "tail"]
        for space in ["x", "y"]
    ] + ["snout", "centre", "tail"]
    assert npz["columns"].tolist() == expected_columns
    assert npz["d"].shape == (4, len(expected_columns))
    np.testing.assert_allclose(npz["t"], np.arange(4) / 30.0)
    np.testing.assert_allclose(npz["start"], [0.0])
    np.testing.assert_allclose(npz["end"], [3 / 30.0])
    # First column is snout_x: position_array[:, x, snout, 0]
    np.testing.assert_allclose(
        npz["d"][:, 0],
        ds["position"].isel(space=0, keypoint=0, individual=0).values,
    )


def test_poses_npz_columns_multi_individual(tmp_path):
    """Test that multi-individual columns are prefixed per individual."""
    ds = _poses_ds(multi_individual=True)
    file = tmp_path / "poses.npz"
    save_pynapple.to_pynapple_file(ds, file)

    columns = _load_npz(file)["columns"].tolist()
    assert columns[:2] == ["Alice_snout_x", "Alice_snout_y"]
    assert "Alice_snout" in columns
    assert "Bob_snout_x" in columns
    assert "Bob_tail_y" in columns
    assert not any(c.startswith("id_") for c in columns)
    assert len(columns) == 2 * 9


def test_bboxes_npz_columns(tmp_path):
    """Test bbox column names, including per-individual prefixes."""
    ds = _bboxes_ds()
    file = tmp_path / "bboxes.npz"
    save_pynapple.to_pynapple_file(ds, file)

    npz = _load_npz(file)
    assert npz["columns"].tolist() == [
        "Alice_x",
        "Alice_y",
        "Alice_width",
        "Alice_height",
        "Alice_confidence",
        "Bob_x",
        "Bob_y",
        "Bob_width",
        "Bob_height",
        "Bob_confidence",
    ]
    np.testing.assert_allclose(
        npz["d"][:, 4], ds["confidence"].isel(individual=0).values
    )


def test_extra_variable_is_included(tmp_path):
    """Test that extra per-frame variables become columns."""
    ds = _poses_ds(multi_individual=False)
    ds["head_direction"] = xr.DataArray(np.linspace(0, np.pi, 4), dims="time")
    file = tmp_path / "poses.npz"
    save_pynapple.to_pynapple_file(ds, file)

    npz = _load_npz(file)
    assert npz["columns"].tolist()[-1] == "head_direction"
    np.testing.assert_allclose(npz["d"][:, -1], np.linspace(0, np.pi, 4))


def test_rejects_time_in_frames(valid_poses_dataset, tmp_path):
    """Test that datasets loaded without fps are rejected."""
    assert valid_poses_dataset.attrs["time_unit"] == "frames"
    with pytest.raises(ValueError, match="fps"):
        save_pynapple.to_pynapple_file(
            valid_poses_dataset, tmp_path / "poses.npz"
        )


def test_save_dataset_dispatch(tmp_path):
    """Test the writer is reachable via save_dataset."""
    ds = _poses_ds()
    file = tmp_path / "poses.npz"
    save_dataset(ds, file, target_software="pynapple")
    assert file.exists()
    assert npz_type(file) == "TsdFrame"


def npz_type(file: Path) -> str:
    """Return the pynapple object type declared in the npz file."""
    return str(_load_npz(file)["type"][0])


def test_rejects_wrong_suffix(tmp_path):
    """Test that a non-.npz suffix is rejected by the writer."""
    ds = _poses_ds()
    with pytest.raises(ValueError, match="npz"):
        save_pynapple.to_pynapple_file(ds, tmp_path / "poses.h5")


def test_rejects_non_movement_dataset(tmp_path):
    """Test that a dataset without ds_type attribute is rejected."""
    ds = xr.Dataset({"foo": ("time", np.arange(3.0))})
    with pytest.raises(ValueError, match="ds_type"):
        save_pynapple.to_pynapple_file(ds, tmp_path / "foo.npz")


def test_pynapple_roundtrip(tmp_path):
    """Test the saved file loads as a pynapple TsdFrame (skipped if
    pynapple is not installed).
    """
    nap = pytest.importorskip("pynapple")

    ds = _poses_ds(multi_individual=False)
    file = tmp_path / "poses.npz"
    save_pynapple.to_pynapple_file(ds, file)

    tsdframe = nap.load_file(file)
    assert isinstance(tsdframe, nap.TsdFrame)
    assert list(tsdframe.columns) == [
        f"{kp}_{space}"
        for kp in ["snout", "centre", "tail"]
        for space in ["x", "y"]
    ] + ["snout", "centre", "tail"]
    np.testing.assert_allclose(
        tsdframe.values, _load_npz(file)["d"], rtol=1e-6
    )
    np.testing.assert_allclose(tsdframe.index.values, np.arange(4) / 30.0)
