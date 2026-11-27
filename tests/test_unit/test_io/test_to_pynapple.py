import subprocess
import sys

import numpy as np
import pynapple as nap
import pytest

from movement import kinematics
from movement.io import to_pynapple_tsdframe


def assert_tsdframe_matches(tsdframe, da):
    """Assert that ``tsdframe`` matches ``da`` column by column.

    Each column must hold the data that its metadata labels select from
    ``da``, at the same timestamps. If any ``keypoint`` or ``space`` labels
    vary across columns, each column must be named after them, joined by ``_``.
    Each metadata field must also be accessible as an attribute.
    """
    np.testing.assert_allclose(tsdframe.times(), da.time.values)
    metadata = tsdframe.metadata
    for field in metadata.columns:
        assert list(getattr(tsdframe, field)) == list(metadata[field])
    for j, column in enumerate(tsdframe.columns):
        labels = metadata.iloc[j].to_dict() if not metadata.empty else {}
        selection = {k: v for k, v in labels.items() if k in da.dims}
        for field in labels.keys() - selection.keys():
            assert labels[field] == str(da[field].item())
        if selection:
            assert column == "_".join(selection.values())
        np.testing.assert_array_equal(
            tsdframe.values[:, j], da.sel(selection).values
        )


@pytest.mark.parametrize(
    "dataset, variable, expected_metadata_fields",
    [
        pytest.param(
            "valid_poses_dataset",
            "position",
            ["keypoint", "space", "individual"],
            id="poses position",
        ),
        pytest.param(
            "valid_poses_dataset",
            "confidence",
            ["keypoint", "individual"],
            id="poses confidence",
        ),
        pytest.param(
            "valid_poses_dataset_with_nan",
            "position",
            ["keypoint", "space", "individual"],
            id="poses position with NaNs",
        ),
        pytest.param(
            "valid_poses_dataset_3d",
            "position",
            ["keypoint", "space", "individual"],
            id="poses position 3D",
        ),
        pytest.param(
            "valid_bboxes_dataset",
            "position",
            ["space", "individual"],
            id="bboxes position",
        ),
        pytest.param(
            "valid_bboxes_dataset",
            "shape",
            ["space", "individual"],
            id="bboxes shape",
        ),
        pytest.param(
            "valid_bboxes_dataset_with_nan",
            "position",
            ["space", "individual"],
            id="bboxes position with NaNs",
        ),
    ],
)
def test_to_pynapple_tsdframe(
    dataset, variable, expected_metadata_fields, request
):
    """Test that one individual's DataArray becomes a TsdFrame of shape
    ``(n_time, n_columns)``, with column names derived from the
    ``keypoint`` and/or ``space`` coordinate labels (e.g. ``centroid_x``,
    ``centroid`` or ``x``), and the labels stored as metadata.
    """
    ds = request.getfixturevalue(dataset)
    da = ds[variable].sel(individual=ds.individual.values[0])
    tsdframe = to_pynapple_tsdframe(da)
    assert isinstance(tsdframe, nap.TsdFrame)
    n_columns = np.prod([da.sizes[d] for d in da.dims if d != "time"])
    assert tsdframe.shape == (da.sizes["time"], n_columns)
    assert list(tsdframe.metadata.columns) == expected_metadata_fields
    assert_tsdframe_matches(tsdframe, da)


@pytest.mark.parametrize(
    "compute, expected_columns, expected_metadata_fields",
    [
        pytest.param(
            kinematics.compute_velocity,
            [
                "centroid_x",
                "centroid_y",
                "left_x",
                "left_y",
                "right_x",
                "right_y",
            ],
            ["keypoint", "space", "individual"],
            id="velocity (keypoint, space)",
        ),
        pytest.param(
            kinematics.compute_speed,
            ["centroid", "left", "right"],
            ["keypoint", "individual"],
            id="speed (keypoint)",
        ),
        pytest.param(
            lambda position: kinematics.compute_head_direction_vector(
                position, "left", "right"
            ),
            ["x", "y"],
            ["space", "individual"],
            id="head direction vector (space)",
        ),
        pytest.param(
            lambda position: kinematics.compute_forward_vector_angle(
                position, "left", "right"
            ),
            ["forward_vector_angle"],
            ["individual"],
            id="forward vector angle (time-only)",
        ),
    ],
)
def test_to_pynapple_tsdframe_derived_variable(
    valid_poses_dataset, compute, expected_columns, expected_metadata_fields
):
    """Test that derived variables can be converted, whichever of the
    ``keypoint`` and ``space`` dimensions they have.
    """
    da = compute(valid_poses_dataset.position.sel(individual="id_0"))
    tsdframe = to_pynapple_tsdframe(da)
    assert list(tsdframe.columns) == expected_columns
    assert list(tsdframe.metadata.columns) == expected_metadata_fields
    assert_tsdframe_matches(tsdframe, da)


@pytest.mark.parametrize(
    "valid_poses_dataset", ["single_individual_array"], indirect=True
)
def test_to_pynapple_tsdframe_single_individual_dim(valid_poses_dataset):
    """Test that a DataArray with a size-1 ``individual`` dimension gives
    the same result as selecting that individual: the individual is not
    in the column names, but is kept in metadata.
    """
    da = valid_poses_dataset.position
    tsdframe = to_pynapple_tsdframe(da)
    expected = to_pynapple_tsdframe(da.sel(individual="id_0"))
    assert list(tsdframe.columns) == list(expected.columns)
    assert tsdframe.metadata.equals(expected.metadata)
    assert set(tsdframe.metadata["individual"]) == {"id_0"}
    np.testing.assert_array_equal(tsdframe.values, expected.values)


@pytest.mark.parametrize(
    "name, expected_column",
    [
        pytest.param("my_variable", "my_variable", id="named"),
        pytest.param(None, "Unnamed", id="unnamed"),
    ],
)
def test_to_pynapple_tsdframe_time_only(
    valid_bboxes_dataset, name, expected_column
):
    """Test that a time-only DataArray becomes a single-column TsdFrame,
    named after the DataArray or ``Unnamed`` if it has no name.

    Bboxes confidence is used because it is time-only once an individual
    is selected, so no other labels end up in metadata.
    """
    da = valid_bboxes_dataset.confidence.sel(individual="id_0").rename(name)
    tsdframe = to_pynapple_tsdframe(da)
    assert isinstance(tsdframe, nap.TsdFrame)
    assert list(tsdframe.columns) == [expected_column]
    assert tsdframe.metadata.to_dict("list") == {"individual": ["id_0"]}
    assert_tsdframe_matches(tsdframe, da)


@pytest.mark.parametrize(
    "selection, expected_columns, expected_metadata",
    [
        pytest.param(
            {"keypoint": "left"},
            ["x", "y"],
            {
                "space": ["x", "y"],
                "individual": ["id_0", "id_0"],
                "keypoint": ["left", "left"],
            },
            id="single keypoint label",
        ),
        pytest.param(
            {"keypoint": ["left"]},
            ["left_x", "left_y"],
            {
                "keypoint": ["left", "left"],
                "space": ["x", "y"],
                "individual": ["id_0", "id_0"],
            },
            id="size-1 keypoint dim",
        ),
        pytest.param(
            {"keypoint": "left", "space": "x"},
            ["position"],
            {"individual": ["id_0"], "keypoint": ["left"], "space": ["x"]},
            id="time-only with single labels",
        ),
    ],
)
def test_to_pynapple_tsdframe_single_labels(
    valid_poses_dataset, selection, expected_columns, expected_metadata
):
    """Test that canonical dimensions with a single selected label are included
    in metadata, with the same value for every column, after the fields that
    vary across columns.

    If a keypoint is selected by passing a list, e.g.
    ``da.sel(keypoint=["left"])``, the ``keypoint`` dimension is kept with
    size 1, so the keypoint label is also included in the column names.
    """
    da = valid_poses_dataset.position.sel(individual="id_0").sel(selection)
    tsdframe = to_pynapple_tsdframe(da)
    assert list(tsdframe.columns) == expected_columns
    assert tsdframe.metadata.to_dict("list") == expected_metadata
    assert_tsdframe_matches(tsdframe, da)


@pytest.mark.parametrize(
    "dataset, selection, match",
    [
        pytest.param(
            "valid_poses_dataset",
            {"time": 0},
            r"must contain \['time'\] as dimensions",
            id="no time dim",
        ),
        pytest.param(
            "valid_poses_dataset",
            {},
            r"2 individuals.*\.sel\(individual=",
            id="multiple individuals",
        ),
        pytest.param(
            "valid_poses_dataset_multiview",
            {"individual": "id_0"},
            "'view'",
            id="non-canonical dim",
        ),
    ],
)
def test_to_pynapple_tsdframe_invalid_dims(dataset, selection, match, request):
    """Test that a DataArray without a ``time`` dimension, with more than
    one individual, or with non-canonical dimensions raises an error.
    """
    da = request.getfixturevalue(dataset).position.sel(selection)
    with pytest.raises(ValueError, match=match):
        to_pynapple_tsdframe(da)


def test_to_pynapple_tsdframe_without_pynapple(
    valid_poses_dataset, monkeypatch
):
    """Test that an informative ImportError is raised if ``pynapple`` is
    not installed.
    """
    monkeypatch.setitem(sys.modules, "pynapple", None)
    with pytest.raises(ImportError, match=r"movement\[pynapple\]"):
        to_pynapple_tsdframe(valid_poses_dataset.position)


def test_movement_io_does_not_import_pynapple():
    """Test that importing ``movement.io`` does not import ``pynapple``,
    which is only needed (and imported) when converting.
    """
    code = "import sys, movement.io; assert 'pynapple' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
