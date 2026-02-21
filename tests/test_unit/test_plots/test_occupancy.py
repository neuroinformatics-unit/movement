from collections.abc import Hashable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from numpy.typing import ArrayLike

from movement.plots import plot_occupancy


def antidiagonal_matrix(diag_values: ArrayLike) -> np.ndarray:
    """Create an antidiagonal matrix.

    An antidiagonal matrix has the ``diag_values`` along the reverse (TR to BL)
    diagonal, with ``diag_values[0]`` appearing in the top-left position.

    Antidiagonal matrices are square.
    """
    return np.fliplr(np.diag(diag_values))


@pytest.fixture()
def occupancy_data() -> xr.DataArray:
    """DataArray of 3 keypoints and 4 individuals.

    Individuals 0 through 2 (inclusive) are identical.
    Individual 4 is a translation by (1,0) of the other individuals.

    The keypoints are left, right, centre.
    Right = left + (1., 1.)
    Centre = mean(left, right)

    The extent of the data is [0,6] x [0,5]. Using bins=list(range(7)) or
    list(range(6)) will force unit-spaced bins.
    """
    time_space = np.array(
        [[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]]
    )

    time_space_keypoints = np.repeat(
        time_space[:, :, np.newaxis], repeats=3, axis=2
    )
    # Set right = left + (1., 1.)
    time_space_keypoints[:, :, 1] += (1.0, 1.0)
    # Set centre = mean(left, right)
    time_space_keypoints[:, :, 2] = np.mean(
        time_space_keypoints[:, :, :2], axis=2
    )

    # individuals 0-2 (inclusive) are copies
    data_vals = np.repeat(
        time_space_keypoints[:, :, :, np.newaxis], repeats=4, axis=3
    )
    # individual 3 is (1., 0) offset from the others
    for keypoint_index in range(data_vals.shape[2]):
        data_vals[:, :, keypoint_index, 3] += (1.0, 0.0)
    return xr.DataArray(
        data=data_vals,
        dims=["time", "space", "keypoint", "individual"],
        coords={
            "space": ["x", "y"],
            "keypoint": ["left", "right", "centre"],
            "individual": [0, 1, 2, 3],
        },
    )


@pytest.fixture
def occupancy_data_with_nans(occupancy_data: xr.DataArray) -> xr.DataArray:
    """Occupancy data with deliberate NaN values.

    The occupancy_data fixture is modified so that:

    - Individual 0 has an NaN value at its left keypoint, "x" coord, 0th index.
    - Individual 1 has an NaN coordinate at its centre keypoint, 0th index.
    - Individual 2 is entirely NaN values down its right keypoint.
    """
    occupancy_data_nans = occupancy_data.copy(deep=True)

    occupancy_data_nans.loc[0, "x", "left", 0] = float("nan")
    occupancy_data_nans.loc[0, :, "centre", 1] = float("nan")
    occupancy_data_nans.loc[:, :, "right", 2] = float("nan")

    return occupancy_data_nans


@pytest.mark.parametrize(
    [
        "data",
        "kwargs_to_pass",
        "expected_output",
        "select_before_passing_to_plot",
    ],
    [
        pytest.param(
            "occupancy_data",
            {"individuals": 0, "bins": [list(range(6)), list(range(6))]},
            antidiagonal_matrix([1] * 5),
            {},
            id="Keypoints: default centroid",
        ),
        pytest.param(
            "occupancy_data",
            {
                "keypoints": ["left", "right"],
                "individuals": 0,
                "bins": [list(range(6)), list(range(6))],
            },
            antidiagonal_matrix([1] * 5),
            {},
            id="Keypoints: selection centroid",
        ),
        pytest.param(
            "occupancy_data",
            {
                "individuals": [0, 1, 2],
                "bins": [list(range(6)), list(range(6))],
                # data will have no keypoints dimension,
                # so the argument below should be ignored
                "keypoints": ["left", "right"],
            },
            3 * antidiagonal_matrix([1] * 5),
            {"keypoint": "centre"},
            id="Keypoints: Handle not a dimension",
        ),
        pytest.param(
            "occupancy_data",
            {
                "keypoints": "centre",
                "bins": [list(range(7)), list(range(6))],
            },
            3
            * np.array(
                [
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            )
            + np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                ]
            ),
            {},
            id="Individuals: default aggregate",
        ),
        pytest.param(
            "occupancy_data",
            {
                "individuals": [0, 1, 2],
                "bins": [list(range(6)), list(range(6))],
            },
            3 * antidiagonal_matrix([1] * 5),
            {},
            id="Individuals: selection aggregate",
        ),
        pytest.param(
            "occupancy_data",
            {
                "keypoints": ["left", "right"],
                "bins": [list(range(6)), list(range(6))],
                # data will have no individuals dimension,
                # so the argument below should be ignored
                "individuals": [0, 2],
            },
            antidiagonal_matrix([1] * 5),
            {"individual": 0},
            id="Individuals: Handle not a dimension",
        ),
        pytest.param(
            "occupancy_data",
            {
                "keypoints": ["left", "right"],
                "individuals": [0, 2],
                # Also check that ax doesn't complain
                "ax": plt.subplots()[1],
                "bins": [list(range(6)), list(range(6))],
            },
            2 * antidiagonal_matrix([1] * 5),
            {},
            id="Sub-selection: mean THEN aggregate",
        ),
        pytest.param(
            "occupancy_data_with_nans",
            {
                "keypoints": "centre",
                "individuals": 1,
                "bins": [list(range(6)), list(range(6))],
            },
            antidiagonal_matrix([0] + ([1] * 4)),
            {},
            id="NaNs: coord does not contribute",
        ),
        pytest.param(
            "occupancy_data_with_nans",
            {
                "keypoints": ["left", "right"],
                "individuals": 1,
                "bins": [list(range(6)), list(range(6))],
            },
            antidiagonal_matrix([1] * 5),
            {},
            id="NaNs: average of valid keypoints still works",
        ),
        pytest.param(
            "occupancy_data_with_nans",
            {
                "keypoints": "right",
                "individuals": 2,
                "bins": [list(range(6)), list(range(6))],
            },
            np.zeros((5, 5)),
            {},
            id="NaNs: no valid points",
        ),
        pytest.param(
            "occupancy_data_with_nans",
            {
                "individuals": [0, 1, 2],
                "bins": [list(range(6)), list(range(6))],
            },
            3 * antidiagonal_matrix([1] * 5),
            {},
            id="NaNs: aggregate can ignore NaNs",
        ),
    ],
)
def test_keypoints_and_individuals_behaviour(
    data: str | xr.DataArray,
    kwargs_to_pass: dict[str, Any],
    expected_output: np.ndarray,
    select_before_passing_to_plot: dict[Hashable, Sequence[Hashable]],
    request,
) -> None:
    if isinstance(data, str):
        data = request.getfixturevalue(data)
    # Remove dimensions from data, if we want to test how the function
    # handles data without certain dimension labels but which can still be
    # plotted
    if select_before_passing_to_plot:
        data = data.sel(select_before_passing_to_plot)

    fig, _, hist_info = plot_occupancy(data, **kwargs_to_pass)
    # This just helps suppress a warning about open plots
    plt.close(fig)

    assert np.allclose(expected_output, hist_info["h"])
