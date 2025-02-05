import numpy as np
import pytest
import xarray as xr
from matplotlib import pyplot as plt

from movement.plot import trajectory


@pytest.fixture
def sample_data():
    """Sample data for plot testing.

    Data has three keypoints (left, centre, right) for one
    individual that moves in a straight line along the y-axis with a
    constant x-coordinate.

    """
    time_steps = 4
    individuals = ["individual_0"]
    keypoints = ["left", "centre", "right"]
    space = ["x", "y"]
    positions = {
        "left": {"x": -1, "y": np.arange(time_steps)},
        "centre": {"x": 0, "y": np.arange(time_steps)},
        "right": {"x": 1, "y": np.arange(time_steps)},
    }

    time = np.arange(time_steps)
    position_data = np.zeros(
        (time_steps, len(space), len(keypoints), len(individuals))
    )

    # Create x and y coordinates arrays
    x_coords = np.array([positions[key]["x"] for key in keypoints])
    y_coords = np.array([positions[key]["y"] for key in keypoints])

    for i, _ in enumerate(keypoints):
        position_data[:, 0, i, 0] = x_coords[i]  # x-coordinates
        position_data[:, 1, i, 0] = y_coords[i]  # y-coordinates

    confidence_data = np.full(
        (time_steps, len(keypoints), len(individuals)), 0.90
    )

    ds = xr.Dataset(
        {
            "position": (
                ["time", "space", "keypoints", "individuals"],
                position_data,
            ),
            "confidence": (
                ["time", "keypoints", "individuals"],
                confidence_data,
            ),
        },
        coords={
            "time": time,
            "space": space,
            "keypoints": keypoints,
            "individuals": individuals,
        },
    )
    return ds


@pytest.mark.parametrize(
    ["image", "selection"],
    [
        pytest.param(
            True, {"keypoints": ["left", "right"]}, id="left-right + image"
        ),
        pytest.param(False, {"keypoints": "centre"}, id="centre"),
        pytest.param(False, None, id="no keypoints"),
    ],
)
def test_trajectory(sample_data, image, selection, tmp_path):
    """Test trajectory plot."""
    plt.switch_backend("Agg")  # to avoid pop-up window
    da = sample_data.position
    if image:
        image_path = tmp_path / "image.png"
        fig, ax = plt.subplots()
        ax.imshow(np.zeros((10, 10)))
        fig.savefig(image_path)
        kwargs = {"image_path": image_path}
    else:
        kwargs = {"image_path": None}

    fig, ax = trajectory(da, selection=selection, **kwargs)

    expected_data = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float)
    ax_data = ax.collections[0].get_offsets().data
    np.testing.assert_array_almost_equal(ax_data, expected_data)
