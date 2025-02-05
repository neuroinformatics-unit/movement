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
    ["image"],
    [
        pytest.param(True, id="with_image"),
        pytest.param(False, id="without_image"),
    ],
)
def test_trajectory(sample_data, image, tmp_path):
    """Test midpoint between left and right keypoints."""
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

    _, ax_no_keypoints = trajectory(da, **kwargs)
    _, ax_centre = trajectory(da, selection={"keypoints": "centre"}, **kwargs)
    _, ax_left_right = trajectory(
        da, selection={"keypoints": ["left", "right"]}, **kwargs
    )
    _, ax_centre_right = trajectory(
        da, selection={"keypoints": ["centre", "right"]}, **kwargs
    )  # + 0.5 shift on x-axis in this case

    expected_data = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float)

    # Retrieve data points from figures
    no_keypoints_data = ax_no_keypoints.collections[0].get_offsets().data
    centre_data = ax_centre.collections[0].get_offsets().data
    left_right_data = ax_left_right.collections[0].get_offsets().data
    ax_centre_right_data = ax_centre_right.collections[0].get_offsets().data

    np.testing.assert_array_almost_equal(no_keypoints_data, expected_data)
    np.testing.assert_array_almost_equal(centre_data, expected_data)
    np.testing.assert_array_almost_equal(left_right_data, expected_data)
    np.testing.assert_array_almost_equal(
        ax_centre_right_data, expected_data + [0.5, 0]
    )  # + 0.5 shift on x-axis
