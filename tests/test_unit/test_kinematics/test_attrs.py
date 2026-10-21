"""Regression tests for metadata across the public kinematics API."""

import numpy as np
import pytest
import xarray as xr

from movement import kinematics

# None means that units must be absent; "spatial" means retain input units.
KINEMATICS_CASES = [
    ("compute_time_derivative", {"order": 1}, None),
    ("compute_time_derivative", {"order": 2}, None),
    ("compute_forward_displacement", {}, "spatial"),
    ("compute_backward_displacement", {}, "spatial"),
    ("compute_velocity", {}, None),
    ("compute_acceleration", {}, None),
    ("compute_speed", {}, None),
    ("compute_kinetic_energy", {}, None),
    ("compute_kinetic_energy", {"decompose": True}, None),
    (
        "compute_kinetic_energy",
        {"keypoints": ["left", "right"], "masses": {"left": 2.0}},
        None,
    ),
    (
        "compute_forward_vector",
        {"left_keypoint": "left", "right_keypoint": "right"},
        None,
    ),
    (
        "compute_head_direction_vector",
        {"left_keypoint": "left", "right_keypoint": "right"},
        None,
    ),
    (
        "compute_forward_vector_angle",
        {"left_keypoint": "left", "right_keypoint": "right"},
        "radians",
    ),
    (
        "compute_forward_vector_angle",
        {
            "left_keypoint": "left",
            "right_keypoint": "right",
            "in_degrees": True,
        },
        "degrees",
    ),
    ("compute_path_length", {}, "spatial"),
    ("compute_path_length", {"nan_policy": "omit"}, "spatial"),
    ("compute_path_straightness", {}, None),
    ("compute_turning_angle", {}, "radians"),
    ("compute_turning_angle", {"in_degrees": True}, "degrees"),
    ("compute_turning_angle", {"min_step_length": 2.0}, "radians"),
    ("compute_directional_change", {}, None),
    ("compute_directional_change", {"in_degrees": True}, None),
    ("compute_path_sinuosity", {}, "inverse_sqrt"),
    ("compute_path_emax", {}, "spatial"),
    ("compute_path_emax", {"in_spatial_units": False}, None),
    ("compute_path_deviation", {}, "spatial"),
    (
        "compute_pairwise_distances",
        {"dim": "keypoint", "pairs": {"left": "right"}},
        "spatial",
    ),
    (
        "compute_pairwise_distances",
        {"dim": "keypoint", "pairs": "all", "metric": "cityblock"},
        "spatial",
    ),
    (
        "compute_pairwise_distances",
        {"dim": "individual", "pairs": "all", "metric": "cosine"},
        None,
    ),
    (
        "compute_pairwise_distances",
        {"dim": "keypoint", "pairs": "all", "metric": "sqeuclidean"},
        "square",
    ),
    (
        "compute_pairwise_distances",
        {"dim": "keypoint", "pairs": "all", "metric": "chebyshev"},
        "spatial",
    ),
    (
        "compute_pairwise_distances",
        {
            "dim": "keypoint",
            "pairs": "all",
            "metric": "minkowski",
            "p": 3,
        },
        "spatial",
    ),
    (
        "compute_pairwise_distances",
        {
            "dim": "keypoint",
            "pairs": "all",
            "metric": lambda a, b: abs(a - b).sum(),
        },
        None,
    ),
]


@pytest.mark.parametrize("input_units", [None, "mm", "pixels", "custom units"])
@pytest.mark.parametrize("keep_attrs", [True, False])
@pytest.mark.parametrize(
    "function_name, kwargs, output_units", KINEMATICS_CASES
)
def test_kinematics_attrs(
    valid_poses_dataset,
    input_units,
    keep_attrs,
    function_name,
    kwargs,
    output_units,
):
    """Keep metadata and reliable units without mutating the input."""
    data = valid_poses_dataset.position.assign_coords(
        keypoint=["left", "right", "centre"]
    ).copy(deep=True)
    # A curved, nonstationary path exercises the turning-angle metrics.
    data.loc[{"space": "y"}] += data.time**2
    data.attrs = {
        "log": '[{"operation": "load"}]',
        "foo": 1,
        "custom_metadata": {"experiment": "test"},
        "long_name": "Raw position",
    }
    if input_units is not None:
        data.attrs["units"] = input_units
    original = data.copy(deep=True)

    with xr.set_options(keep_attrs=keep_attrs):
        result = getattr(kinematics, function_name)(data, **kwargs)
        assert xr.get_options()["keep_attrs"] is keep_attrs
    outputs = list(result.values()) if isinstance(result, dict) else [result]
    assert outputs
    expected_attrs = original.attrs.copy()
    expected_attrs.pop("units", None)
    long_names = {
        "compute_path_length": "Path Length",
        "compute_path_straightness": "Path Straightness Index",
        "compute_path_sinuosity": "Path Sinuosity",
        "compute_directional_change": "Directional Change",
        "compute_path_emax": "Maximum Expected Displacement",
        "compute_path_deviation": "Path Deviation",
    }
    if function_name in long_names:
        expected_attrs["long_name"] = long_names[function_name]
    if output_units == "spatial":
        expected_units = input_units
    elif output_units in ("inverse_sqrt", "square"):
        exponent = "(-1/2)" if output_units == "inverse_sqrt" else "2"
        expected_units = (
            f"{input_units}^{exponent}"
            if input_units in ("mm", "pixels")
            else None
        )
    else:
        expected_units = output_units
    if expected_units is not None:
        expected_attrs["units"] = expected_units
    for output in outputs:
        assert output.size > 0
        assert output.attrs == expected_attrs
    xr.testing.assert_identical(data, original)


def test_attrs_cases_cover_public_api():
    """Require metadata regression coverage for every public function."""
    assert {name for name, _, _ in KINEMATICS_CASES} == set(kinematics.__all__)


@pytest.mark.parametrize("keep_attrs", [True, False])
@pytest.mark.parametrize(
    "function_name, kwargs, attrs, time_attrs, expected_units",
    [
        (
            "compute_velocity",
            {},
            {"units": "mm"},
            {"units": "seconds"},
            "mm/s",
        ),
        (
            "compute_velocity",
            {},
            {"space_unit": "cm", "time_unit": "frames"},
            {},
            "cm/frame",
        ),
        (
            "compute_acceleration",
            {},
            {"units": "mm"},
            {"units": "s"},
            "mm/s^2",
        ),
        (
            "compute_speed",
            {},
            {"units": "px", "time_unit": "seconds"},
            {},
            "px/s",
        ),
        (
            "compute_time_derivative",
            {"order": 3},
            {"units": "rad"},
            {"units": "s"},
            "rad/s^3",
        ),
        (
            "compute_time_derivative",
            {"order": 1},
            {"units": "degrees", "space_unit": "mm"},
            {"units": "s"},
            "degrees/s",
        ),
        (
            "compute_directional_change",
            {},
            {},
            {"units": "seconds"},
            "radians/s",
        ),
        (
            "compute_directional_change",
            {"in_degrees": True},
            {"time_unit": "frames"},
            {},
            "degrees/frame",
        ),
        ("compute_path_sinuosity", {}, {"space_unit": "mm"}, {}, "mm^(-1/2)"),
        (
            "compute_velocity",
            {},
            {"units": "mm", "time_unit": "seconds"},
            {"units": "s"},
            "mm/s",
        ),
        (
            "compute_velocity",
            {},
            {"units": "mm", "time_unit": "frames"},
            {"units": "s"},
            None,
        ),
        ("compute_velocity", {}, {"units": "mm"}, {"units": "unknown"}, None),
        ("compute_velocity", {}, {"units": "mm"}, {}, None),
        ("compute_velocity", {}, {"units": "mm/s"}, {"units": "s"}, None),
        ("compute_velocity", {}, {"units": "unknown"}, {"units": "s"}, None),
        (
            "compute_velocity",
            {},
            {"units": "mm", "space_unit": "cm"},
            {"units": "s"},
            None,
        ),
        ("compute_velocity", {}, {"units": 1}, {"units": "s"}, None),
        (
            "compute_velocity",
            {},
            {"units": np.array(["mm", "cm"]), "space_unit": "mm"},
            {"units": "s"},
            None,
        ),
        ("compute_velocity", {}, {"units": "mm"}, {"units": 1}, None),
        (
            "compute_directional_change",
            {},
            {"time_unit": "frames"},
            {"units": "s"},
            None,
        ),
        ("compute_path_sinuosity", {}, {"units": "custom units"}, {}, None),
        (
            "compute_path_sinuosity",
            {},
            {"units": "mm", "space_unit": "cm"},
            {},
            None,
        ),
        (
            "compute_kinetic_energy",
            {},
            {"units": "mm", "time_unit": "seconds"},
            {},
            None,
        ),
    ],
)
def test_explicit_units(
    valid_poses_dataset,
    keep_attrs,
    function_name,
    kwargs,
    attrs,
    time_attrs,
    expected_units,
):
    """Derive only supported units, retaining metadata without conversions."""
    data = valid_poses_dataset.position.copy(deep=True)
    data.loc[{"space": "y"}] += data.time**2
    data.attrs = {"log": "[]", "foo": 1, **attrs}
    data.time.attrs = time_attrs.copy()
    original = data.copy(deep=True)
    expected_attrs = data.attrs.copy()
    expected_attrs.pop("units", None)
    if function_name == "compute_directional_change":
        expected_attrs["long_name"] = "Directional Change"
    elif function_name == "compute_path_sinuosity":
        expected_attrs["long_name"] = "Path Sinuosity"
    if expected_units is not None:
        expected_attrs["units"] = expected_units
    with xr.set_options(keep_attrs=keep_attrs):
        result = getattr(kinematics, function_name)(data, **kwargs)
    assert result.attrs == expected_attrs
    # Metadata must not alter the calculated values or the input.
    reference_data = data.copy(deep=True)
    reference_data.attrs = {}
    with xr.set_options(keep_attrs=True):
        reference = getattr(kinematics, function_name)(
            reference_data, **kwargs
        )
    xr.testing.assert_equal(result, reference)
    xr.testing.assert_identical(data, original)


@pytest.mark.parametrize("keep_attrs", [True, False])
@pytest.mark.parametrize(
    "canonical, alias",
    [
        ("euclidean", "euclid"),
        ("euclidean", "EUCLIDEAN"),
        ("euclidean", "e"),
        ("euclidean", "eu"),
        ("cityblock", "cblock"),
        ("cityblock", "cb"),
        ("cityblock", "c"),
        ("chebyshev", "chebychev"),
        ("chebyshev", "cheb"),
        ("chebyshev", "cheby"),
        ("chebyshev", "ch"),
        ("minkowski", "mi"),
        ("minkowski", "m"),
        ("minkowski", "pnorm"),
        ("sqeuclidean", "sqe"),
        ("sqeuclidean", "sqeuclid"),
    ],
)
def test_distance_aliases(valid_poses_dataset, keep_attrs, canonical, alias):
    """Equivalent supported metric spellings have identical data and attrs."""
    data = valid_poses_dataset.position.copy(deep=True)
    data.attrs = {"units": "mm", "log": "[]", "foo": 1}
    with xr.set_options(keep_attrs=keep_attrs):
        expected = kinematics.compute_pairwise_distances(
            data, "individual", {"id_0": "id_1"}, metric=canonical
        )
        actual = kinematics.compute_pairwise_distances(
            data, "individual", {"id_0": "id_1"}, metric=alias
        )
    xr.testing.assert_identical(actual, expected)
    assert actual.attrs == {
        "units": "mm^2" if canonical == "sqeuclidean" else "mm",
        "log": "[]",
        "foo": 1,
    }


@pytest.mark.parametrize("keep_attrs", [True, False])
def test_datetime_derivative_units(valid_poses_dataset, keep_attrs):
    """Do not label datetime derivatives using a numeric time-unit hint."""
    data = valid_poses_dataset.position.copy(deep=True)
    time = np.datetime64("2020-01-01") + np.arange(data.sizes["time"]).astype(
        "timedelta64[s]"
    )
    data = data.assign_coords(time=time)
    data.attrs = {"units": "mm", "time_unit": "seconds", "log": "[]"}
    data.time.attrs["units"] = "seconds"
    original = data.copy(deep=True)
    with xr.set_options(keep_attrs=keep_attrs):
        result = kinematics.compute_time_derivative(data, order=1)
        reference = data.differentiate("time")
    assert result.attrs == {"time_unit": "seconds", "log": "[]"}
    xr.testing.assert_equal(result, reference)
    xr.testing.assert_identical(data, original)
