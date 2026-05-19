import numpy as np
import xarray as xr

from movement.utils import compute_vertex_angle


def test_right_angle():
    p1 = np.array([1, 0])
    vtx = np.array([0, 0])
    p3 = np.array([0, 1])
    angle = compute_vertex_angle(p1, vtx, p3)
    assert np.isclose(angle, np.pi / 2)


def test_straight_line():
    p1 = np.array([-1, 0])
    vtx = np.array([0, 0])
    p3 = np.array([1, 0])
    angle = compute_vertex_angle(p1, vtx, p3)
    assert np.isclose(angle, np.pi)


def test_colinear():
    p1 = np.array([1, 1])
    vtx = np.array([0, 0])
    p3 = np.array([-1, -1])
    angle = compute_vertex_angle(p1, vtx, p3)
    assert np.isclose(angle, np.pi)


def test_xarray_input():
    p1 = xr.DataArray([1, 0], dims=["space"], coords={"space": ["x", "y"]})
    vtx = xr.DataArray([0, 0], dims=["space"], coords={"space": ["x", "y"]})
    p3 = xr.DataArray([0, 1], dims=["space"], coords={"space": ["x", "y"]})
    angle = compute_vertex_angle(p1, vtx, p3)
    assert np.isclose(angle, np.pi / 2)
