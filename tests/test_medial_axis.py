import numpy as np
import pytest

from pymedax import medial_axis


def test_rectangle_has_center_axis_and_diagnostic_maps():
    image = np.zeros((9, 11), dtype=np.uint8)
    image[2:7, 3:8] = 1

    default_axis = medial_axis(image)
    axis, distance, closest = medial_axis(image, return_maps=True)

    assert isinstance(default_axis, np.ndarray)
    assert np.array_equal(default_axis, axis)
    assert axis.dtype == np.uint8
    assert axis.shape == image.shape
    assert np.all(axis[0] == 0)
    assert np.all(axis[:, 0] == 0)
    assert np.all(axis[3:6, 4:7] == 1)
    assert distance[4, 5] == pytest.approx(2.0)
    assert np.all(closest[image == 0] == -1)
    assert np.all(np.isfinite(distance[image != 0]))


def test_empty_image_has_no_axis_or_sources():
    image = np.zeros((4, 5), dtype=np.uint8)

    axis, distance, closest = medial_axis(image, return_maps=True)

    assert not axis.any()
    assert np.all(np.isinf(distance))
    assert np.all(closest == -1)


def test_angle_threshold_only_removes_low_angle_front_merges():
    image = np.zeros((9, 11), dtype=np.uint8)
    image[2:7, 3:8] = 1

    permissive_axis = medial_axis(image, angle_threshold_degrees=0.0)
    filtered_axis = medial_axis(image, angle_threshold_degrees=179.0)

    assert permissive_axis.any()
    assert filtered_axis.any()
    assert filtered_axis.sum() <= permissive_axis.sum()


def test_angle_threshold_must_be_valid():
    image = np.ones((3, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        medial_axis(image, angle_threshold_degrees=-1.0)
    with pytest.raises(ValueError):
        medial_axis(image, angle_threshold_degrees=181.0)


def test_minimum_distance_removes_near_boundary_axis_pixels():
    image = np.zeros((9, 11), dtype=np.uint8)
    image[2:7, 3:8] = 1

    axis, distance, _ = medial_axis(image, return_maps=True)
    filtered_axis = medial_axis(image, minimum_distance=2.1)

    assert axis.any()
    assert np.max(distance[axis != 0]) == pytest.approx(2.0)
    assert not filtered_axis.any()


def test_minimum_distance_must_be_valid():
    image = np.ones((3, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        medial_axis(image, minimum_distance=-1.0)
    with pytest.raises(ValueError):
        medial_axis(image, minimum_distance=float("inf"))


def test_requires_contiguous_uint8_2d_input():
    with pytest.raises(TypeError):
        medial_axis(np.ones((3, 3), dtype=np.int32))

    with pytest.raises(ValueError):
        medial_axis(np.ones((3, 3, 1), dtype=np.uint8))

    with pytest.raises(ValueError):
        medial_axis(np.ones((4, 4), dtype=np.uint8)[:, ::2])
