"""Pinhole camera model: projection, back-projection, ground-plane intersection, triangulation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from chest_hunter.vision.perspective import (
    CameraIntrinsics,
    CameraPose,
    angle_difference_deg,
    bearing_deg,
    camera_basis,
    intersect_horizontal_plane,
    pixel_ray,
    pixel_rays,
    project,
    project_point,
    triangulate_rays,
)

INTR = CameraIntrinsics(640, 360, 90.0)


def test_intrinsics_follow_the_horizontal_fov() -> None:
    assert INTR.fx == pytest.approx(320.0)                 # tan(45 deg) = 1
    assert INTR.fy == INTR.fx and (INTR.cx, INTR.cy) == (320.0, 180.0)
    assert INTR.vfov_deg == pytest.approx(2 * math.degrees(math.atan(180 / 320)))
    assert CameraIntrinsics.from_dict(INTR.to_dict()) == INTR


def test_intrinsics_reject_inconsistent_or_invalid_data() -> None:
    bad = INTR.to_dict() | {"fx": 500.0}
    with pytest.raises(ValueError):
        CameraIntrinsics.from_dict(bad)
    with pytest.raises(ValueError):
        CameraIntrinsics(8, 4, 90.0)
    with pytest.raises(ValueError):
        CameraIntrinsics(640, 360, 179.0)


def test_basis_is_orthonormal_and_right_handed_in_ue_axes() -> None:
    for yaw, pitch in [(0, 0), (90, 0), (37, -20), (250, 15)]:
        f, r, u = camera_basis(yaw, pitch)
        m = np.stack([f, r, u])
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)
    f, r, u = camera_basis(0.0, 0.0)
    assert np.allclose(f, [1, 0, 0]) and np.allclose(r, [0, 1, 0]) and np.allclose(u, [0, 0, 1])


def test_point_straight_ahead_projects_to_the_principal_point() -> None:
    pose = CameraPose(0, 0, 1.6, 30.0, -10.0)
    f, _, _ = pose.basis()
    p = pose.position + 7.0 * f
    u, v, d = project_point(tuple(p), pose, INTR)              # type: ignore[arg-type,misc]
    assert (u, v, d) == pytest.approx((320.0, 180.0, 7.0))


def test_right_and_up_map_to_image_right_and_up() -> None:
    pose = CameraPose(0, 0, 0, 0.0, 0.0)
    u, v, _ = project(np.array([[5.0, 1.0, 0.0], [5.0, 0.0, 1.0]]), pose, INTR)
    assert u[0] > INTR.cx and v[1] < INTR.cy


def test_points_behind_the_camera_are_not_projected() -> None:
    pose = CameraPose(0, 0, 0, 0.0, 0.0)
    assert project_point((-3.0, 0.0, 0.0), pose, INTR) is None
    u, _, d = project(np.array([-3.0, 0.0, 0.0]), pose, INTR)
    assert float(d) < 0 and np.isnan(u)


@pytest.mark.parametrize(("u", "v"), [(10.0, 20.0), (320.0, 180.0), (600.0, 340.0)])
def test_pixel_ray_inverts_projection(u: float, v: float) -> None:
    pose = CameraPose(3, -2, 1.7, 123.0, -25.0)
    ray = pixel_ray(u, v, pose, INTR)
    assert np.linalg.norm(ray) == pytest.approx(1.0)
    pu, pv, _ = project_point(tuple(pose.position + 9.0 * ray), pose, INTR)  # type: ignore[misc,arg-type]
    assert (pu, pv) == pytest.approx((u, v), abs=1e-9)
    rays = pixel_rays(np.array([u]), np.array([v]), pose, INTR)
    assert np.allclose(rays[0], ray)


def test_ground_plane_intersection_recovers_distance() -> None:
    pose = CameraPose(0, 0, 2.0, 0.0, -20.0)
    target = np.array([6.0, 1.0, 0.0])
    u, v, _ = project_point(tuple(target), pose, INTR)        # type: ignore[misc,arg-type]
    hit = intersect_horizontal_plane(pose.position, pixel_ray(u, v, pose, INTR), 0.0)
    assert hit is not None and np.allclose(hit, target, atol=1e-9)
    # A ray pointing up never meets a plane below it; a horizontal ray never meets it either.
    assert intersect_horizontal_plane(pose.position, np.array([1.0, 0, 0.2]), 0.0) is None
    assert intersect_horizontal_plane(pose.position, np.array([1.0, 0, 0.0]), 0.0) is None


def test_bearings_and_angle_differences() -> None:
    assert bearing_deg(1, 0) == 0.0 and bearing_deg(0, 1) == 90.0
    assert bearing_deg(-1, 0) == 180.0 and bearing_deg(0, -1) == 270.0
    assert angle_difference_deg(10, 350) == pytest.approx(20.0)
    assert angle_difference_deg(350, 10) == pytest.approx(-20.0)
    assert angle_difference_deg(0, 180) == 180.0


def test_triangulation_of_two_views() -> None:
    target = np.array([10.0, 4.0, 0.3])
    origins = np.array([[0.0, 0.0, 1.6], [0.0, 6.0, 1.6], [3.0, -2.0, 1.6]])
    dirs = target - origins
    point, conditioning = triangulate_rays(origins, dirs)
    assert np.allclose(point, target, atol=1e-9) and conditioning > 0.01
    # Parallel rays: no baseline -> conditioning ~ 0.
    _, bad = triangulate_rays(origins[:2], np.array([[1.0, 0, 0], [1.0, 0, 0]]))
    assert bad < 1e-9
