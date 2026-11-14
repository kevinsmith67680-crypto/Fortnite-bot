"""Pinhole camera model: projection, back-projection, ground-plane intersection, triangulation."""

from __future__ import annotations

import math
import unittest

import numpy as np

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

# Approximate checks use a relative tolerance of 1e-6 (delta = 1e-6 * |expected|) unless
# they state an absolute one.


class TestIntrinsics(unittest.TestCase):
    def test_intrinsics_follow_the_horizontal_fov(self) -> None:
        self.assertAlmostEqual(INTR.fx, 320.0, delta=1e-6 * 320.0)  # tan(45 deg) = 1
        self.assertEqual(INTR.fy, INTR.fx)
        self.assertEqual((INTR.cx, INTR.cy), (320.0, 180.0))
        vfov = 2 * math.degrees(math.atan(180 / 320))
        self.assertAlmostEqual(INTR.vfov_deg, vfov, delta=1e-6 * vfov)
        self.assertEqual(CameraIntrinsics.from_dict(INTR.to_dict()), INTR)

    def test_intrinsics_reject_inconsistent_or_invalid_data(self) -> None:
        bad = INTR.to_dict() | {"fx": 500.0}
        with self.assertRaises(ValueError):
            CameraIntrinsics.from_dict(bad)
        with self.assertRaises(ValueError):
            CameraIntrinsics(8, 4, 90.0)
        with self.assertRaises(ValueError):
            CameraIntrinsics(640, 360, 179.0)


class TestProjection(unittest.TestCase):
    def test_basis_is_orthonormal_and_right_handed_in_ue_axes(self) -> None:
        for yaw, pitch in [(0, 0), (90, 0), (37, -20), (250, 15)]:
            f, r, u = camera_basis(yaw, pitch)
            m = np.stack([f, r, u])
            self.assertTrue(np.allclose(m @ m.T, np.eye(3), atol=1e-12))
        f, r, u = camera_basis(0.0, 0.0)
        self.assertTrue(np.allclose(f, [1, 0, 0]))
        self.assertTrue(np.allclose(r, [0, 1, 0]))
        self.assertTrue(np.allclose(u, [0, 0, 1]))

    def test_point_straight_ahead_projects_to_the_principal_point(self) -> None:
        pose = CameraPose(0, 0, 1.6, 30.0, -10.0)
        f, _, _ = pose.basis()
        p = pose.position + 7.0 * f
        u, v, d = project_point(tuple(p), pose, INTR)              # type: ignore[arg-type,misc]
        for got, want in zip((u, v, d), (320.0, 180.0, 7.0), strict=True):
            self.assertAlmostEqual(got, want, delta=1e-6 * want)

    def test_right_and_up_map_to_image_right_and_up(self) -> None:
        pose = CameraPose(0, 0, 0, 0.0, 0.0)
        u, v, _ = project(np.array([[5.0, 1.0, 0.0], [5.0, 0.0, 1.0]]), pose, INTR)
        self.assertGreater(u[0], INTR.cx)
        self.assertLess(v[1], INTR.cy)

    def test_points_behind_the_camera_are_not_projected(self) -> None:
        pose = CameraPose(0, 0, 0, 0.0, 0.0)
        self.assertIsNone(project_point((-3.0, 0.0, 0.0), pose, INTR))
        u, _, d = project(np.array([-3.0, 0.0, 0.0]), pose, INTR)
        self.assertLess(float(d), 0)
        self.assertTrue(np.isnan(u))

    def test_pixel_ray_inverts_projection(self) -> None:
        for u, v in [(10.0, 20.0), (320.0, 180.0), (600.0, 340.0)]:
            with self.subTest(u=u, v=v):
                pose = CameraPose(3, -2, 1.7, 123.0, -25.0)
                ray = pixel_ray(u, v, pose, INTR)
                self.assertAlmostEqual(np.linalg.norm(ray), 1.0, delta=1e-6)
                point = tuple(pose.position + 9.0 * ray)
                pu, pv, _ = project_point(point, pose, INTR)  # type: ignore[misc,arg-type]
                self.assertAlmostEqual(pu, u, delta=1e-9)
                self.assertAlmostEqual(pv, v, delta=1e-9)
                rays = pixel_rays(np.array([u]), np.array([v]), pose, INTR)
                self.assertTrue(np.allclose(rays[0], ray))


class TestGeometry(unittest.TestCase):
    def test_ground_plane_intersection_recovers_distance(self) -> None:
        pose = CameraPose(0, 0, 2.0, 0.0, -20.0)
        target = np.array([6.0, 1.0, 0.0])
        u, v, _ = project_point(tuple(target), pose, INTR)        # type: ignore[misc,arg-type]
        hit = intersect_horizontal_plane(pose.position, pixel_ray(u, v, pose, INTR), 0.0)
        self.assertIsNotNone(hit)
        self.assertTrue(np.allclose(hit, target, atol=1e-9))  # type: ignore[arg-type]
        # A ray pointing up never meets a plane below it; a horizontal ray never meets it either.
        self.assertIsNone(intersect_horizontal_plane(pose.position, np.array([1.0, 0, 0.2]), 0.0))
        self.assertIsNone(intersect_horizontal_plane(pose.position, np.array([1.0, 0, 0.0]), 0.0))

    def test_bearings_and_angle_differences(self) -> None:
        self.assertEqual(bearing_deg(1, 0), 0.0)
        self.assertEqual(bearing_deg(0, 1), 90.0)
        self.assertEqual(bearing_deg(-1, 0), 180.0)
        self.assertEqual(bearing_deg(0, -1), 270.0)
        self.assertAlmostEqual(angle_difference_deg(10, 350), 20.0, delta=1e-6 * 20.0)
        self.assertAlmostEqual(angle_difference_deg(350, 10), -20.0, delta=1e-6 * 20.0)
        self.assertEqual(angle_difference_deg(0, 180), 180.0)

    def test_triangulation_of_two_views(self) -> None:
        target = np.array([10.0, 4.0, 0.3])
        origins = np.array([[0.0, 0.0, 1.6], [0.0, 6.0, 1.6], [3.0, -2.0, 1.6]])
        dirs = target - origins
        point, conditioning = triangulate_rays(origins, dirs)
        self.assertTrue(np.allclose(point, target, atol=1e-9))
        self.assertGreater(conditioning, 0.01)
        # Parallel rays: no baseline -> conditioning ~ 0.
        _, bad = triangulate_rays(origins[:2], np.array([[1.0, 0, 0], [1.0, 0, 0]]))
        self.assertLess(bad, 1e-9)
