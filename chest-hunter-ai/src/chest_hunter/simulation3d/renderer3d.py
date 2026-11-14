"""Software renderer for the reference arena: perspective frames with a z-buffer.

This stands in for the Unreal ``USceneCaptureComponent2D`` render target in
environments without Unreal Engine. Faces are rasterised with OpenCV
(``fillConvexPoly``) inside their screen bounding box; the inverse depth of a
planar face is an affine function of the pixel coordinates, so a per-pixel
z-buffer test costs a few vectorised operations per face. Occlusion is
therefore exact, which matters for the evaluation labels.

Output: a BGR ``uint8`` frame (flat shading, sun lighting, distance fog,
seeded sensor noise and brightness change) and, on request, an object-ID
buffer used only by evaluation code.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from chest_hunter.simulation3d.geometry import COLOURS, Box, World3D
from chest_hunter.vision.perspective import CameraIntrinsics, CameraPose

SUN = np.array([0.35, 0.25, 0.90]) / np.linalg.norm([0.35, 0.25, 0.90])
AMBIENT = 0.55
DIFFUSE = 0.45
AGENT_ID = 30000          # label id of the character's own body
BACKGROUND_ID = 0


@dataclass(frozen=True, slots=True)
class Face:
    vertices: np.ndarray          # (4, 3) world
    normal: np.ndarray            # (3,)
    colour: tuple[float, float, float]
    owner: int                    # label id (0 = static scenery)


@dataclass(slots=True)
class Billboard:
    """A camera-facing overlay anchored in the world (prompt, progress bar, reward)."""

    kind: str                     # "prompt", "progress" or "reward"
    position: tuple[float, float, float]
    size: float                   # metres
    value: float = 0.0            # progress fraction


def box_faces(b: Box, owner: int = 0) -> list[Face]:
    """Visible-candidate faces of an axis-aligned box (bottom faces only when raised)."""
    x0, y0, z0, x1, y1, z1 = b.x0, b.y0, b.z0, b.x1, b.y1, b.z1
    base = COLOURS[b.colour]
    faces = [
        Face(np.array([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]),
             np.array([0.0, 0.0, 1.0]), base, owner),
        Face(np.array([[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]]),
             np.array([-1.0, 0.0, 0.0]), base, owner),
        Face(np.array([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]),
             np.array([1.0, 0.0, 0.0]), base, owner),
        Face(np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]),
             np.array([0.0, -1.0, 0.0]), base, owner),
        Face(np.array([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]]),
             np.array([0.0, 1.0, 0.0]), base, owner),
    ]
    if z0 > 0.05:
        under = COLOURS["ceiling"] if b.kind in ("floor", "roof", "platform") else base
        faces.append(Face(np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]),
                          np.array([0.0, 0.0, -1.0]), under, owner))
    return faces


def oriented_box_faces(cx: float, cy: float, z0: float, width: float, depth: float,
                       height: float, yaw_deg: float, colour: str, owner: int) -> list[Face]:
    """Faces of a box rotated about Z (``depth`` along the yaw direction)."""
    yaw = math.radians(yaw_deg)
    f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    r = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    c = np.array([cx, cy, z0])
    hd, hw = depth / 2, width / 2
    corners = [c + sf * hd * f + sr * hw * r for sf, sr in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    up = np.array([0.0, 0.0, height])
    bottom = np.array(corners)
    top = bottom + up
    base = COLOURS[colour]
    faces = [Face(top, np.array([0.0, 0.0, 1.0]), base, owner)]
    normals = [-r, f, r, -f]
    for k in range(4):
        a, b_ = bottom[k], bottom[(k + 1) % 4]
        quad = np.array([a, b_, b_ + up, a + up])
        n = normals[k]
        faces.append(Face(quad, n, base, owner))
    return faces


class Renderer3D:
    NOISE_BANK = 6

    def __init__(self, world: World3D, intr: CameraIntrinsics, *, noise_std: float = 5.0,
                 lighting_variation: float = 0.08, fog_start: float = 45.0,
                 fog_end: float = 140.0, seed: int = 0) -> None:
        self.world = world
        self.intr = intr
        self.noise_std = noise_std
        self.lighting_variation = lighting_variation
        self.fog_start, self.fog_end = fog_start, fog_end
        self.rng = np.random.default_rng(seed)
        static = [f for b in world.boxes if b.kind != "ground" for f in box_faces(b)]
        self._static_v = np.array([f.vertices for f in static], dtype=np.float64).reshape(-1, 4, 3)
        self._static_n = np.array([f.normal for f in static], dtype=np.float64).reshape(-1, 3)
        self._static_c = np.array([f.colour for f in static], dtype=np.float64).reshape(-1, 3)
        self._static_owner = np.zeros(len(static), dtype=np.int32)
        bx0, by0, bx1, by1 = world.bounds
        gz = world.ground_z
        self._ground = Face(np.array([[bx0 - 400, by0 - 400, gz], [bx1 + 400, by0 - 400, gz],
                                      [bx1 + 400, by1 + 400, gz], [bx0 - 400, by1 + 400, gz]]),
                            np.array([0.0, 0.0, 1.0]), COLOURS["grass"], 0)
        self._resize(intr)
        self.faces_drawn = 0

    def _resize(self, intr: CameraIntrinsics) -> None:
        self.intr = intr
        h, w = intr.height, intr.width
        self._u = np.broadcast_to(np.arange(w, dtype=np.float32) + 0.5, (h, w))
        self._v = np.broadcast_to((np.arange(h, dtype=np.float32) + 0.5)[:, None], (h, w))
        self._zbuf = np.zeros((h, w), np.float32)
        self._frame = np.empty((h, w, 3), np.uint8)
        self._ids = np.zeros((h, w), np.int32)
        self._mask = np.zeros((h, w), np.uint8)
        self._noise_pos: list[np.ndarray] = []
        self._noise_neg: list[np.ndarray] = []
        if self.noise_std > 0:
            for _ in range(self.NOISE_BANK):
                n = self.rng.normal(0.0, self.noise_std, (h, w, 3))
                self._noise_pos.append(np.clip(n, 0, 255).astype(np.uint8))
                self._noise_neg.append(np.clip(-n, 0, 255).astype(np.uint8))

    # -- rendering -------------------------------------------------------------------------

    def render(self, pose: CameraPose, dynamic: Sequence[Face], billboards: Sequence[Billboard],
               *, labels: bool = False, lighting: float = 1.0,
               noise: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        intr = self.intr
        fx, fy, cx, cy, near = intr.fx, intr.fy, intr.cx, intr.cy, intr.near
        frame, zbuf = self._frame, self._zbuf
        frame[:] = COLOURS["sky"]
        zbuf[:] = 0.0
        if labels:
            self._ids[:] = BACKGROUND_ID
        rot = pose.rotation()
        cam = pose.position
        self.faces_drawn = 0

        # All faces (static scenery, ground, chests, character) in one vectorised pass.
        if dynamic:
            verts = np.concatenate([self._static_v, [self._ground.vertices],
                                    np.stack([f.vertices for f in dynamic])])
            normals = np.concatenate([self._static_n, [self._ground.normal],
                                      np.stack([f.normal for f in dynamic])])
            colours = np.concatenate([self._static_c, [self._ground.colour],
                                      np.array([f.colour for f in dynamic], dtype=np.float64)])
            owners = np.concatenate([self._static_owner, [0],
                                     np.array([f.owner for f in dynamic], dtype=np.int32)])
        else:
            verts = np.concatenate([self._static_v, [self._ground.vertices]])
            normals = np.concatenate([self._static_n, [self._ground.normal]])
            colours = np.concatenate([self._static_c, [self._ground.colour]])
            owners = np.concatenate([self._static_owner, [0]])
        centres = verts.mean(axis=1)
        facing = np.einsum("ij,ij->i", normals, centres - cam) < 0.0
        cam_v = (verts - cam) @ rot.T                                   # (F, 4, 3)
        z = cam_v[..., 2]
        zc = np.maximum(z, 0.0)
        tan_h, tan_v = cx / fx * 1.05, cy / fy * 1.05
        outside = ((cam_v[..., 0] < -tan_h * zc).all(axis=1) | (cam_v[..., 0] > tan_h * zc).all(axis=1)
                   | (cam_v[..., 1] < -tan_v * zc).all(axis=1) | (cam_v[..., 1] > tan_v * zc).all(axis=1))
        visible = facing & (z > near).any(axis=1) & ~outside
        idx = np.nonzero(visible)[0]
        if idx.size == 0:
            return self._finish(frame, labels, lighting, noise)
        dist = np.linalg.norm(centres[idx] - cam, axis=1)
        order = np.argsort(dist)
        idx, dist = idx[order], dist[order]
        shade = self._shade(normals[idx], colours[idx], lighting, dist).astype(np.uint8)
        n_cam = normals[idx] @ rot.T
        cv_all = cam_v[idx]
        crossing = (cv_all[..., 2] < near).any(axis=1)
        d_all = np.einsum("ij,ij->i", n_cam, cv_all[:, 0])
        for k in range(idx.size):
            cv = cv_all[k]
            if crossing[k]:
                clipped = _clip_near(cv, near)
                if clipped is None:
                    continue
                cv = clipped
            d = float(d_all[k])
            if abs(d) < 1e-9:
                continue
            nc = n_cam[k]
            a = nc[0] / (fx * d)
            b = -nc[1] / (fy * d)
            c = (-nc[0] * cx / fx + nc[1] * cy / fy + nc[2]) / d
            u = cx + fx * cv[:, 0] / cv[:, 2]
            v = cy - fy * cv[:, 1] / cv[:, 2]
            self._raster(u, v, a, b, c, shade[k], int(owners[idx[k]]), labels)
        for bb in billboards:
            self._draw_billboard(bb, pose)
        return self._finish(frame, labels, lighting, noise)

    def _finish(self, frame: np.ndarray, labels: bool, lighting: float,
                noise: bool) -> tuple[np.ndarray, np.ndarray | None]:
        out = self._sensor(frame, lighting) if noise else frame.copy()
        return out, (self._ids.copy() if labels else None)

    def _shade(self, normals: np.ndarray, colours: np.ndarray, lighting: float,
               distances: np.ndarray) -> np.ndarray:
        diffuse = np.clip(normals @ SUN, 0.0, None)
        factor = (AMBIENT + DIFFUSE * diffuse) * lighting
        shaded = colours * factor[:, None]
        fog = np.clip((distances - self.fog_start) / (self.fog_end - self.fog_start), 0.0, 0.55)
        sky = np.array(COLOURS["sky"], dtype=np.float64)
        return np.clip(shaded * (1 - fog[:, None]) + sky * fog[:, None], 0, 255)

    def _raster(self, u: np.ndarray, v: np.ndarray, a: float, b: float, c: float,
                colour: np.ndarray, owner: int, labels: bool) -> None:
        w, h = self.intr.width, self.intr.height
        x0 = max(0, int(math.floor(float(u.min()))))
        x1 = min(w, int(math.ceil(float(u.max()))) + 1)
        y0 = max(0, int(math.floor(float(v.min()))))
        y1 = min(h, int(math.ceil(float(v.max()))) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        mask = self._mask[y0:y1, x0:x1]
        mask[:] = 0
        pts = np.empty((len(u), 2), np.int32)
        pts[:, 0] = np.round((u - x0) * 16)
        pts[:, 1] = np.round((v - y0) * 16)
        cv2.fillConvexPoly(mask, pts, 1, lineType=cv2.LINE_8, shift=4)
        # Inverse depth of the face plane is affine in the pixel coordinates.
        inv = self._u[y0:y1, x0:x1] * np.float32(a)
        inv += self._v[y0:y1, x0:x1] * np.float32(b)
        inv += np.float32(c)
        zb = self._zbuf[y0:y1, x0:x1]
        m = inv > zb
        m &= mask.view(bool)
        if not m.any():
            return
        np.copyto(zb, inv, where=m)
        self._frame[y0:y1, x0:x1][m] = colour
        if labels:
            self._ids[y0:y1, x0:x1][m] = owner
        self.faces_drawn += 1

    def _draw_billboard(self, bb: Billboard, pose: CameraPose) -> None:
        intr = self.intr
        cv = (np.asarray(bb.position) - pose.position) @ pose.rotation().T
        if cv[2] <= intr.near:
            return
        u = intr.cx + intr.fx * cv[0] / cv[2]
        v = intr.cy - intr.fy * cv[1] / cv[2]
        if not (0 <= u < intr.width and 0 <= v < intr.height):
            return
        if float(self._zbuf[int(v), int(u)]) > 1.0 / cv[2] + 1e-4:
            return                                     # hidden behind geometry
        half = max(2.0, intr.fx * bb.size / cv[2] / 2.0)
        frame = self._frame
        if bb.kind == "prompt":
            p0 = (int(u - half), int(v - half))
            p1 = (int(u + half), int(v + half))
            cv2.rectangle(frame, p0, p1, COLOURS["prompt"], -1)
            cv2.rectangle(frame, p0, p1, COLOURS["prompt_text"], 1)
            scale = max(0.25, half / 11.0)
            (tw, th), _ = cv2.getTextSize("E", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            cv2.putText(frame, "E", (int(u - tw / 2), int(v + th / 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, COLOURS["prompt_text"], max(1, int(scale * 2)), cv2.LINE_AA)
        elif bb.kind == "progress":
            hh = max(1.5, half * 0.14)
            p0 = (int(u - half), int(v - hh))
            p1 = (int(u + half), int(v + hh))
            cv2.rectangle(frame, p0, p1, COLOURS["progress_frame"], -1)
            fill = int(round((p1[0] - p0[0] - 2) * min(1.0, max(0.0, bb.value))))
            if fill > 0:
                cv2.rectangle(frame, (p0[0] + 1, p0[1] + 1), (p0[0] + 1 + fill, p1[1] - 1),
                              COLOURS["progress_fill"], -1)
        elif bb.kind == "reward":
            cv2.circle(frame, (int(u), int(v)), int(half), COLOURS["reward"], -1, cv2.LINE_AA)

    def _sensor(self, frame: np.ndarray, lighting: float) -> np.ndarray:
        out = frame
        if self._noise_pos:
            k = int(self.rng.integers(len(self._noise_pos)))
            out = cv2.add(frame, self._noise_pos[k])
            cv2.subtract(out, self._noise_neg[k], dst=out)
        return out


def _clip_near(cv: np.ndarray, near: float) -> np.ndarray | None:
    """Sutherland-Hodgman clip of a camera-space polygon against ``z >= near``."""
    out = []
    n = len(cv)
    for k in range(n):
        a, b = cv[k], cv[(k + 1) % n]
        a_in, b_in = a[2] >= near, b[2] >= near
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = (near - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    if len(out) < 3:
        return None
    return np.array(out)
