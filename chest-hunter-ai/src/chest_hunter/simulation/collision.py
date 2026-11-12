"""Occupancy grid and grid-based collision detection.

Coordinates are continuous *grid units*: cell ``(cx, cy)`` covers the square
``[cx, cx + 1) x [cy, cy + 1)``. ``y`` grows to the north (the renderer flips
it for the screen). Anything outside the grid counts as blocked.

The agent is a circle; :func:`move_with_collision` resolves movement one axis
at a time so the agent slides along walls instead of sticking to them.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

Cell = tuple[int, int]
Point = tuple[float, float]


class GridMap:
    """A width x height occupancy grid stored in a ``bytearray`` (1 = blocked).

    ``version`` increases whenever the occupancy changes, so path and distance
    caches can tell when they are stale.
    """

    __slots__ = ("width", "height", "_cells", "version")

    def __init__(self, width: int, height: int) -> None:
        if width < 1 or height < 1:
            raise ValueError("Grid dimensions must be positive.")
        self.width = width
        self.height = height
        self._cells = bytearray(width * height)
        self.version = 0

    # -- queries ------------------------------------------------------------

    @property
    def cells(self) -> bytearray:
        """Raw row-major occupancy (index ``y * width + x``). Treat as read-only."""
        return self._cells

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def index(self, x: int, y: int) -> int:
        return y * self.width + x

    def is_blocked(self, x: int, y: int) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        return self._cells[y * self.width + x] != 0

    def is_walkable(self, x: int, y: int) -> bool:
        return not self.is_blocked(x, y)

    @property
    def blocked_count(self) -> int:
        return self._cells.count(1)

    @property
    def blocked_fraction(self) -> float:
        return self.blocked_count / len(self._cells)

    @staticmethod
    def cell_of(x: float, y: float) -> Cell:
        return (math.floor(x), math.floor(y))

    @staticmethod
    def cell_center(cell: Cell) -> Point:
        return (cell[0] + 0.5, cell[1] + 0.5)

    def walkable_cells(self) -> Iterator[Cell]:
        width = self.width
        for index, value in enumerate(self._cells):
            if not value:
                yield (index % width, index // width)

    # -- mutation -----------------------------------------------------------

    def set_blocked(self, x: int, y: int, blocked: bool = True) -> None:
        if not self.in_bounds(x, y):
            raise ValueError(f"Cell ({x}, {y}) is outside the {self.width}x{self.height} grid.")
        value = 1 if blocked else 0
        index = y * self.width + x
        if self._cells[index] != value:
            self._cells[index] = value
            self.version += 1

    def fill_rect(self, x: int, y: int, width: int, height: int, blocked: bool = True) -> int:
        """Set every in-bounds cell of the rectangle; return how many changed."""
        value = 1 if blocked else 0
        changed = 0
        for cy in range(max(0, y), min(self.height, y + height)):
            row = cy * self.width
            for cx in range(max(0, x), min(self.width, x + width)):
                if self._cells[row + cx] != value:
                    self._cells[row + cx] = value
                    changed += 1
        if changed:
            self.version += 1
        return changed

    def copy(self) -> GridMap:
        clone = GridMap(self.width, self.height)
        clone._cells[:] = self._cells
        clone.version = self.version
        return clone

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridMap):
            return NotImplemented
        return (self.width, self.height, self._cells) == (other.width, other.height, other._cells)

    __hash__ = None  # type: ignore[assignment]  # mutable


# ---------------------------------------------------------------------------
# Circle collision
# ---------------------------------------------------------------------------


def circle_collides(grid: GridMap, x: float, y: float, radius: float) -> bool:
    """Whether a circle at ``(x, y)`` overlaps any blocked (or out-of-bounds) cell."""
    r2 = radius * radius
    for cy in range(math.floor(y - radius), math.floor(y + radius) + 1):
        for cx in range(math.floor(x - radius), math.floor(x + radius) + 1):
            if grid.is_blocked(cx, cy):
                nearest_x = cx if x < cx else (cx + 1 if x > cx + 1 else x)
                nearest_y = cy if y < cy else (cy + 1 if y > cy + 1 else y)
                dx = x - nearest_x
                dy = y - nearest_y
                if dx * dx + dy * dy < r2 - 1e-12:
                    return True
    return False


def move_with_collision(
    grid: GridMap, x: float, y: float, dx: float, dy: float, radius: float
) -> tuple[float, float, bool, bool]:
    """Move a circle by ``(dx, dy)`` with axis-separated sliding.

    Returns ``(new_x, new_y, blocked_x, blocked_y)``. The caller should keep
    per-tick displacement below ``radius`` so the circle cannot tunnel.
    """
    if not circle_collides(grid, x + dx, y + dy, radius):
        return x + dx, y + dy, False, False
    blocked_x = blocked_y = False
    if dx and not circle_collides(grid, x + dx, y, radius):
        x += dx
    elif dx:
        blocked_x = True
    if dy and not circle_collides(grid, x, y + dy, radius):
        y += dy
    elif dy:
        blocked_y = True
    return x, y, blocked_x, blocked_y


# ---------------------------------------------------------------------------
# Line of sight
# ---------------------------------------------------------------------------


def supercover_cells(a: Point, b: Point) -> Iterator[Cell]:
    """Yield every cell the segment ``a -> b`` touches, in order.

    When the segment passes exactly through a cell corner, both cells beside
    the corner are yielded (conservative, so diagonal squeezes are rejected).
    """
    x0, y0 = a
    x1, y1 = b
    cx, cy = math.floor(x0), math.floor(y0)
    end_x, end_y = math.floor(x1), math.floor(y1)
    dx, dy = x1 - x0, y1 - y0
    step_x = (dx > 0) - (dx < 0)
    step_y = (dy > 0) - (dy < 0)
    t_max_x = ((cx + (step_x > 0)) - x0) / dx if dx else math.inf
    t_max_y = ((cy + (step_y > 0)) - y0) / dy if dy else math.inf
    t_delta_x = abs(1.0 / dx) if dx else math.inf
    t_delta_y = abs(1.0 / dy) if dy else math.inf

    yield (cx, cy)
    remaining = abs(end_x - cx) + abs(end_y - cy) + 2
    while (cx, cy) != (end_x, end_y) and remaining > 0:
        remaining -= 1
        if abs(t_max_x - t_max_y) < 1e-12:
            if min(t_max_x, t_max_y) > 1.0:
                break
            yield (cx + step_x, cy)
            yield (cx, cy + step_y)
            cx += step_x
            cy += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            if t_max_x > 1.0:
                break
            cx += step_x
            t_max_x += t_delta_x
        else:
            if t_max_y > 1.0:
                break
            cy += step_y
            t_max_y += t_delta_y
        yield (cx, cy)


def segment_clear(grid: GridMap, a: Point, b: Point, radius: float = 0.0) -> bool:
    """Whether a circle of ``radius`` can sweep from ``a`` to ``b`` without collision.

    Checks the centre line plus the two lines offset by ``radius`` on either
    side (a unit cell cannot fit between them when ``radius`` is below 0.5)
    and the circles at both ends.
    """
    if radius > 0.0 and (
        circle_collides(grid, a[0], a[1], radius) or circle_collides(grid, b[0], b[1], radius)
    ):
        return False
    lines: list[tuple[Point, Point]] = [(a, b)]
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if radius > 0.0 and length > 1e-12:
        nx = -(b[1] - a[1]) / length * radius
        ny = (b[0] - a[0]) / length * radius
        lines.append(((a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny)))
        lines.append(((a[0] - nx, a[1] - ny), (b[0] - nx, b[1] - ny)))
    for start, end in lines:
        for cx, cy in supercover_cells(start, end):
            if grid.is_blocked(cx, cy):
                return False
    return True
