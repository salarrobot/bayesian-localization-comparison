"""
Shared simulation environment for the SLAM / localization comparison.

This module factors out everything that is common to every algorithm:
  * the 2D map (walls / obstacles),
  * the ground-truth robot with an autonomous way-point controller,
  * the sensor models (odometry, compass, range-to-beacon),
  * geometry helpers.

Every localization filter receives EXACTLY the same control input
(odometry) and the same measurements at every time step, so the
comparison between algorithms is fair.

State convention everywhere:  x = [px, py, theta]  with theta in [-pi, pi].
Control input:                u = (v, w)            (linear, angular velocity)
Measurement (dict):
    {"beacon_ranges": np.array[B],   # noisy range to each known beacon
     "compass":       float}         # noisy absolute heading
"""

from dataclasses import dataclass, field

import numpy as np
from matplotlib.patches import Rectangle


# =====================================================================
# ---------------------------- PARAMETERS -----------------------------
# =====================================================================
DT = 0.1                       # simulation / filter time step [s]

MIN_X, MAX_X = 0.0, 45.0       # map bounds [m]
MIN_Y, MAX_Y = 0.0, 25.0

# Grid for the histogram filter (1 m cells -> fair accuracy, still fast)
GRID_RESOLUTION = 1.0
THETA_BINS = 36

# Sensor noise (standard deviations)
ODOM_V_STD = 0.05              # linear velocity odometry noise [m/s]
ODOM_W_STD = 0.02              # angular velocity odometry noise [rad/s]
COMPASS_STD = 0.10             # absolute heading noise [rad]
BEACON_STD = 0.30              # range-to-beacon noise [m]

# True process noise actually applied to the robot
PROC_V_STD = 0.02
PROC_W_STD = 0.01

ROBOT_RADIUS = 0.5

WALL_COLOR = "#d9822b"
WALL_LW = 2.0


# =====================================================================
# ------------------------- OBSTACLE OBJECTS --------------------------
# =====================================================================
@dataclass(frozen=True)
class SegmentObstacle:
    x1: float
    y1: float
    x2: float
    y2: float

    def walls(self):
        return [(self.x1, self.y1, self.x2, self.y2)]

    def draw(self, ax, color=WALL_COLOR, lw=WALL_LW):
        ax.plot([self.x1, self.x2], [self.y1, self.y2], color=color, linewidth=lw)


@dataclass(frozen=True)
class RectObstacle:
    x: float
    y: float
    w: float
    h: float

    def walls(self):
        x1, y1 = self.x, self.y
        x2, y2 = self.x + self.w, self.y + self.h
        return [
            (x1, y1, x2, y1),
            (x2, y1, x2, y2),
            (x2, y2, x1, y2),
            (x1, y2, x1, y1),
        ]

    def draw(self, ax, color=WALL_COLOR, lw=WALL_LW):
        ax.add_patch(Rectangle((self.x, self.y), self.w, self.h,
                               facecolor="none", edgecolor=color, linewidth=lw))


def _default_obstacles():
    """The same office-like map used by the original histogram demo."""
    obstacles = [
        # Outer scope
        SegmentObstacle(2, 2, 44, 2),
        SegmentObstacle(2, 18, 44, 18),
        # Left border (with doorway opening at y in [8.5, 11.5])
        SegmentObstacle(2, 2, 2, 8.5),
        SegmentObstacle(2, 11.5, 2, 18),
        SegmentObstacle(0, 8.5, 2, 8.5),
        SegmentObstacle(0, 11.5, 2, 11.5),
        # Right border
        SegmentObstacle(44, 2, 44, 8.5),
        SegmentObstacle(44, 11.5, 44, 18),
        SegmentObstacle(45, 8.5, 44, 8.5),
        SegmentObstacle(45, 11.5, 44, 11.5),
        # Room A
        SegmentObstacle(5, 6.5, 10, 6.5),
        SegmentObstacle(5, 8.5, 10, 8.5),
        SegmentObstacle(5, 6.5, 5, 8.5),
        SegmentObstacle(10, 6.5, 10, 7),
        SegmentObstacle(10, 8, 10, 8.5),
        SegmentObstacle(10, 8, 13, 8),
        SegmentObstacle(10, 7, 13, 7),
        SegmentObstacle(13, 7, 13, 8),
        # Room B
        SegmentObstacle(5, 13.5, 10, 13.5),
        SegmentObstacle(5, 11.5, 10, 11.5),
        SegmentObstacle(5, 11.5, 5, 13.5),
        SegmentObstacle(10, 11.5, 10, 12),
        SegmentObstacle(10, 13.5, 10, 13),
        SegmentObstacle(10, 12, 13, 12),
        SegmentObstacle(10, 13, 13, 13),
        SegmentObstacle(13, 12, 13, 13),
        # Middle blocks
        RectObstacle(15, 12.5, 5, 2.5),
        RectObstacle(15, 5, 5, 2.5),
        RectObstacle(15, 9.5, 3, 1),
        # Room C
        SegmentObstacle(35, 5, 35, 15),
        SegmentObstacle(23, 15, 35, 15),
        SegmentObstacle(23, 12.5, 25, 12.5),
        SegmentObstacle(23, 12.5, 23, 15),
        SegmentObstacle(25, 7.5, 25, 12.5),
        SegmentObstacle(23, 7.5, 25, 7.5),
        SegmentObstacle(23, 5, 35, 5),
        SegmentObstacle(23, 5, 23, 7.5),
        # Two square pillars
        RectObstacle(27.25, 8.75, 1.5, 1.5),
        RectObstacle(29.25, 8.75, 1.5, 1.5),
    ]
    return obstacles


# =====================================================================
# ------------------------- GEOMETRY HELPERS --------------------------
# =====================================================================
def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _segments_intersect(p1, p2, p3, p4):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)
            and ccw(p1, p2, p3) != ccw(p1, p2, p4))


def will_collide(x_old, y_old, x_new, y_new, walls):
    for wx1, wy1, wx2, wy2 in walls:
        if _segments_intersect((x_old, y_old), (x_new, y_new),
                               (wx1, wy1), (wx2, wy2)):
            return True
    return False


def angle_mean(angles, weights=None):
    """Circular mean of a set of angles (optionally weighted)."""
    angles = np.asarray(angles, dtype=float)
    if weights is None:
        s, c = np.sin(angles).sum(), np.cos(angles).sum()
    else:
        weights = np.asarray(weights, dtype=float)
        s, c = (weights * np.sin(angles)).sum(), (weights * np.cos(angles)).sum()
    return np.arctan2(s, c)


# =====================================================================
# ---------------------------- ENVIRONMENT ----------------------------
# =====================================================================
@dataclass
class Environment:
    """Holds the static world: map, beacons, grid definition, noise model."""
    obstacles: list = field(default_factory=_default_obstacles)
    beacons: np.ndarray = field(default_factory=lambda: np.array([
        [4.0, 3.0],
        [38.0, 3.0],
        [38.0, 17.0],
        [4.0, 17.0],
    ], dtype=float))

    def __post_init__(self):
        self.walls = [w for obs in self.obstacles for w in obs.walls()]
        self.n_beacons = len(self.beacons)

        # Histogram-filter grid definition (shared so we can size the belief)
        self.x_edges = np.arange(MIN_X, MAX_X + GRID_RESOLUTION, GRID_RESOLUTION)
        self.y_edges = np.arange(MIN_Y, MAX_Y + GRID_RESOLUTION, GRID_RESOLUTION)
        self.x_centers = self.x_edges[:-1] + 0.5 * GRID_RESOLUTION
        self.y_centers = self.y_edges[:-1] + 0.5 * GRID_RESOLUTION
        self.Nx = len(self.x_centers)
        self.Ny = len(self.y_centers)
        self.theta_centers = np.arange(THETA_BINS) * (2 * np.pi / THETA_BINS) - np.pi

    # --- sensor model used both by the simulator and by the filters ---
    def true_beacon_ranges(self, x, y):
        """Noise-free range from (x, y) to every beacon."""
        return np.hypot(self.beacons[:, 0] - x, self.beacons[:, 1] - y)

    def draw_map(self, ax, color=WALL_COLOR, lw=WALL_LW):
        for obs in self.obstacles:
            obs.draw(ax, color=color, lw=lw)

    def draw_beacons(self, ax):
        ax.plot(self.beacons[:, 0], self.beacons[:, 1], "s",
                color="#7b2d8e", markersize=7, label="Beacons")


# =====================================================================
# ----------------------- GROUND-TRUTH ROBOT --------------------------
# =====================================================================
class RobotSimulator:
    """
    Autonomous robot that follows a rectangular patrol loop through the
    open corridor of the map. At every step it returns:
        u     : noisy odometry control (v, w) given to the filters
        meas  : noisy measurements (beacon ranges + compass)
        truth : the true pose (px, py, theta) for error evaluation
    """

    def __init__(self, env: Environment, seed: int = 0,
                 v_max: float = 2.0, w_max: float = 2.5):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.v_max = v_max
        self.w_max = w_max

        # Start pose and the open-space patrol loop (avoids all obstacles)
        self.x, self.y, self.theta = 3.5, 4.0, np.pi / 2
        self.waypoints = [
            (3.5, 16.5),
            (14.0, 16.5),
            (14.0, 4.0),
            (3.5, 4.0),
        ]
        self.wp_idx = 0

    @property
    def pose(self):
        return (self.x, self.y, self.theta)

    def _control(self):
        """Simple turn-toward-then-drive way-point controller."""
        tx, ty = self.waypoints[self.wp_idx]
        dist = np.hypot(tx - self.x, ty - self.y)
        if dist < 0.6:                       # reached -> advance (loop)
            self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
            tx, ty = self.waypoints[self.wp_idx]
            dist = np.hypot(tx - self.x, ty - self.y)

        desired = np.arctan2(ty - self.y, tx - self.x)
        heading_err = wrap_angle(desired - self.theta)

        w = float(np.clip(2.0 * heading_err, -self.w_max, self.w_max))
        # slow down while turning sharply and when approaching a way-point
        v = self.v_max * max(0.0, np.cos(heading_err))
        v = min(v, 0.8 + dist)
        return v, w

    def step(self, dt=DT):
        v_cmd, w_cmd = self._control()

        # true motion with process noise
        v_true = v_cmd + self.rng.normal(0.0, PROC_V_STD)
        w_true = w_cmd + self.rng.normal(0.0, PROC_W_STD)
        x_new = self.x + dt * v_true * np.cos(self.theta)
        y_new = self.y + dt * v_true * np.sin(self.theta)

        if will_collide(self.x, self.y, x_new, y_new, self.env.walls):
            v_true = 0.0                     # blocked: rotate in place
            self.theta = wrap_angle(self.theta + dt * w_true)
        else:
            self.x, self.y = x_new, y_new
            self.theta = wrap_angle(self.theta + dt * w_true)

        # odometry control reported to the filters
        u = (v_true + self.rng.normal(0.0, ODOM_V_STD),
             w_true + self.rng.normal(0.0, ODOM_W_STD))

        # measurements
        ranges = (self.env.true_beacon_ranges(self.x, self.y)
                  + self.rng.normal(0.0, BEACON_STD, size=self.env.n_beacons))
        compass = wrap_angle(self.theta + self.rng.normal(0.0, COMPASS_STD))
        meas = {"beacon_ranges": np.maximum(0.05, ranges), "compass": compass}

        return u, meas, self.pose
