import copy
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from pynput import keyboard
from scipy.ndimage import gaussian_filter


# =========================
# ---------- PARAMETERS ---
# =========================
DT = 0.1
SIM_TIME = 1000.0

MIN_X, MAX_X = 0.0, 45.0
MIN_Y, MAX_Y = 0.0, 25.0
GRID_RESOLUTION = 2.0

# Histogram / pose grid
THETA_BINS = 36
THETA_RES = 2.0 * np.pi / THETA_BINS

# Motion uncertainty (for smoothing after shift)
MOTION_SIGMA_XY = 0.8      # cells (approximately)
MOTION_SIGMA_TH = 1.0      # theta bins (approximately)

# LiDAR
LIDAR_MAX_RANGE = 10.0
LIDAR_NUM_RAYS = 12
LIDAR_RANGE_STD = 0.4
LIDAR_MIN_RANGE = 0.1

# Odometry/IMU/Compass/Beacons
ODOM_V_STD = 0.05
ODOM_W_STD = 0.02
GYRO_STD = 0.02
COMPASS_STD = 0.12   # rad
BEACON_STD = 0.30    # meters

# Robot + collision
ROBOT_RADIUS = 0.5
AVOID_TURN_RATE = 0.15

# Keyboard commands (desired)
KEY_CMD = {"v": 0.0, "w": 0.0}


# =========================
# ---- OBSTACLE OBJECTS ---
# =========================
WALL_COLOR = "orange"
WALL_LW = 2


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


@dataclass(frozen=True)
class CubeObstacle:
    cx: float
    cy: float
    size: float  # side length (2D square “cube”)

    def as_rect(self):
        s = 0.5 * self.size
        return RectObstacle(self.cx - s, self.cy - s, self.size, self.size)

    def walls(self):
        return self.as_rect().walls()

    def draw(self, ax, color=WALL_COLOR, lw=WALL_LW):
        self.as_rect().draw(ax, color=color, lw=lw)


def build_walls(obstacles):
    walls = []
    for obs in obstacles:
        walls.extend(obs.walls())
    return walls


def draw_map(ax, obstacles):
    for obs in obstacles:
        obs.draw(ax)


# =========================
# ------- MAP SETUP -------
# =========================
OBSTACLES = [
    # Outer scope
    SegmentObstacle(2, 2, 44, 2),
    SegmentObstacle(2, 18, 44, 18),

    # Left border
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

    # Middle blocks (rectangles)
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
]

# Add extra “cube” (square) obstacles via deepcopy template
cube_template = {"cx": 28.0, "cy": 9.5, "size": 1.5}
cube2 = copy.deepcopy(cube_template)
cube2["cx"] = 30.0

OBSTACLES += [
    CubeObstacle(**cube_template),
    CubeObstacle(**cube2),
]

# Segment list used by collision + LiDAR
WALLS = build_walls(OBSTACLES)


# =========================
# ------ GEOMETRY ---------
# =========================
def segment_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    a = (x1, y1)
    b = (x2, y2)
    c = (x3, y3)
    d = (x4, y4)
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def will_collide(x_old, y_old, x_new, y_new):
    for wx1, wy1, wx2, wy2 in WALLS:
        if segment_intersection((x_old, y_old), (x_new, y_new), (wx1, wy1), (wx2, wy2)):
            return True
    return False


def line_intersection(ray_origin, ray_dir, wall):
    x1, y1, x2, y2 = wall
    rx, ry = ray_origin
    dx, dy = ray_dir

    wx = x2 - x1
    wy = y2 - y1

    denom = dx * wy - dy * wx
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - rx) * wy - (y1 - ry) * wx) / denom
    s = ((x1 - rx) * dy - (y1 - ry) * dx) / denom
    if t > 0 and 0 <= s <= 1:
        return t
    return None


def cast_ray(robot_x, robot_y, angle):
    dx = np.cos(angle)
    dy = np.sin(angle)
    min_dist = LIDAR_MAX_RANGE
    for wall in WALLS:
        dist = line_intersection((robot_x, robot_y), (dx, dy), wall)
        if dist is not None and dist < min_dist:
            min_dist = dist
    hit_x = robot_x + min_dist * dx
    hit_y = robot_y + min_dist * dy
    return min_dist, hit_x, hit_y


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def logsumexp(a):
    amax = np.max(a)
    return amax + np.log(np.sum(np.exp(a - amax)) + 1e-300)


def shift_2d_zero_fill(arr2d, sx, sy):
    """Shift [Nx, Ny] by sx,sy cells, filling with 0 instead of wrap."""
    nx, ny = arr2d.shape
    out = np.zeros_like(arr2d)
    x0 = max(0, sx)
    x1 = min(nx, nx + sx)
    y0 = max(0, sy)
    y1 = min(ny, ny + sy)
    out[x0:x1, y0:y1] = arr2d[x0 - sx:x1 - sx, y0 - sy:y1 - sy]
    return out


# =========================
# ---- MAIN LOCALIZER -----
# =========================
class GridHistogramSLAM:
    def __init__(self):
        # True state (simulation)
        self.robot_x = 1.0
        self.robot_y = 10.0
        self.robot_theta = 0.0

        self.running = True

        # Grid definition
        self.x_edges = np.arange(MIN_X, MAX_X + GRID_RESOLUTION, GRID_RESOLUTION)
        self.y_edges = np.arange(MIN_Y, MAX_Y + GRID_RESOLUTION, GRID_RESOLUTION)
        self.x_centers = self.x_edges[:-1] + 0.5 * GRID_RESOLUTION
        self.y_centers = self.y_edges[:-1] + 0.5 * GRID_RESOLUTION
        self.Nx = len(self.x_centers)
        self.Ny = len(self.y_centers)

        self.theta_centers = np.arange(THETA_BINS) * THETA_RES - np.pi

        # Belief in log domain: [T, Nx, Ny]
        total = THETA_BINS * self.Nx * self.Ny
        self.log_bel = np.full((THETA_BINS, self.Nx, self.Ny), -np.log(total), dtype=float)

        # LiDAR ray angles (relative to robot heading)
        self.lidar_rel = np.linspace(-np.pi, np.pi, LIDAR_NUM_RAYS, endpoint=False)

        # Beacons (UWB anchors)
        self.beacons = np.array([
            [4.0, 3.0],
            [38.0, 3.0],
            [38.0, 17.0],
            [4.0, 17.0],
        ], dtype=float)

        # Precompute expected sensors
        self.expected_lidar = None     # [T, R, Nx, Ny]
        self.expected_beacon = None    # [B, Nx, Ny]
        self._precompute_expected_measurements()

        # Custom colormap: blue -> red
        self.blue_red_cmap = LinearSegmentedColormap.from_list(
            "blue_red", ["blue", "red"], N=256
        )

    def _precompute_expected_measurements(self):
        # Expected beacon ranges (doesn't depend on theta)
        Xc, Yc = np.meshgrid(self.x_centers, self.y_centers, indexing="ij")  # [Nx,Ny]
        exp_b = []
        for bx, by in self.beacons:
            exp_b.append(np.hypot(Xc - bx, Yc - by))
        self.expected_beacon = np.stack(exp_b, axis=0)  # [B,Nx,Ny]

        # Expected LiDAR for every (theta_bin, ray, x, y)
        self.expected_lidar = np.zeros((THETA_BINS, LIDAR_NUM_RAYS, self.Nx, self.Ny), dtype=float)
        for it, th in enumerate(self.theta_centers):
            for ir, rel in enumerate(self.lidar_rel):
                ang = th + rel
                for ix, x in enumerate(self.x_centers):
                    for iy, y in enumerate(self.y_centers):
                        d, _, _ = cast_ray(x, y, ang)
                        self.expected_lidar[it, ir, ix, iy] = d

    # -------------------------
    # SIM: motion + sensors
    # -------------------------
    def _apply_true_motion(self, v_cmd, w_cmd):
        # Process noise on actual motion
        v_true = v_cmd + np.random.randn() * 0.03
        w_true = w_cmd + np.random.randn() * 0.01

        x_new = self.robot_x + DT * np.cos(self.robot_theta) * v_true
        y_new = self.robot_y + DT * np.sin(self.robot_theta) * v_true

        if will_collide(self.robot_x, self.robot_y, x_new, y_new):
            v_true = 0.0
            turn_sign = np.sign(w_true) if abs(w_true) > 1e-6 else 1.0
            self.robot_theta = wrap_angle(self.robot_theta + turn_sign * AVOID_TURN_RATE)
        else:
            self.robot_x = x_new
            self.robot_y = y_new
            self.robot_theta = wrap_angle(self.robot_theta + DT * w_true)

        return v_true, w_true

    def _lidar_scan(self):
        ranges = np.zeros(LIDAR_NUM_RAYS, dtype=float)
        hits = []
        for i, rel in enumerate(self.lidar_rel):
            ang = self.robot_theta + rel
            d, hx, hy = cast_ray(self.robot_x, self.robot_y, ang)
            d_noisy = d + np.random.randn() * LIDAR_RANGE_STD
            d_noisy = float(np.clip(d_noisy, LIDAR_MIN_RANGE, LIDAR_MAX_RANGE))
            ranges[i] = d_noisy
            hits.append((hx, hy))
        return ranges, hits

    def _beacon_ranges(self):
        d = np.hypot(self.beacons[:, 0] - self.robot_x, self.beacons[:, 1] - self.robot_y)
        d = d + np.random.randn(len(self.beacons)) * BEACON_STD
        return np.maximum(0.05, d)

    def _odometry(self, v_true, w_true):
        v_odom = v_true + np.random.randn() * ODOM_V_STD
        w_odom = w_true + np.random.randn() * ODOM_W_STD
        return v_odom, w_odom

    def _gyro(self, w_true):
        return w_true + np.random.randn() * GYRO_STD

    def _compass(self):
        return wrap_angle(self.robot_theta + np.random.randn() * COMPASS_STD)

    # -------------------------
    # FILTER: motion update
    # -------------------------
    def motion_update(self, v_odom, w_odom):
        dth = DT * w_odom
        ds = DT * v_odom

        bel = np.exp(self.log_bel)

        # 1) rotate in theta bins (shift + smooth)
        th_shift = int(np.round(dth / THETA_RES))
        bel_rot = np.roll(bel, shift=th_shift, axis=0)

        # smooth across theta bins a bit (uncertain rotation)
        bel_rot = gaussian_filter(
            bel_rot, sigma=(MOTION_SIGMA_TH, 0.0, 0.0), mode="constant", cval=0.0
        )

        # 2) translate in x,y depending on theta
        bel_tr = np.zeros_like(bel_rot)
        for it, th in enumerate(self.theta_centers):
            dx = ds * np.cos(th)
            dy = ds * np.sin(th)
            sx = int(np.round(dx / GRID_RESOLUTION))
            sy = int(np.round(dy / GRID_RESOLUTION))
            bel_tr[it] = shift_2d_zero_fill(bel_rot[it], sx, sy)

        # 3) smooth translation uncertainty in x,y
        bel_tr = gaussian_filter(
            bel_tr, sigma=(0.0, MOTION_SIGMA_XY, MOTION_SIGMA_XY), mode="constant", cval=0.0
        )

        s = np.sum(bel_tr)
        if s <= 0:
            total = THETA_BINS * self.Nx * self.Ny
            self.log_bel[:] = -np.log(total)
            return

        bel_tr /= s
        self.log_bel = np.log(bel_tr + 1e-300)

    # -------------------------
    # FILTER: measurement update
    # -------------------------
    def observation_update(self, lidar_ranges, compass_theta, beacon_ranges):
        # Start from prior
        log_post = self.log_bel.copy()

        # LiDAR log-likelihood (vectorized)
        diff = lidar_ranges[None, :, None, None] - self.expected_lidar
        log_c = -np.log(LIDAR_RANGE_STD * np.sqrt(2.0 * np.pi))
        log_lidar = log_c - 0.5 * (diff / LIDAR_RANGE_STD) ** 2
        log_post += np.sum(log_lidar, axis=1)  # sum over rays -> [T,Nx,Ny]

        # Compass likelihood over theta (broadcast)
        th_diff = wrap_angle(compass_theta - self.theta_centers)  # [T]
        log_c_th = -np.log(COMPASS_STD * np.sqrt(2.0 * np.pi))
        log_th = log_c_th - 0.5 * (th_diff / COMPASS_STD) ** 2
        log_post += log_th[:, None, None]

        # Beacons likelihood (vectorized over x,y; same for every theta)
        bdiff = beacon_ranges[:, None, None] - self.expected_beacon
        log_c_b = -np.log(BEACON_STD * np.sqrt(2.0 * np.pi))
        log_b = log_c_b - 0.5 * (bdiff / BEACON_STD) ** 2
        log_post += np.sum(log_b, axis=0)[None, :, :]

        # Normalize in log space
        lse = logsumexp(log_post)
        self.log_bel = log_post - lse

    def estimate_pose(self):
        idx = np.unravel_index(np.argmax(self.log_bel), self.log_bel.shape)
        it, ix, iy = idx
        return float(self.x_centers[ix]), float(self.y_centers[iy]), float(self.theta_centers[it])

    # -------------------------
    # Keyboard
    # -------------------------
    def on_press(self, key):
        try:
            if key == keyboard.Key.up:
                KEY_CMD["v"] += 0.2
            elif key == keyboard.Key.down:
                KEY_CMD["v"] -= 0.2
            elif key == keyboard.Key.left:
                KEY_CMD["w"] += 0.05
            elif key == keyboard.Key.right:
                KEY_CMD["w"] -= 0.05
            elif hasattr(key, "char") and key.char == " ":
                KEY_CMD["v"] = 0.0
                KEY_CMD["w"] = 0.0
        except Exception:
            pass

    def on_release(self, key):
        if key == keyboard.Key.esc:
            self.running = False
            return False
        return True

    # -------------------------
    # Visualization helpers
    # -------------------------
    def draw_lidar(self, ax, hits):
        for (hx, hy) in hits:
            ax.plot([self.robot_x, hx], [self.robot_y, hy], "g-", alpha=0.25, linewidth=0.6)
            ax.plot(hx, hy, "go", markersize=2)

    def draw_beacons(self, ax):
        ax.plot(self.beacons[:, 0], self.beacons[:, 1], "ms", markersize=6, label="Beacons")

    def draw_heat_map(self, ax):
        bel = np.exp(self.log_bel)
        marginal = np.sum(bel, axis=0)  # [Nx,Ny]
        maxv = float(np.max(marginal))
        if maxv <= 0:
            maxv = 1.0

        ax.pcolormesh(
            self.x_edges,
            self.y_edges,
            marginal.T,
            cmap=self.blue_red_cmap,   # blue -> red
            vmin=0.0,
            vmax=maxv,
            shading="auto",
        )
        ax.set_xlim(MIN_X, MAX_X)
        ax.set_ylim(MIN_Y, MAX_Y)
        ax.set_aspect("equal")

    # -------------------------
    # Main loop
    # -------------------------
    def main(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        t = 0.0
        txt_time = fig.text(0.12, 0.92, "", ha="left", va="top", fontsize=10)
        txt_cmd = fig.text(0.12, 0.89, "", ha="left", va="top", fontsize=10)
        txt_est = fig.text(0.12, 0.86, "", ha="left", va="top", fontsize=10)

        while self.running and t <= SIM_TIME:
            t += DT

            # commands
            v_cmd = float(KEY_CMD["v"])
            w_cmd = float(KEY_CMD["w"])

            # true motion + sensors
            v_true, w_true = self._apply_true_motion(v_cmd, w_cmd)
            lidar_ranges, hits = self._lidar_scan()
            beacon_ranges = self._beacon_ranges()
            compass_theta = self._compass()
            v_odom, w_odom = self._odometry(v_true, w_true)
            _gyro_w = self._gyro(w_true)  # available if you want to fuse separately

            # filter
            self.motion_update(v_odom, w_odom)
            self.observation_update(lidar_ranges, compass_theta, beacon_ranges)

            # estimate
            ex, ey, eth = self.estimate_pose()

            # draw
            ax1.cla()
            ax2.cla()

            ax1.set_xlim(MIN_X, MAX_X)
            ax1.set_ylim(MIN_Y, MAX_Y)
            ax1.set_aspect("equal")
            draw_map(ax1, OBSTACLES)
            self.draw_beacons(ax1)
            self.draw_lidar(ax1, hits)

            # true robot
            ax1.add_patch(
                Circle((self.robot_x, self.robot_y), radius=ROBOT_RADIUS,
                       facecolor="blue", edgecolor="black")
            )
            ax1.arrow(
                self.robot_x, self.robot_y,
                np.cos(self.robot_theta), np.sin(self.robot_theta),
                head_width=0.3, head_length=0.3,
                fc="blue", ec="blue", length_includes_head=True
            )

            # estimated robot pose
            ax1.add_patch(
                Circle((ex, ey), radius=ROBOT_RADIUS,
                       facecolor="none", edgecolor="red", linewidth=2)
            )
            ax1.arrow(
                ex, ey,
                np.cos(eth), np.sin(eth),
                head_width=0.3, head_length=0.3,
                fc="red", ec="red", length_includes_head=True
            )

            ax1.set_title("1")
            self.draw_heat_map(ax2)
            ax2.set_title("2")

            txt_time.set_text(f"Time: {t:.1f}s   (ESC to quit)")
            txt_cmd.set_text(f"Cmd: v={v_cmd:.2f} m/s, w={w_cmd:.2f} rad/s   (SPACE to stop)")
            txt_est.set_text(f"Est: x={ex:.2f}, y={ey:.2f}, th={eth:.2f} rad")

            plt.pause(0.001)

        plt.show()


if __name__ == "__main__":
    sim = GridHistogramSLAM()
    sim.main()
