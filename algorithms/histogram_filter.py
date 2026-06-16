"""
Grid Histogram filter localization (discrete Markov localization).

A *non-parametric* Bayes filter whose belief is a discrete probability
distribution over a 3-D grid of poses [theta_bin, x_cell, y_cell].
This is the algorithm the original `histogram.py` implemented; here it
is refactored to the shared interface and to the common measurement set
(beacon ranges + compass) so it can be compared fairly.

Reference: Thrun, Burgard, Fox - "Probabilistic Robotics", ch. 8.4.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from environment import (wrap_angle, GRID_RESOLUTION, THETA_BINS,
                         COMPASS_STD, BEACON_STD)
from .base import BaseFilter


def _shift_2d_zero_fill(arr, sx, sy):
    """Shift a 2-D array by (sx, sy) cells, filling vacated cells with 0."""
    nx, ny = arr.shape
    out = np.zeros_like(arr)
    x0, x1 = max(0, sx), min(nx, nx + sx)
    y0, y1 = max(0, sy), min(ny, ny + sy)
    out[x0:x1, y0:y1] = arr[x0 - sx:x1 - sx, y0 - sy:y1 - sy]
    return out


class HistogramFilter(BaseFilter):
    name = "Histogram"
    color = "#ff7f0e"

    MOTION_SIGMA_XY = 1.0      # diffusion after a translation step [cells]
    MOTION_SIGMA_TH = 1.0      # diffusion after a rotation step [bins]

    def __init__(self, env, init_pose, init_cov):
        super().__init__(env, init_pose, init_cov)
        self.x_centers = env.x_centers
        self.y_centers = env.y_centers
        self.theta_centers = env.theta_centers
        self.Nx, self.Ny = env.Nx, env.Ny

        # expected beacon range for every (x, y) cell -> [B, Nx, Ny]
        Xc, Yc = np.meshgrid(self.x_centers, self.y_centers, indexing="ij")
        self.exp_beacon = np.stack(
            [np.hypot(Xc - bx, Yc - by) for bx, by in env.beacons], axis=0)

        # initial belief: Gaussian bump around the shared initial guess
        self.bel = self._gaussian_belief(init_pose, init_cov)

    def _gaussian_belief(self, pose, cov):
        sx, sy = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
        sth = np.sqrt(cov[2, 2])
        gx = np.exp(-0.5 * ((self.x_centers - pose[0]) / sx) ** 2)
        gy = np.exp(-0.5 * ((self.y_centers - pose[1]) / sy) ** 2)
        dth = wrap_angle(self.theta_centers - pose[2])
        gth = np.exp(-0.5 * (dth / sth) ** 2)
        bel = gth[:, None, None] * gx[None, :, None] * gy[None, None, :]
        return bel / bel.sum()

    # ----------------------------- predict -----------------------------
    def predict(self, u, dt):
        v, w = u
        dth, ds = dt * w, dt * v
        bel = self.bel

        # 1) rotate over theta bins, then diffuse the heading uncertainty
        th_shift = int(np.round(dth / (2 * np.pi / THETA_BINS)))
        bel = np.roll(bel, shift=th_shift, axis=0)
        bel = gaussian_filter(bel, sigma=(self.MOTION_SIGMA_TH, 0, 0),
                              mode="constant", cval=0.0)

        # 2) translate x,y per heading bin, then diffuse position uncertainty
        out = np.zeros_like(bel)
        for it, th in enumerate(self.theta_centers):
            sx = int(np.round(ds * np.cos(th) / GRID_RESOLUTION))
            sy = int(np.round(ds * np.sin(th) / GRID_RESOLUTION))
            out[it] = _shift_2d_zero_fill(bel[it], sx, sy)
        out = gaussian_filter(out, sigma=(0, self.MOTION_SIGMA_XY,
                                          self.MOTION_SIGMA_XY),
                              mode="constant", cval=0.0)

        s = out.sum()
        self.bel = (np.full_like(out, 1.0 / out.size) if s <= 0 else out / s)

    # ----------------------------- update ------------------------------
    def update(self, meas):
        ranges = meas["beacon_ranges"]
        compass = meas["compass"]

        log_post = np.log(self.bel + 1e-300)

        # beacon range likelihood (same for every theta bin)
        bdiff = ranges[:, None, None] - self.exp_beacon          # [B,Nx,Ny]
        log_b = -0.5 * (bdiff / BEACON_STD) ** 2
        log_post += np.sum(log_b, axis=0)[None, :, :]

        # compass likelihood over theta bins
        dth = wrap_angle(compass - self.theta_centers)           # [T]
        log_th = -0.5 * (dth / COMPASS_STD) ** 2
        log_post += log_th[:, None, None]

        log_post -= log_post.max()
        bel = np.exp(log_post)
        self.bel = bel / bel.sum()

    # --------------------------- accessors -----------------------------
    def estimate(self):
        marg_xy = self.bel.sum(axis=0)                # [Nx, Ny]
        px = np.average(self.x_centers, weights=marg_xy.sum(axis=1))
        py = np.average(self.y_centers, weights=marg_xy.sum(axis=0))
        marg_th = self.bel.sum(axis=(1, 2))           # [T]
        s = np.sum(marg_th * np.sin(self.theta_centers))
        c = np.sum(marg_th * np.cos(self.theta_centers))
        return float(px), float(py), float(np.arctan2(s, c))

    def belief_grid(self):
        return self.bel.sum(axis=0)                   # marginal over theta
