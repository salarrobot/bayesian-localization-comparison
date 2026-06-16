"""
Extended Kalman Filter localization.

A *parametric* Bayes filter: the belief is a single Gaussian N(mu, Sigma).
The non-linear velocity motion model and the range/compass measurement
models are linearized with first-order Jacobians.

Reference: Thrun, Burgard, Fox - "Probabilistic Robotics", ch. 7.
"""

import numpy as np

from environment import (wrap_angle, ODOM_V_STD, ODOM_W_STD,
                         COMPASS_STD, BEACON_STD)
from .base import BaseFilter


class EKFLocalization(BaseFilter):
    name = "EKF"
    color = "#d62728"

    def __init__(self, env, init_pose, init_cov):
        super().__init__(env, init_pose, init_cov)
        self.mu = self.init_pose.copy()
        self.Sigma = self.init_cov.copy()
        # small additive process noise for numerical stability
        self.R_extra = np.diag([0.01, 0.01, 0.005]) ** 2

    # ----------------------------- predict -----------------------------
    def predict(self, u, dt):
        v, w = u
        th = self.mu[2]

        # velocity motion model g(mu, u)
        self.mu = self.mu + np.array([dt * v * np.cos(th),
                                      dt * v * np.sin(th),
                                      dt * w])
        self.mu[2] = wrap_angle(self.mu[2])

        # Jacobian of g w.r.t. state
        G = np.array([
            [1.0, 0.0, -dt * v * np.sin(th)],
            [0.0, 1.0,  dt * v * np.cos(th)],
            [0.0, 0.0, 1.0],
        ])
        # Jacobian of g w.r.t. control (maps odometry noise into state)
        V = np.array([
            [dt * np.cos(th), 0.0],
            [dt * np.sin(th), 0.0],
            [0.0, dt],
        ])
        M = np.diag([ODOM_V_STD ** 2, ODOM_W_STD ** 2])

        self.Sigma = G @ self.Sigma @ G.T + V @ M @ V.T + self.R_extra

    # ----------------------------- update ------------------------------
    def update(self, meas):
        ranges = meas["beacon_ranges"]
        compass = meas["compass"]
        beacons = self.env.beacons
        B = len(beacons)

        # stacked measurement: [B beacon ranges, 1 compass]
        z = np.concatenate([ranges, [compass]])
        h = np.zeros(B + 1)
        H = np.zeros((B + 1, 3))

        px, py, th = self.mu
        for i, (bx, by) in enumerate(beacons):
            dx, dy = px - bx, py - by
            r = max(1e-3, np.hypot(dx, dy))
            h[i] = r
            H[i] = [dx / r, dy / r, 0.0]
        h[B] = th
        H[B] = [0.0, 0.0, 1.0]

        innovation = z - h
        innovation[B] = wrap_angle(innovation[B])      # heading residual

        R = np.diag([BEACON_STD ** 2] * B + [COMPASS_STD ** 2])
        S = H @ self.Sigma @ H.T + R
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        self.mu = self.mu + K @ innovation
        self.mu[2] = wrap_angle(self.mu[2])
        self.Sigma = (np.eye(3) - K @ H) @ self.Sigma

    # --------------------------- accessors -----------------------------
    def estimate(self):
        return float(self.mu[0]), float(self.mu[1]), float(self.mu[2])

    def cov_xy(self):
        return self.Sigma[:2, :2].copy()
