"""
Unscented Kalman Filter localization.

Like the EKF the belief is a single Gaussian, but instead of linearizing
the models with Jacobians the UKF propagates a deterministic set of
"sigma" points through the *exact* non-linear functions and recovers the
posterior mean/covariance from them (the unscented transform).

Care is taken with the circular heading dimension: means are computed
with a circular average and all residuals involving an angle are wrapped.

Reference: Thrun, Burgard, Fox - "Probabilistic Robotics", ch. 3.4.
"""

import numpy as np

from environment import (wrap_angle, angle_mean, ODOM_V_STD, ODOM_W_STD,
                         COMPASS_STD, BEACON_STD)
from .base import BaseFilter


class UKFLocalization(BaseFilter):
    name = "UKF"
    color = "#9467bd"

    def __init__(self, env, init_pose, init_cov):
        super().__init__(env, init_pose, init_cov)
        self.mu = self.init_pose.copy()
        self.Sigma = self.init_cov.copy()

        self.n = 3
        self.alpha, self.beta, self.kappa = 1e-3, 2.0, 0.0
        self.lam = self.alpha ** 2 * (self.n + self.kappa) - self.n

        # sigma-point weights
        c = self.n + self.lam
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2.0 * c))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / c
        self.Wc[0] = self.lam / c + (1.0 - self.alpha ** 2 + self.beta)

        self.Q = np.diag([0.05, 0.05, 0.02]) ** 2     # process noise

    # ------------------------- sigma points ----------------------------
    def _sigma_points(self, mu, Sigma):
        n = self.n
        Sigma = 0.5 * (Sigma + Sigma.T) + 1e-9 * np.eye(n)   # symmetrize
        try:
            L = np.linalg.cholesky((n + self.lam) * Sigma)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky((n + self.lam) * Sigma + 1e-6 * np.eye(n))
        pts = np.zeros((2 * n + 1, n))
        pts[0] = mu
        for i in range(n):
            pts[i + 1] = mu + L[:, i]
            pts[n + i + 1] = mu - L[:, i]
        return pts

    @staticmethod
    def _state_mean(pts, Wm):
        mean = np.zeros(3)
        mean[0] = np.dot(Wm, pts[:, 0])
        mean[1] = np.dot(Wm, pts[:, 1])
        mean[2] = angle_mean(pts[:, 2], Wm)
        return mean

    @staticmethod
    def _state_resid(pts, mean):
        d = pts - mean
        d[:, 2] = wrap_angle(d[:, 2])
        return d

    # ----------------------------- predict -----------------------------
    def predict(self, u, dt):
        v, w = u
        pts = self._sigma_points(self.mu, self.Sigma)

        prop = np.zeros_like(pts)
        for i, (px, py, th) in enumerate(pts):
            prop[i] = [px + dt * v * np.cos(th),
                       py + dt * v * np.sin(th),
                       wrap_angle(th + dt * w)]

        self.mu = self._state_mean(prop, self.Wm)
        d = self._state_resid(prop, self.mu)
        self.Sigma = (d.T * self.Wc) @ d + self.Q
        self._sigma_cache = prop

    # ----------------------------- update ------------------------------
    def update(self, meas):
        ranges = meas["beacon_ranges"]
        compass = meas["compass"]
        beacons = self.env.beacons
        B = len(beacons)
        m = B + 1                                   # measurement dimension

        pts = self._sigma_points(self.mu, self.Sigma)

        # transform sigma points through the measurement model
        Z = np.zeros((pts.shape[0], m))
        for i, (px, py, th) in enumerate(pts):
            Z[i, :B] = np.hypot(px - beacons[:, 0], py - beacons[:, 1])
            Z[i, B] = th

        # predicted measurement mean (compass component is circular)
        z_mean = np.zeros(m)
        z_mean[:B] = self.Wm @ Z[:, :B]
        z_mean[B] = angle_mean(Z[:, B], self.Wm)

        dz = Z - z_mean
        dz[:, B] = wrap_angle(dz[:, B])
        dx = self._state_resid(pts, self.mu)

        R = np.diag([BEACON_STD ** 2] * B + [COMPASS_STD ** 2])
        S = (dz.T * self.Wc) @ dz + R
        Pxz = (dx.T * self.Wc) @ dz
        K = Pxz @ np.linalg.inv(S)

        z = np.concatenate([ranges, [compass]])
        innovation = z - z_mean
        innovation[B] = wrap_angle(innovation[B])

        self.mu = self.mu + K @ innovation
        self.mu[2] = wrap_angle(self.mu[2])
        self.Sigma = self.Sigma - K @ S @ K.T

    # --------------------------- accessors -----------------------------
    def estimate(self):
        return float(self.mu[0]), float(self.mu[1]), float(self.mu[2])

    def cov_xy(self):
        return self.Sigma[:2, :2].copy()
