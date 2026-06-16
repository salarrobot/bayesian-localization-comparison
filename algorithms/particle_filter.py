"""
Particle Filter localization (Monte Carlo Localization, MCL).

A *non-parametric* Bayes filter: the belief is represented by a set of
weighted samples (particles). It can represent arbitrary, multi-modal
distributions. Each step:
    predict  -- push every particle through the noisy motion model
    update   -- weight particles by measurement likelihood
    resample -- (low-variance / systematic) when the cloud degenerates

Reference: Thrun, Burgard, Fox - "Probabilistic Robotics", ch. 8.
"""

import numpy as np

from environment import wrap_angle, angle_mean, COMPASS_STD, BEACON_STD
from .base import BaseFilter


class ParticleFilter(BaseFilter):
    name = "Particle Filter"
    color = "#2ca02c"

    def __init__(self, env, init_pose, init_cov, n_particles=600, seed=1):
        super().__init__(env, init_pose, init_cov)
        self.N = n_particles
        self.rng = np.random.default_rng(seed)

        # sample the initial cloud from the shared initial Gaussian
        std = np.sqrt(np.diag(self.init_cov))
        self.particles = self.init_pose + self.rng.normal(
            0.0, std, size=(self.N, 3))
        self.particles[:, 2] = wrap_angle(self.particles[:, 2])
        self.weights = np.full(self.N, 1.0 / self.N)

        # motion noise applied to particles (a bit larger than odometry
        # noise to avoid sample impoverishment)
        self.mv, self.mw = 0.10, 0.05

    # ----------------------------- predict -----------------------------
    def predict(self, u, dt):
        v, w = u
        vs = v + self.rng.normal(0.0, self.mv, self.N)
        ws = w + self.rng.normal(0.0, self.mw, self.N)
        th = self.particles[:, 2]
        self.particles[:, 0] += dt * vs * np.cos(th)
        self.particles[:, 1] += dt * vs * np.sin(th)
        self.particles[:, 2] = wrap_angle(th + dt * ws)

    # ----------------------------- update ------------------------------
    def update(self, meas):
        ranges = meas["beacon_ranges"]
        compass = meas["compass"]
        beacons = self.env.beacons

        px = self.particles[:, 0][:, None]
        py = self.particles[:, 1][:, None]
        exp_r = np.hypot(px - beacons[:, 0], py - beacons[:, 1])  # [N, B]

        # log-likelihood (sum over beacons) + compass term
        ll = -0.5 * np.sum(((exp_r - ranges) / BEACON_STD) ** 2, axis=1)
        dth = wrap_angle(self.particles[:, 2] - compass)
        ll += -0.5 * (dth / COMPASS_STD) ** 2

        ll -= ll.max()                       # stabilize before exp
        w = self.weights * np.exp(ll)
        s = w.sum()
        self.weights = (np.full(self.N, 1.0 / self.N)
                        if s <= 0 else w / s)

        # resample only when the effective sample size gets small
        neff = 1.0 / np.sum(self.weights ** 2)
        if neff < self.N / 2.0:
            self._resample()

    def _resample(self):
        # systematic (low-variance) resampling
        positions = (self.rng.random() + np.arange(self.N)) / self.N
        idx = np.searchsorted(np.cumsum(self.weights), positions)
        idx = np.clip(idx, 0, self.N - 1)
        self.particles = self.particles[idx]
        self.weights = np.full(self.N, 1.0 / self.N)
        # roughening: tiny jitter to fight particle depletion
        self.particles[:, :2] += self.rng.normal(0.0, 0.05, size=(self.N, 2))
        self.particles[:, 2] = wrap_angle(
            self.particles[:, 2] + self.rng.normal(0.0, 0.02, self.N))

    # --------------------------- accessors -----------------------------
    def estimate(self):
        x = np.average(self.particles[:, 0], weights=self.weights)
        y = np.average(self.particles[:, 1], weights=self.weights)
        th = angle_mean(self.particles[:, 2], self.weights)
        return float(x), float(y), float(th)

    def cov_xy(self):
        mean = np.average(self.particles[:, :2], axis=0, weights=self.weights)
        d = self.particles[:, :2] - mean
        return (d.T * self.weights) @ d

    def particles_xy(self):
        return self.particles[:, :2].copy()
