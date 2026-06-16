"""Common interface every localization filter implements."""

import numpy as np


class BaseFilter:
    """
    A localization filter for a known map.

    Sub-classes implement:
        predict(u, dt)   -- motion / prediction step, u = (v, w)
        update(meas)     -- measurement / correction step
        estimate()       -- returns (x, y, theta)

    Optional, used only for visualization:
        cov_xy()         -- 2x2 position covariance (or None)
        particles_xy()   -- (N, 2) particle cloud (or None)
        belief_grid()    -- (Nx, Ny) marginal belief (or None)
    """

    name = "Base"
    color = "red"

    def __init__(self, env, init_pose, init_cov):
        self.env = env
        self.init_pose = np.asarray(init_pose, dtype=float)
        self.init_cov = np.asarray(init_cov, dtype=float)

    # ---- to be overridden ----
    def predict(self, u, dt):
        raise NotImplementedError

    def update(self, meas):
        raise NotImplementedError

    def estimate(self):
        raise NotImplementedError

    # ---- optional visualization hooks ----
    def cov_xy(self):
        return None

    def particles_xy(self):
        return None

    def belief_grid(self):
        return None
