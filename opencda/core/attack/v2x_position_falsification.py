"""
V2X Position Falsification attack and detection.

Unlike GNSS Spoofing (which corrupts a vehicle's OWN position perception at
the sensor level), this attack falsifies the position/speed broadcast via
V2X (CAM messages) to deceive OTHER vehicles. The attacker knows its real
position but intentionally lies when other vehicles query it.

This module also provides V2XAttackDetector, used on the receiver side to
flag suspicious neighbor broadcasts via three strategies (threshold, drift,
velocity_consistency).
"""

import logging
import math
from collections import defaultdict, deque

import carla

logger = logging.getLogger("cavise.opencda.opencda.core.attack.v2x_position_falsification")


class V2XConstantOffsetAttacker:
    """
    Reports position with a constant offset (e.g., shifted to adjacent lane).
    Speed is not modified.
    """

    def __init__(self, dx: float, dy: float, dz: float) -> None:
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def falsify_position(self, true_pos: carla.Transform) -> carla.Transform:
        loc = carla.Location(
            x=true_pos.location.x + self.dx,
            y=true_pos.location.y + self.dy,
            z=true_pos.location.z + self.dz,
        )
        return carla.Transform(loc, true_pos.rotation)

    def falsify_speed(self, true_speed: float) -> float:
        return true_speed


class V2XGhostVehicleAttacker:
    """
    Reports a completely fabricated ("ghost") location and speed,
    ignoring true values entirely.
    """

    def __init__(self, ghost_x: float, ghost_y: float, ghost_z: float, ghost_speed: float = 0.0) -> None:
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z
        self.ghost_speed = ghost_speed

    def falsify_position(self, true_pos: carla.Transform) -> carla.Transform:
        loc = carla.Location(x=self.ghost_x, y=self.ghost_y, z=self.ghost_z)
        return carla.Transform(loc, true_pos.rotation)

    def falsify_speed(self, true_speed: float) -> float:
        return self.ghost_speed


class V2XProgressiveDriftAttacker:
    """
    Gradually drifts reported position away from the real one by
    (dx, dy, dz) meters per tick. Speed is not modified.
    """

    def __init__(self, dx: float, dy: float, dz: float) -> None:
        self.tick = 0
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def falsify_position(self, true_pos: carla.Transform) -> carla.Transform:
        loc = carla.Location(
            x=true_pos.location.x + self.tick * self.dx,
            y=true_pos.location.y + self.tick * self.dy,
            z=true_pos.location.z + self.tick * self.dz,
        )
        self.tick += 1
        return carla.Transform(loc, true_pos.rotation)

    def falsify_speed(self, true_speed: float) -> float:
        return true_speed


class V2XAttackDetector:
    """
    Receiver-side detector for V2X position falsification.

    Compares positions reported by neighbors over V2X with their ground-truth
    CARLA position (in production this would come from independent perception
    such as LiDAR/camera/radar; here CARLA's get_location is used as that
    proxy).

    Strategies:
      - threshold:            single-tick distance check (catches Constant /
                              Ghost-style attacks).
      - drift:                growing mean error over a sliding window
                              (catches Progressive Drift).
      - velocity_consistency: reported displacement vs. reported speed
                              (passive, GT-free sanity check).
    """

    def __init__(
        self,
        ego_id,
        method,
        position_threshold,
        window_size,
        drift_mean_threshold,
        speed_threshold,
        dt,
    ):
        self.ego_id = ego_id
        self.method = method
        self.position_threshold = position_threshold
        self.window_size = window_size
        self.drift_mean_threshold = drift_mean_threshold
        self.speed_threshold = speed_threshold
        self.dt = dt

        self._error_history = defaultdict(lambda: deque(maxlen=self.window_size))
        self._last_reported = {}

    def check(self, neighbor_id, reported_pos, reported_speed, true_loc, tick):
        if self.method == "threshold":
            return self._threshold(neighbor_id, reported_pos, true_loc, tick)
        if self.method == "drift":
            return self._drift(neighbor_id, reported_pos, true_loc, tick)
        if self.method == "velocity_consistency":
            return self._velocity_consistency(neighbor_id, reported_pos, reported_speed, tick)
        logger.warning(f"Unknown V2X attack_detection method: {self.method}")
        return False

    @staticmethod
    def _xy_distance(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def _threshold(self, neighbor_id, reported_pos, true_loc, tick):
        err = self._xy_distance(reported_pos.location, true_loc)
        if err > self.position_threshold:
            logger.warning(
                f"[V2X-DETECTOR][CAV {self.ego_id}] Anomaly in V2X data from CAV {neighbor_id}: "
                f"method=threshold, error={err:.2f}m, threshold={self.position_threshold}m, tick={tick}"
            )
            return True
        return False

    def _drift(self, neighbor_id, reported_pos, true_loc, tick):
        err = self._xy_distance(reported_pos.location, true_loc)
        history = self._error_history[neighbor_id]
        history.append(err)
        if len(history) < self.window_size:
            return False
        mean_err = sum(history) / len(history)
        if mean_err > self.drift_mean_threshold and history[-1] > history[0]:
            logger.warning(
                f"[V2X-DETECTOR][CAV {self.ego_id}] Anomaly in V2X data from CAV {neighbor_id}: "
                f"method=drift, mean_error={mean_err:.2f}m, window={self.window_size}, "
                f"threshold={self.drift_mean_threshold}m, tick={tick}"
            )
            return True
        return False

    def _velocity_consistency(self, neighbor_id, reported_pos, reported_speed, tick):
        prev = self._last_reported.get(neighbor_id)
        self._last_reported[neighbor_id] = (reported_pos.location, reported_speed)
        if prev is None:
            return False
        prev_loc, prev_speed = prev
        actual_disp = self._xy_distance(reported_pos.location, prev_loc)
        # OpenCDA reports speed in km/h; convert to m/s for displacement comparison.
        avg_speed_mps = ((prev_speed + reported_speed) / 2.0) / 3.6
        expected_disp = avg_speed_mps * self.dt
        diff = abs(actual_disp - expected_disp)
        if diff > self.speed_threshold:
            logger.warning(
                f"[V2X-DETECTOR][CAV {self.ego_id}] Anomaly in V2X data from CAV {neighbor_id}: "
                f"method=velocity_consistency, |actual-expected|={diff:.2f}m, "
                f"actual={actual_disp:.2f}m, expected={expected_disp:.2f}m, "
                f"threshold={self.speed_threshold}m, tick={tick}"
            )
            return True
        return False
