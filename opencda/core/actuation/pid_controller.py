# -*- coding: utf-8 -*-
"""
PID controller implementation for vehicle control.

This module provides a PID (Proportional-Integral-Derivative) controller
for both longitudinal and lateral vehicle control in CARLA simulation.
"""

# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from collections import deque

from typing import Dict, Any, Optional
import math
import numpy as np

import carla


class Controller:
    """
    PID controller for vehicle longitudinal and lateral control.

    Implements a dual PID controller system for managing vehicle acceleration
    and steering to achieve target speeds and waypoints.

    Parameters
    ----------
    args : Dict[str, Any]
        Configuration dictionary parsed from YAML file containing:
        - max_brake : float
            Maximum brake value.
        - max_throttle : float
            Maximum throttle value.
        - max_steering : float
            Maximum steering angle.
        - lon : Dict[str, float]
            Longitudinal PID gains (k_p, k_d, k_i).
        - lat : Dict[str, float]
            Lateral PID gains (k_p, k_d, k_i).
        - dt : float
            Simulation time-step in seconds.
        - dynamic : bool
            Enable dynamic PID gain adjustment.

    Attributes
    ----------
    max_brake : float
        Maximum brake value.
    max_throttle : float
        Maximum throttle value.
    max_steering : float
        Maximum steering angle.
    _lon_k_p : float
        Longitudinal proportional gain.
    _lon_k_d : float
        Longitudinal derivative gain.
    _lon_k_i : float
        Longitudinal integral gain.
    _lon_ebuffer : deque
        Deque buffer storing longitudinal control errors.
    _lat_k_p : float
        Lateral proportional gain.
    _lat_k_d : float
        Lateral derivative gain.
    _lat_k_i : float
        Lateral integral gain.
    _lat_ebuffer : deque
        Deque buffer storing lateral control errors.
    dt : float
        Simulation time-step in seconds.
    current_transform : carla.Transform or None
        Current ego vehicle transformation in CARLA world.
    current_speed : float
        Current ego vehicle speed in m/s.
    past_steering : float
        Steering angle from previous control step.
    dynamic : bool
        Flag for dynamic PID gain adjustment.
    """

    def __init__(self, args: Dict[str, Any]):
        # longitudinal related
        self.max_brake = args["max_brake"]
        self.max_throttle = args["max_throttle"]

        self._lon_k_p = args["lon"]["k_p"]  # noqa: DC05
        self._lon_k_d = args["lon"]["k_d"]  # noqa: DC05
        self._lon_k_i = args["lon"]["k_i"]  # noqa: DC05

        self._lon_ebuffer = deque(maxlen=10)

        # lateral related
        self.max_steering = args["max_steering"]

        self._lat_k_p = args["lat"]["k_p"]  # noqa: DC05
        self._lat_k_d = args["lat"]["k_d"]  # noqa: DC05
        self._lat_k_i = args["lat"]["k_i"]  # noqa: DC05

        self._lat_ebuffer = deque(maxlen=10)

        # simulation time-step
        self.dt = args["dt"]

        # current speed and localization retrieved from sensing layer
        self.current_transform = None
        self.current_speed = 0.0
        # past steering
        self.past_steering = 0.0

        self.dynamic = args["dynamic"]

    def dynamic_pid(self) -> None:
        """
        Compute PID gains based on current speed.

        Adjusts k_p, k_d, k_i parameters dynamically according to vehicle
        speed for improved control performance.
        """
        pass

    def update_info(self, ego_pos: carla.Transform, ego_spd: float) -> None:
        """
        Update ego vehicle position and speed in controller.

        Parameters
        ----------
        ego_pos : carla.Transform
            Current transform of the ego vehicle.
        ego_spd : float
            Current speed of the ego vehicle in m/s.

        """

        self.current_transform = ego_pos
        self.current_speed = ego_spd
        if self.dynamic:
            self.dynamic_pid()

    def lon_run_step(self, target_speed: float) -> float:
        """
        Execute longitudinal PID control step.

        Computes desired acceleration to achieve target speed using PID
        control with error buffering.

        Parameters
        ----------
        target_speed : float
            Target speed of the ego vehicle in m/s.

        Returns
        -------
        float
            Desired acceleration value clipped to [-1.0, 1.0] range.
        """
        error = target_speed - self.current_speed
        self._lat_ebuffer.append(error)

        if len(self._lat_ebuffer) >= 2:
            _de = (self._lat_ebuffer[-1] - self._lat_ebuffer[-2]) / self.dt
            _ie = sum(self._lat_ebuffer) * self.dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._lat_k_p * error) + (self._lat_k_d * _de) + (self._lat_k_i * _ie), -1.0, 1.0)

    def lat_run_step(self, target_location: carla.Location) -> float:
        """
        Execute lateral PID control step.

        Computes desired steering angle to reach target location using PID
        control based on angular error between vehicle heading and target.

        Parameters
        ----------
        target_location : carla.Location
            Target location in CARLA world coordinates.

        Returns
        -------
        float
            Desired steering angle value clipped to [-1.0, 1.0] range.
        """
        v_begin = self.current_transform.location
        v_end = v_begin + carla.Location(
            x=math.cos(math.radians(self.current_transform.rotation.yaw)), y=math.sin(math.radians(self.current_transform.rotation.yaw))
        )
        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([target_location.x - v_begin.x, target_location.y - v_begin.y, 0.0])
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))
        _cross = np.cross(v_vec, w_vec)

        if _cross[2] < 0:
            _dot *= -1.0

        self._lon_ebuffer.append(_dot)
        if len(self._lon_ebuffer) >= 2:
            _de = (self._lon_ebuffer[-1] - self._lon_ebuffer[-2]) / self.dt
            _ie = sum(self._lon_ebuffer) * self.dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._lat_k_p * _dot) + (self._lat_k_d * _de) + (self._lat_k_i * _ie), -1.0, 1.0)

    def run_step(self, target_speed: float, waypoint: Optional[carla.Location]) -> carla.VehicleControl:
        """
        Execute complete control step with both longitudinal and lateral PID.

        Combines longitudinal and lateral PID controllers to generate vehicle
        control commands (throttle, brake, steering) to reach target waypoint
        at target speed.

        Parameters
        ----------
        target_speed : float
            Target speed of the ego vehicle in m/s.
        waypoint : carla.Location or None
            Target location. None triggers emergency stop.

        Returns
        -------
        carla.VehicleControl
            Vehicle control command containing throttle, brake, steering,
            and other control flags.
        """
        # control class for carla vehicle
        control = carla.VehicleControl()

        # emergency stop
        if target_speed == 0 or waypoint is None:
            control.steer = 0.0  # noqa: DC05
            control.throttle = 0.0  # noqa: DC05
            control.brake = 1.0  # noqa: DC05
            control.hand_brake = False  # noqa: DC05
            return control

        acceleration = self.lon_run_step(target_speed)
        current_steering = self.lat_run_step(waypoint)

        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throttle)  # noqa: DC05
            control.brake = 0.0  # noqa: DC05
        else:
            control.throttle = 0.0  # noqa: DC05
            control.brake = min(abs(acceleration), self.max_brake)  # noqa: DC05

        # Steering regulation: changes cannot happen abruptly, can't steer too
        # much.
        if current_steering > self.past_steering + 0.2:
            current_steering = self.past_steering + 0.2
        elif current_steering < self.past_steering - 0.2:
            current_steering = self.past_steering - 0.2

        if current_steering >= 0:
            steering = min(self.max_steering, current_steering)
        else:
            steering = max(-self.max_steering, current_steering)

        control.steer = steering  # noqa: DC05
        control.hand_brake = False  # noqa: DC05
        control.manual_gear_shift = False  # noqa: DC05
        self.past_steering = steering
        return control
