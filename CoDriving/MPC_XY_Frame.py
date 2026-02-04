"""
Linear MPC controller (X-Y frame)
Code modified from MPC_XY_Frame.py developed by Huiming Zhou et al.
link: https://github.com/zhm-real/MotionPlanning/blob/master/Control/MPC_XY_Frame.py
"""

import math
import cvxpy
import yaml
import numpy as np

from typing import Any, Dict, Optional, Sequence, List, Tuple
import numpy.typing as npt


def load_config() -> Dict[str, Any]:
    """
    Load MPC configuration from YAML.

    Returns
    -------
    dict
        Parsed configuration object (nested dictionaries/lists).
    """
    with open(r"configuration\MPC_config.yaml") as f:
        return yaml.safe_load(f)


# K_STEER = 56.0
# K_THROTTLE = 1.5
# K_BRAKE = 0.15
cfg = load_config()
system_params = cfg["mpc"]["system"]
vehicle_params = cfg["mpc"]["vehicle"]
base_params = cfg["mpc"]["base"]


class P:
    """
    MPC and vehicle parameters loaded from configuration.

    This class stores configuration constants as class attributes and is used as
    a global parameter holder by the controller/state update logic.
    """

    # System config
    NX = system_params["nx"]  # state vector: z = [x, y, v, phi]
    NU = system_params["nu"]  # input vector: u = [acceleration, steer]
    T = system_params["t"]  # finite time horizon length
    T_aug = system_params["t_aug"]  # finite time horizon length
    # Dekai: if T is 1, the vehicle would have a larger turning radius

    # MPC config
    # Q = np.diag([12.0, 12.0, 1.0, 12.0])  # penalty for states   # Dekai: if set the third value (penalty for velocity) to 0.0, the vehicle is difficult to start.
    # Qf = np.diag([5.0, 5.0, 1.0, 20.0])  # penalty for end state # Dekai: since now we only trace a single target point but not a desired traj, only Qf is used but not Q
    Qf = np.diag(
        base_params["qf"]
    )  # penalty for end state # Dekai: since now we only trace a single target point but not a desired traj, only Qf is used but not Q
    R = np.diag(base_params["r"])  # penalty for inputs  # Dekai: had better choose large penalty for steering to avoid zig-zag
    Rd = np.diag(base_params["rd"])  # penalty for change of inputs

    # dist_stop = base_params["dist_stop"]  # stop permitted when dist to goal < dist_stop
    # speed_stop = base_params["speed_stop"]  # stop permitted when speed < speed_stop
    # time_max = base_params["time_max"]  # max simulation time
    iter_max = base_params["iter_max"]  # max iteration
    target_speed = base_params["target_speed"]  # target speed
    # N_IND = base_params["N_IND"]  # search index number
    dt = base_params["dt"]  # time step
    # d_dist = base_params["d_dist"]  # dist step
    du_res = base_params["du_res"]  # threshold for stopping iteration

    # vehicle config
    # RF = vehicle_params["rf"]  # [m] distance from rear to vehicle front end of vehicle
    # RB = vehicle_params["rb"]  # [m] distance from rear to vehicle back end of vehicle
    W = vehicle_params["w"]  # [m] width of vehicle
    # WD = vehicle_params["wd"] * W  # [m] distance between left-right wheels
    WB = vehicle_params["wb"]  # [m] Wheel base
    # TR = vehicle_params["tr"]  # [m] Tyre radius
    # TW = vehicle_params["tw"]  # [m] Tyre width

    steer_max = np.deg2rad(vehicle_params["steer_max"])  # max steering angle [rad]
    # steer_change_max = np.deg2rad(vehicle_params["steer_change_max"])  # maximum steering speed [rad/s]
    speed_max = vehicle_params["speed_max"]  # maximum speed [m/s]
    speed_min = vehicle_params["speed_min"]  # minimum speed [m/s]
    # acceleration_max = vehicle_params["acceleration_max"]  # maximum acceleration [m/s2]


class Node:
    """
    Vehicle kinematic state (x-y frame) and update step.

    Parameters
    ----------
    x : float, optional
        Position x [m].
    y : float, optional
        Position y [m].
    yaw : float, optional
        Heading angle [rad].
    v : float, optional
        Speed [m/s].
    direct : float, optional
        Direction multiplier (commonly 1.0 forward, -1.0 backward).

    Attributes
    ----------
    x : float
        Current x position [m].
    y : float
        Current y position [m].
    yaw : float
        Current heading [rad].
    v : float
        Current speed [m/s].
    direct : float
        Current direction multiplier.
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, v: float = 0.0, direct: float = 1.0):  # current state
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.direct = direct

    def update(self, a: float, delta: float, direct: float) -> None:
        """
        Update state using a simple kinematic bicycle-like model.

        Parameters
        ----------
        a : float
            Longitudinal acceleration command [m/s^2].
        delta : float
            Steering angle command [rad] (will be clamped to [-P.steer_max, P.steer_max]).
        direct : float
            Direction multiplier to apply during update (e.g., 1.0 or -1.0).
        """
        delta = self.limit_input_delta(delta)
        self.x += self.v * math.cos(self.yaw) * P.dt
        self.y += self.v * math.sin(self.yaw) * P.dt
        self.yaw += self.v / P.WB * math.tan(delta) * P.dt
        self.direct = direct
        self.v += self.direct * a * P.dt
        self.v = self.limit_speed(self.v)

    @staticmethod
    def limit_input_delta(delta: float) -> float:
        """
        Clamp steering angle to vehicle limits.

        Parameters
        ----------
        delta : float
            Steering angle [rad].

        Returns
        -------
        float
            Clamped steering angle in [-P.steer_max, P.steer_max].
        """
        if delta >= P.steer_max:
            return P.steer_max

        if delta <= -P.steer_max:
            return -P.steer_max

        return delta

    @staticmethod
    def limit_speed(v: float) -> float:
        """
        Clamp speed to vehicle limits.

        Parameters
        ----------
        v : float
            Speed [m/s].

        Returns
        -------
        float
            Clamped speed in [P.speed_min, P.speed_max].
        """
        if v >= P.speed_max:
            return P.speed_max

        if v <= P.speed_min:
            return P.speed_min

        return v


# class PATH:
#     def __init__(self, cx, cy, cyaw, cv, ck=None):
#         self.cx = [cx]
#         self.cy = [cy]
#         self.cyaw = [cyaw]
#         # self.ck = [ck]
#         # self.length = len(cx)
#         self.ind_old = 0
#         self.cv = [cv]

#     def update_route(self, cx, cy, cyaw, cv):
#         self.cx += [cx]
#         self.cy += [cy]
#         self.cyaw += [cyaw]
#         self.cv += [cv]

#         # self.cx = [cx]
#         # self.cy = [cy]
#         # self.cyaw = [cyaw]

#     def nearest_index(self, node):
#         """
#         calc index of the nearest node in N steps
#         :param node: current information
#         :return: nearest index, lateral distance to ref point
#         """

#         dx = [node.x - x for x in self.cx[self.ind_old : (self.ind_old + P.N_IND)]]
#         dy = [node.y - y for y in self.cy[self.ind_old : (self.ind_old + P.N_IND)]]
#         dist = np.hypot(dx, dy)

#         ind_in_N = int(np.argmin(dist))
#         ind = self.ind_old + ind_in_N
#         self.ind_old = ind

#         rear_axle_vec_rot_90 = np.array([[math.cos(node.yaw + math.pi / 2.0)], [math.sin(node.yaw + math.pi / 2.0)]])

#         vec_target_2_rear = np.array([[dx[ind_in_N]], [dy[ind_in_N]]])

#         er = np.dot(vec_target_2_rear.T, rear_axle_vec_rot_90)
#         er = er[0][0]

#         return ind, er


# def calc_ref_trajectory_in_T_step(node, ref_path, sp=None) -> np.ndarray:
#     """
#     calc referent trajectory in T steps: [x, y, v, yaw]
#     using the current velocity, calc the T points along the reference path
#     :param node: current information
#     :param ref_path: reference path: [x, y, v, yaw]
#     :param sp: speed profile (designed speed strategy)
#     :return: reference trajectory [4, T+1]
#     """

#     # if sp is None:
#     # sp = np.ones(200, dtype=np.float) * 40 / 3.6    # max speed in sumo routefile

#     z_ref = np.zeros((P.NX, P.T + 1))
#     length = len(ref_path.cx)

#     # ============== get the clost step and look further for N steps ============== #

#     # ind, _ = ref_path.nearest_index(node)

#     # z_ref[0, 0] = ref_path.cx[ind]
#     # z_ref[1, 0] = ref_path.cy[ind]
#     # # z_ref[2, 0] = sp[ind]
#     # z_ref[2, 0] = ref_path.cv[ind]
#     # z_ref[3, 0] = ref_path.cyaw[ind]

#     # dist_move = 0.0

#     # for i in range(1, P.T + 1):
#     #     dist_move += abs(node.v) * P.dt
#     #     ind_move = int(round(dist_move / P.d_dist))
#     #     index = min(ind + ind_move, length - 1)

#     #     z_ref[0, i] = ref_path.cx[index]
#     #     z_ref[1, i] = ref_path.cy[index]
#     #     # z_ref[2, i] = sp[index]
#     #     z_ref[2, i] = ref_path.cv[index]
#     #     z_ref[3, i] = ref_path.cyaw[index]

#     # ============== get the clost step and look further for N steps ============== #

#     # ============== get the last N steps ============== #

#     z_ref[0, -1] = ref_path.cx[-1]
#     z_ref[1, -1] = ref_path.cy[-1]
#     z_ref[2, -1] = ref_path.cv[-1]
#     z_ref[3, -1] = ref_path.cyaw[-1]
#     dist_move = 0.0
#     for i in range(P.T - 1, -1, -1):
#         dist_move += abs(node.v) * P.dt
#         ind_move = int(round(dist_move / P.d_dist))
#         index = max(length - 1 - ind_move, 0)

#         z_ref[0, i] = ref_path.cx[index]
#         z_ref[1, i] = ref_path.cy[index]
#         z_ref[2, i] = ref_path.cv[index]
#         z_ref[3, i] = ref_path.cyaw[index]

#     # ============== get the last N steps ============== #

#     return z_ref, 0


# def get_destination_in_T_step(node, ref_path) -> np.ndarray:
#     """
#     calc desired destination in T steps: [x, y, v, yaw]
#     :param node: current information
#     :param ref_path: reference path: [x, y, v, yaw]
#     :return: destination [4]
#     """

#     z_target = np.zeros(4)
#     z_target[0] = ref_path.cx[-1]
#     z_target[1] = ref_path.cy[-1]
#     z_target[2] = ref_path.cv[-1]
#     z_target[3] = ref_path.cyaw[-1]

#     return z_target


# def linear_mpc_control(z_ref, z0, a_old, delta_old):
#     """
#     linear mpc controller
#     :param z_ref: reference trajectory in T steps
#     :param z0: initial state vector
#     :param a_old: acceleration of T steps of last time
#     :param delta_old: delta of T steps of last time
#     :return: acceleration and delta strategy based on current information
#     """

#     if a_old is None or delta_old is None:
#         a_old = [0.0] * P.T
#         delta_old = [0.0] * P.T

#     x, y, yaw, v = None, None, None, None

#     for k in range(P.iter_max):
#         z_bar = predict_states_in_T_step(z0, a_old, delta_old, z_ref)
#         a_rec, delta_rec = a_old[:], delta_old[:]
#         a_old, delta_old, x, y, yaw, v = solve_linear_mpc(z_ref, z_bar, z0, delta_old)

#         du_a_max = max([abs(ia - iao) for ia, iao in zip(a_old, a_rec)])
#         du_d_max = max([abs(ide - ido) for ide, ido in zip(delta_old, delta_rec)])

#         if max(du_a_max, du_d_max) < P.du_res:
#             break

#     return a_old, delta_old, x, y, yaw, v


def linear_mpc_control_data_aug(
    z_ref: npt.NDArray[np.float64], z0: Sequence[float], a_old: Optional[List[float]], delta_old: Optional[List[float]]
) -> Tuple:
    """
    Run iterative linear MPC over the augmented horizon.

    Parameters
    ----------
    z_ref : numpy.typing.NDArray[numpy.float64]
        Reference trajectory of shape (4, P.T_aug + 1). State order:
        [x, y, v, yaw].
    z0 : Sequence[float]
        Initial state [x, y, v, yaw], length 4.
    a_old : list[float] | None
        Previous acceleration sequence of length P.T_aug. If None, initializes with zeros.
    delta_old : list[float] | None
        Previous steering sequence of length P.T_aug. If None, initializes with zeros.

    Returns
    -------
    Acceleration and delta strategy based on current information
    """

    if a_old is None or delta_old is None:
        a_old = [0.0] * P.T_aug
        delta_old = [0.0] * P.T_aug

    x, y, yaw, v = None, None, None, None

    for k in range(P.iter_max):
        z_bar = predict_states_in_T_step(z0, a_old, delta_old, z_ref, pred_len=P.T_aug)
        a_rec, delta_rec = a_old[:], delta_old[:]
        a_old, delta_old, x, y, yaw, v = solve_linear_mpc(z_ref, z_bar, z0, delta_old, pred_len=P.T_aug)

        du_a_max = max([abs(ia - iao) for ia, iao in zip(a_old, a_rec)])
        du_d_max = max([abs(ide - ido) for ide, ido in zip(delta_old, delta_rec)])

        if max(du_a_max, du_d_max) < P.du_res:
            break

    return a_old, delta_old, x, y, yaw, v


def predict_states_in_T_step(
    z0: Sequence[float], a: npt.NDArray[np.float64], delta: npt.NDArray[np.float64], z_ref: npt.NDArray[np.float64], pred_len: int = P.T
) -> npt.NDArray[np.float64]:
    """
    Predict vehicle states over a horizon using given control sequences.

    Parameters
    ----------
    z0 : Sequence[float]
        Initial state [x, y, v, yaw], length 4.
    a : numpy.typing.NDArray[numpy.float64]
        Acceleration sequence, shape (pred_len,).
    delta : numpy.typing.NDArray[numpy.float64]
        Steering sequence, shape (pred_len,).
    z_ref : numpy.typing.NDArray[numpy.float64]
        Reference trajectory used for sizing the output, shape (4, pred_len + 1).
    pred_len : int, optional
        Prediction horizon.

    Returns
    -------
    z_bar : numpy.typing.NDArray[numpy.float64]
        Predicted state trajectory of shape (4, pred_len + 1).
    """

    z_bar = z_ref * 0.0

    for i in range(P.NX):
        z_bar[i, 0] = z0[i]

    node = Node(x=z0[0], y=z0[1], v=z0[2], yaw=z0[3])

    for ai, di, i in zip(a, delta, range(1, pred_len + 1)):
        node.update(ai, di, 1.0)  # 1.0 is forward direction
        z_bar[0, i] = node.x
        z_bar[1, i] = node.y
        z_bar[2, i] = node.v
        z_bar[3, i] = node.yaw

    return z_bar


def predict_states_in_T_step_2(curr_state: Node, a: npt.NDArray[np.float64], delta: npt.NDArray[np.float64], T: int = P.T) -> npt.NDArray[np.float64]:
    """
    Predict states over a horizon starting from a Node object.

    Parameters
    ----------
    curr_state : Node
        Current vehicle state. This object is updated in-place during prediction.
    a : numpy.typing.NDArray[numpy.float64]
        Acceleration sequence, shape (T,).
    delta : numpy.typing.NDArray[numpy.float64]
        Steering sequence, shape (T,).
    T : int, optional
        Prediction horizon.

    Returns
    -------
    z_bar : numpy.typing.NDArray[numpy.float64]
        Predicted state trajectory of shape (4, T + 1), including the current state.
    """

    z_bar = np.zeros((4, T + 1))
    z_bar[:, 0] = curr_state.x, curr_state.y, curr_state.v, curr_state.yaw

    for ai, di, i in zip(a, delta, range(1, P.T + 1)):
        curr_state.update(ai, di, 1.0)  # 1.0 is forward direction
        z_bar[0, i] = curr_state.x
        z_bar[1, i] = curr_state.y
        z_bar[2, i] = curr_state.v
        z_bar[3, i] = curr_state.yaw

    return z_bar


def calc_linear_discrete_model(
    v: float, phi: float, delta: float
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute linearized discrete-time dynamics matrices.

    Parameters
    ----------
    v : float
        Linearization speed (v_bar).
    phi : float
        Linearization heading angle (phi_bar), radians.
    delta : float
        Linearization steering angle (delta_bar), radians.

    Returns
    -------
    A : numpy.typing.NDArray[numpy.float64]
        State transition matrix, shape (4, 4).
    B : numpy.typing.NDArray[numpy.float64]
        Input matrix, shape (4, 2).
    C : numpy.typing.NDArray[numpy.float64]
        Affine term, shape (4,).
    """

    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    P_dt_v = P.dt * v

    A = np.array(
        [
            [1.0, 0.0, P.dt * cos_phi, -P_dt_v * sin_phi],
            [0.0, 1.0, P.dt * sin_phi, P_dt_v * cos_phi],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, P.dt * math.tan(delta) / P.WB, 1.0],
        ]
    )

    B = np.array([[0.0, 0.0], [0.0, 0.0], [P.dt, 0.0], [0.0, P_dt_v / (P.WB * math.cos(delta) ** 2)]])

    C = np.array([P_dt_v * sin_phi * phi, -P_dt_v * cos_phi * phi, 0.0, -P_dt_v * delta / (P.WB * math.cos(delta) ** 2)])

    return A, B, C


def solve_linear_mpc(
    z_ref: npt.NDArray[np.float64], z_bar: npt.NDArray[np.float64], z0: Sequence[float], d_bar: npt.NDArray[np.float64], pred_len: int = P.T
) -> Tuple:
    """
    Solve the linear MPC QP with CVXPY (OSQP).

    Parameters
    ----------
    z_ref : numpy.typing.NDArray[numpy.float64]
        Reference trajectory, shape (4, pred_len + 1).
    z_bar : numpy.typing.NDArray[numpy.float64]
        Predicted states used for linearization, shape (4, pred_len + 1).
    z0 : Sequence[float]
        Initial state [x, y, v, yaw], length 4.
    d_bar : numpy.typing.NDArray[numpy.float64]
        Steering angles used for linearization, shape (pred_len,).
    pred_len : int, optional
        Horizon length.

    Returns
    -------
    Tuple
    """

    z = cvxpy.Variable((P.NX, pred_len + 1))
    u = cvxpy.Variable((P.NU, pred_len))

    cost = 0.0
    constrains = []

    for t in range(pred_len):
        cost += cvxpy.quad_form(u[:, t], P.R)
        # cost += cvxpy.quad_form(z_ref[:, t] - z[:, t], P.Q)

        A, B, C = calc_linear_discrete_model(z_bar[2, t], z_bar[3, t], d_bar[t])

        constrains += [z[:, t + 1] == A @ z[:, t] + B @ u[:, t] + C]

        if t < pred_len - 1:
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], P.Rd)
            # constrains += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= P.steer_change_max * P.dt]

    cost += cvxpy.quad_form(z_ref[:, pred_len] - z[:, pred_len], P.Qf)

    constrains += [z[:, 0] == z0]
    # constrains += [z[2, :] <= P.speed_max]
    # constrains += [z[2, :] >= P.speed_min]
    # constrains += [cvxpy.abs(u[0, :]) <= P.acceleration_max]
    constrains += [cvxpy.abs(u[1, :]) <= P.steer_max]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constrains)
    prob.solve(solver=cvxpy.OSQP)

    a, delta, x, y, yaw, v = None, None, None, None, None, None

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        x = z.value[0, :]
        y = z.value[1, :]
        v = z.value[2, :]
        yaw = z.value[3, :]
        a = u.value[0, :]
        delta = u.value[1, :]
    else:
        print("Cannot solve linear mpc!")

    return a, delta, x, y, yaw, v


def solve_linear_mpc_2(z_target: npt.NDArray[np.float64], z_bar: npt.NDArray[np.float64], z0: List, d_bar: npt.NDArray[np.float64]) -> Tuple:
    """
    Solve MPC with a terminal target cost (CVXPY + OSQP).

    Parameters
    ----------
    z_target : numpy.typing.NDArray[numpy.float64]
        Target state [x, y, v, yaw], shape (4,).
    z_bar : numpy.typing.NDArray[numpy.float64]
        Predicted states used for linearization, shape (4, P.T + 1).
    z0 : Sequence[float]
        Initial state [x, y, v, yaw], length 4.
    d_bar : numpy.typing.NDArray[numpy.float64]
        Steering angles used for linearization, shape (P.T,).

    Returns
    -------
    Optimal acceleration and steering strategy
    """

    z = cvxpy.Variable((P.NX, P.T + 1))
    u = cvxpy.Variable((P.NU, P.T))

    cost = 0.0
    constrains = []

    for t in range(P.T):
        cost += cvxpy.quad_form(u[:, t], P.R)
        # cost += cvxpy.quad_form(z_ref[:, t] - z[:, t], P.Q)

        A, B, C = calc_linear_discrete_model(z_bar[2, t], z_bar[3, t], d_bar[t])

        constrains += [z[:, t + 1] == A @ z[:, t] + B @ u[:, t] + C]

        if t < P.T - 1:
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], P.Rd)
            # constrains += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= P.steer_change_max * P.dt]

    cost += cvxpy.quad_form(z_target - z[:, P.T], P.Qf)

    constrains += [z[:, 0] == z0]
    # constrains += [z[2, :] <= P.speed_max]
    # constrains += [z[2, :] >= P.speed_min]
    # constrains += [cvxpy.abs(u[0, :]) <= P.acceleration_max]
    constrains += [cvxpy.abs(u[1, :]) <= P.steer_max]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constrains)
    prob.solve(solver=cvxpy.OSQP)

    a, delta, x, y, yaw, v = None, None, None, None, None, None

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        x = z.value[0, :]
        y = z.value[1, :]
        v = z.value[2, :]
        yaw = z.value[3, :]
        a = u.value[0, :]
        delta = u.value[1, :]
    else:
        print("Cannot solve linear mpc!")

    return a, delta, x, y, yaw, v


# def calc_speed_profile(cx, cy, cyaw, target_speed) -> list:
#     """
#     design appropriate speed strategy
#     :param cx: x of reference path [m]
#     :param cy: y of reference path [m]
#     :param cyaw: yaw of reference path [m]
#     :param target_speed: target speed [m/s]
#     :return: speed profile
#     """

#     speed_profile = [target_speed] * len(cx)
#     direction = 1.0  # forward

#     # Set stop point
#     for i in range(len(cx) - 1):
#         dx = cx[i + 1] - cx[i]
#         dy = cy[i + 1] - cy[i]

#         move_direction = math.atan2(dy, dx)

#         if dx != 0.0 and dy != 0.0:
#             dangle = abs(pi_2_pi(move_direction - cyaw[i]))
#             if dangle >= math.pi / 4.0:
#                 direction = -1.0
#             else:
#                 direction = 1.0

#         if direction != 1.0:
#             speed_profile[i] = -target_speed
#         else:
#             speed_profile[i] = target_speed

#     speed_profile[-1] = 0.0

#     return speed_profile


# def pi_2_pi(angle):
#     if angle > math.pi:
#         return angle - 2.0 * math.pi

#     if angle < -math.pi:
#         return angle + 2.0 * math.pi

#     return angle


def MPC_module(
    curr_state: Node, target_state: npt.NDArray[np.float64], a_old: List, delta_old: List, T: int = P.T
) -> Tuple[List[float], List[float]]:
    """
    High-level MPC wrapper returning control sequences.

    Parameters
    ----------
    curr_state : Node
        Current state (x, y, v, yaw).
    target_state : numpy.typing.NDArray[numpy.float64]
        Target state [x, y, v, yaw], shape (4,).
    a_old : list[float] | None
        Previous acceleration sequence of length T, or None to initialize.
    delta_old : list[float] | None
        Previous steering sequence of length T, or None to initialize.
    T : int, optional
        Horizon length. If a_old/delta_old are provided, their length must equal T.

    Returns
    -------
    a_seq : list[float]
        Acceleration sequence of length T.
    delta_seq : list[float]
        Steering sequence of length T.
    """

    if a_old is None or delta_old is None:
        a_old = [0.0] * P.T
        delta_old = [0.0] * P.T
    else:
        assert len(a_old) == T and len(delta_old) == T

    x, y, yaw, v = None, None, None, None
    z0 = [curr_state.x, curr_state.y, curr_state.v, curr_state.yaw]

    for k in range(P.iter_max):
        z_bar = predict_states_in_T_step_2(curr_state, a_old, delta_old, T)
        a_rec, delta_rec = a_old[:], delta_old[:]
        a_old, delta_old, x, y, yaw, v = solve_linear_mpc_2(target_state, z_bar, z0, delta_old)

        du_a_max = max([abs(ia - iao) for ia, iao in zip(a_old, a_rec)])
        du_d_max = max([abs(ide - ido) for ide, ido in zip(delta_old, delta_rec)])

        if max(du_a_max, du_d_max) < P.du_res:
            break

    return a_old, delta_old
