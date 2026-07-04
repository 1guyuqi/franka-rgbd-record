from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional
import abc
from utils.trajectory_state import TrajectoryState
from utils.utils import Buffer, compute_pose_diff
import threading
from scipy.interpolate import CubicSpline
# from scipy.optimize import minimize_scalar


class TrajectoryBase(abc.ABC, metaclass=abc.ABCMeta):
    """
    Abstract base class for trajectory planning.
    """

    def __init__(self, config, buffer: Buffer, lock: threading.Lock):
        """
            @params:
                config: config file for construction of 
        """
        self._config = config
        self._buffer = buffer
        self._buffer_lock = lock
        self.dt = config["dt"]
        self._max_vel = config["max_velocity"]
        self._interpolation_type = config["interpolation_type"]
        self.trajectory_idle = True
        self._enable_online_planning  = config.get("enable_online_planning", False)
        self._interrupt_planning = False
        # self._need_clear_buffer = False
    
    @abc.abstractmethod
    def plan_trajectory(self, target:TrajectoryState, finish_time: float | None = None):
        """
            @params: 
                trajectory planner with the current and target pose
        """
        pass
    
    def interrupt_current_planning(self):
        if not self._enable_online_planning:
            return
        self._interrupt_planning = True
    
    # basic general useful method
    def _auto_generate_end_time(self, start: np.ndarray, end: np.ndarray, user_specified_time: float) -> List[float]:
        """
            Auto-generate times based on distances between waypoints.
        """
        pose_diff = compute_pose_diff(end, start)
        tranlation_time = np.linalg.norm(pose_diff[:3]) / self._max_vel
        rotation_time = np.linalg.norm(pose_diff[3:]) /  self._max_vel
        finish_time = max(max(tranlation_time, rotation_time),user_specified_time)
        if finish_time < 1 * self.dt:
            finish_time = -1
        return finish_time

    def _plan_cubic_spline(
            self,
            waypoints: np.ndarray,
            times: List[float],
            velocities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Plan trajectory using cubic splines.
        """
        n_waypoints, n_joints = waypoints.shape
        t_total = times[-1]
        t_vec = np.arange(0, t_total + self.dt, self.dt)

        positions = np.zeros((len(t_vec), n_joints))
        velocities_out = np.zeros((len(t_vec), n_joints))
        accelerations = np.zeros((len(t_vec), n_joints))

        for joint in range(n_joints):
            # Create cubic spline with velocity constraints
            cs = CubicSpline(
                times,
                waypoints[:, joint],
                bc_type=((1, velocities[0, joint]), (1, velocities[-1, joint]))
            )

            positions[:, joint] = cs(t_vec)
            velocities_out[:, joint] = cs(t_vec, 1)
            accelerations[:, joint] = cs(t_vec, 2)

        return t_vec, positions, velocities_out, accelerations

    def _plan_trapezoidal_profile(
            self,
            waypoints: np.ndarray,
            times: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Plan trajectory using trapezoidal velocity profiles.
        """
        n_waypoints, n_joints = waypoints.shape
        t_total = times[-1]
        t_vec = np.arange(0, t_total + self.dt, self.dt)

        positions = np.zeros((len(t_vec), n_joints))
        velocities = np.zeros((len(t_vec), n_joints))
        accelerations = np.zeros((len(t_vec), n_joints))

        for i in range(n_waypoints - 1):
            t_start, t_end = times[i], times[i + 1]
            mask = (t_vec >= t_start) & (t_vec <= t_end)
            t_seg = t_vec[mask] - t_start
            dt_seg = t_end - t_start

            for joint in range(n_joints):
                q0, qf = waypoints[i, joint], waypoints[i + 1, joint]

                # Generate trapezoidal profile
                pos, vel, acc = self._generate_trapezoidal_profile(q0, qf, dt_seg, t_seg)

                positions[mask, joint] = pos
                velocities[mask, joint] = vel
                accelerations[mask, joint] = acc

        return t_vec, positions, velocities, accelerations
    
    def _construct_polynomial_matrix(self, degree: str, t: float):
        A = None
        if degree == "quintic":
            A = np.array([
                    [1, t, t ** 2, t ** 3, t ** 4, t ** 5],  # q(t) = qf
                    [0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4],  # v(t) = vf
                    [0, 0, 2, 6 * t, 12 * t ** 2, 20 * t ** 3]  # a(t) = af
                ])
        
        return A
    
    def _solve_quintic_coefficients(
            self, q0: float, qf: float, v0: float, vf: float, a0: float, af: float, T: float
    ) -> np.ndarray:
        """
        Solve for quintic polynomial coefficients.
        q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        """
        num_variables = len(q0)
        # Boundary conditions matrix
        A = np.array([
                [1, 0, 0, 0, 0, 0],  # q(0) = q0
                [1, T, T ** 2, T ** 3, T ** 4, T ** 5],  # q(T) = qf
                [0, 1, 0, 0, 0, 0],  # v(0) = v0
                [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],  # v(T) = vf
                [0, 0, 2, 0, 0, 0],  # a(0) = a0
                [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]  # a(T) = af
            ])
        A_all = np.zeros((6*num_variables, 6*num_variables))
        b_all = np.array([])
        for i in range(num_variables):
            A_all[i*6:i*6+6, i*6:i*6+6] = A
            b = np.array([q0[i], qf[i], v0[i], vf[i], a0[i], af[i]])
            b_all = np.hstack((b_all, b))
        b_all = np.array(b_all).astype(np.float32)
        return np.linalg.solve(A_all, b_all)

    def _eval_polynomial(self, coeffs: np.ndarray, t: np.ndarray, derivative: int = 0) -> np.ndarray:
        """
        Evaluate polynomial and its derivatives.
        """
        num_variables = int(coeffs.shape[0] / 6)
        A_all = np.zeros((num_variables*3, num_variables*6))
        for i in range(num_variables):
            A_all[i*3:i*3+3, i*6:i*6+6] = self._construct_polynomial_matrix("quintic", t)
        curr_point = A_all @ coeffs 
        
        if derivative > 3:
                raise ValueError("Only derivatives 0, 1, 2 are supported")
        res = np.zeros(num_variables)
        for i in range(num_variables):
            res[i] = curr_point[i*3+derivative]
          
        return res

    def _generate_trapezoidal_profile(
            self, q0: float, qf: float, T: float, t: np.ndarray,
            max_vel: float = 2.0, max_acc: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate trapezoidal velocity profile.
        """
        distance = abs(qf - q0)
        direction = np.sign(qf - q0)

        # Calculate acceleration and cruise phases
        t_acc = min(max_vel / max_acc, T / 2)
        v_cruise = min(max_vel, distance / (T - t_acc))
        t_cruise = (distance - max_acc * t_acc ** 2) / v_cruise if v_cruise > 0 else 0
        t_dec = t_acc

        # Adjust if total time doesn't match
        if t_acc + t_cruise + t_dec > T:
            # Pure triangular profile
            t_acc = t_dec = T / 2
            t_cruise = 0
            v_cruise = distance / T

        positions = np.zeros_like(t)
        velocities = np.zeros_like(t)
        accelerations = np.zeros_like(t)

        for i, ti in enumerate(t):
            if ti <= t_acc:
                # Acceleration phase
                positions[i] = q0 + direction * 0.5 * max_acc * ti ** 2
                velocities[i] = direction * max_acc * ti
                accelerations[i] = direction * max_acc
            elif ti <= t_acc + t_cruise:
                # Cruise phase
                positions[i] = (q0 + direction * (0.5 * max_acc * t_acc ** 2 +
                                                  v_cruise * (ti - t_acc)))
                velocities[i] = direction * v_cruise
                accelerations[i] = 0
            else:
                # Deceleration phase
                t_dec_phase = ti - t_acc - t_cruise
                positions[i] = (q0 + direction * (0.5 * max_acc * t_acc ** 2 +
                                                  v_cruise * t_cruise +
                                                  v_cruise * t_dec_phase -
                                                  0.5 * max_acc * t_dec_phase ** 2))
                velocities[i] = direction * (v_cruise - max_acc * t_dec_phase)
                accelerations[i] = -direction * max_acc

        return positions, velocities, accelerations