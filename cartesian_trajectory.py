from __future__ import annotations

from trajectory_base import TrajectoryBase
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from utils.utils import TrajectoryState, check_traj_size
from scipy.spatial.transform import Rotation, Slerp
from utils.utils import Buffer
import threading
import time
import logging  # stdlib logging (replaces glog)

# Directly support the dual target interpolation
class CartessianTrajectory(TrajectoryBase):
    def __init__(self, config, buffer: Buffer, lock: threading.Lock):
        super().__init__(config, buffer, lock)
        self._enable_motion = config.get("enable_motion", False)
        self.is_duo_target = True if buffer._dim == 14 else False

    def plan_trajectory(self, target: TrajectoryState, finish_time: float | None = None):
        if not self.trajectory_idle:
            logging.info(f'not idle, busy!!!')
            return
        self.trajectory_idle = False

        # interpolation check
        if not self._interpolation_type in ["cubic", "quintic", "trapezoidal"]:
            raise ValueError(f"{self._interpolation_type} interpolation, "
                             "for translation is not supported!!!")

        flag = check_traj_size(target, self._buffer._dim)
        if not flag:
            logging.warning("target dim is different with buffer dim!!!")
            logging.warning(f'target shape in cartesian traj: {target._zero_order_values.shape},'
                     f'buffer size: {self._buffer._dim}')
            self.trajectory_idle = True
            return

        finish_time = -1 if finish_time is None else finish_time
        end_time = self._auto_generate_end_time(target._zero_order_values[0][:7], 
                                                target._zero_order_values[1][:7],
                                                finish_time)
        if self.is_duo_target:
            end_time1 = self._auto_generate_end_time(target._zero_order_values[0][7:],
                                                     target._zero_order_values[1][7:],
                                                finish_time)
            end_time = max(end_time, end_time1)

        if end_time < 0:
            self.trajectory_idle = True
            return 
        
        # Profile generation
        trans_coeff, slerp = self._generate_traj_profile(target, end_time)

        # Precompute all trajectory points
        all_points = []
        for t in np.arange(0, end_time, self.dt):
            if not self.is_duo_target:
                curr_point = self._eval_profile(trans_coeff['single'], slerp['single'], t)
            elif self.is_duo_target:
                curr_point_l = self._eval_profile(trans_coeff['left'], slerp['left'], t)
                curr_point_r = self._eval_profile(trans_coeff['right'], slerp['right'], t)
                curr_point = np.hstack((curr_point_l, curr_point_r))
            all_points.append(curr_point)

        # Push the precomputed points to the buffer
        self._buffer_lock.acquire()
        for point in all_points:
            time_stamp = time.perf_counter()  # Get the timestamp for each point
            self._buffer.push_data(point, time_stamp)
        self._buffer_lock.release()

        self.trajectory_idle = True
        
    def _generate_traj_profile(self, target: TrajectoryState, end_time: float):
        # slerp profile
        rotations = Rotation.concatenate([
            Rotation.from_quat(target._zero_order_values[0][3:7]),
            Rotation.from_quat(target._zero_order_values[1][3:7])
        ])
        slerp = Slerp([0, end_time], rotations)
        if self.is_duo_target:
            rotations = Rotation.concatenate([
                Rotation.from_quat(target._zero_order_values[0][10:14]),
                Rotation.from_quat(target._zero_order_values[1][10:14])
            ])
            slerp1 = Slerp([0, end_time], rotations)
            slerp = {'left': slerp, 'right': slerp1}
        else:
            slerp = {'single': slerp}
        
        # trans
        trans_coeff = None
        if self._interpolation_type == "cubic":
            raise NotImplementedError
            # self._plan_cubic_spline(translation_target_posi, [finish_time],
            #                               translation_target_vel)
        elif self._interpolation_type == "quintic":
            trans_coeff = self._solve_quintic_coefficients(target._zero_order_values[0][:3],
                                                            target._zero_order_values[1][:3],
                                                            target._first_order_values[0][:3],
                                                            target._first_order_values[1][:3],
                                                            target._second_order_values[0][:3],
                                                            target._second_order_values[1][:3],
                                                            end_time)
            if self.is_duo_target:
                trans_coeff1 = self._solve_quintic_coefficients(target._zero_order_values[0][7:10],
                                                            target._zero_order_values[1][7:10],
                                                            target._first_order_values[0][7:10],
                                                            target._first_order_values[1][7:10],
                                                            target._second_order_values[0][7:10],
                                                            target._second_order_values[1][7:10],
                                                            end_time)
                trans_coeff = {'left': trans_coeff, 'right': trans_coeff1}
            else:
                trans_coeff = {'single': trans_coeff} 
        elif self._interpolation_type == "trapezoidal":
            raise NotImplementedError
            # self._plan_trapezoidal_profile(translation_target_posi, [finish_time])
            
        return trans_coeff, slerp
        
    def _eval_profile(self, trans_profile, rot_profile, t):
        # trans
        if not self._enable_motion:
            curr_trans = self._eval_polynomial(trans_profile, t, 0)
        else:
            curr_trans = self._get_traj_position(trans_profile, t)
        
        # rotation slerp
        curr_quat = rot_profile(t).as_quat()
        curr_point = np.hstack((curr_trans, curr_quat))
        return curr_point
    
    def _get_traj_position(self, trans_profile, t):
        curr_posi = self._eval_polynomial(trans_profile, t, 0)
        curr_vel = self._eval_polynomial(trans_profile, t, 1)
        curr_acc = self._eval_polynomial(trans_profile, t, 2)
        curr_posi += curr_vel * self.dt + 0.5 * curr_acc * (self.dt**2)
        return curr_posi
        
if __name__ == '__main__':
    import yaml
    import os
    config = None
    cur_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cur_path, "config", "cartesian_polynomial_traj_cfg.yaml")
    logging.info(f'cfg file name: {cfg_file}')
    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)
    
    buffer = Buffer(config["buffer"]["size"], config["buffer"]["dim"])
    lock = threading.Lock()
    cart_traj = CartessianTrajectory(config["cart_polynomial"], buffer, lock)
    
    start = [0, 1, 2, 0, 0, 0, 1]
    end = [2, 2, 3, 0.707, 0.707, 0, 0]
    
    target = TrajectoryState()
    target._zero_order_values = np.vstack((start, end))
    logging.info(f'posi: {target._zero_order_values}')
    target._first_order_values = np.zeros((2,7))
    target._second_order_values = np.zeros((2,7))
    
    cart_traj.plan_trajectory(target)
    logging.info(f"Trajectory done!!!, buffer size: {buffer.size()}")
    logging.info(f'Traj final data: {cart_traj._buffer._data[-1]}')
    
    
    duo_buffer = Buffer(config["duo_buffer"]["size"], config["duo_buffer"]["dim"])
    duo_traj = CartessianTrajectory(config["cart_polynomial"], duo_buffer, lock)
    start_r = [2, 2, 3, 0.707, 0.707, 0, 0]
    end_r = [0, 1, 2, 0, 0, 0, 1]
    start = np.hstack((start, start_r))
    end = np.hstack((end, end_r))
    target._zero_order_values = np.vstack((start, end))
    logging.info(f'posi: {target._zero_order_values}')
    target._first_order_values = np.zeros((2,14))
    target._second_order_values = np.zeros((2,14))
    duo_traj.plan_trajectory(target)
    logging.info(f"duo Trajectory done!!!, buffer size: {duo_buffer.size()}")
    logging.info(f'duo Traj final data: {duo_traj._buffer._data[-1]}')
    