from __future__ import annotations

from trajectory_base import TrajectoryBase
import numpy as np
from utils.trajectory_state import TrajectoryState
import warnings
from utils.utils import Buffer, check_traj_size
import threading
import time



class JointTrajectory(TrajectoryBase):
    def __init__(self, config, buffer: Buffer, lock: threading.Lock):
        super().__init__(config, buffer, lock)
        self.trajectory_idle = True
        self._enable_motion = config.get("enable_motion", False)
        self._max_acceleration = config.get("max_acceleration", 5.0)
        # if buffer dim is 7, single arm; 14 for dual arm
        self.is_duo_arm = True if buffer._dim == 14 else False
        
    def plan_trajectory(self, target: TrajectoryState, finish_time: float | None = None):
        if not self.trajectory_idle:
            print(f'not idle, busy!!!')
            return
        self.trajectory_idle = False
        
        # interpolation check
        if not self._interpolation_type in ["cubic", "quintic", "trapezoidal"]:
            raise ValueError(f"{self._interpolation_type} interpolation " 
                             "is not supported!!!")
        
        flag = check_traj_size(target, self._buffer._dim)
        if not flag:
            warnings.warn("target dim is different with buffer dim!!!")
            self.trajectory_idle = True
            return
        
        finish_time = -1 if finish_time is None else finish_time
        end_time = self._auto_generate_joint_end_time(
            target._zero_order_values[0], 
            target._zero_order_values[1],
            finish_time
        )
        
        print(f'end time: {end_time}')
        if end_time < 0:
            self.trajectory_idle = True
            return 
        
        # profile generation
        joint_coeff = self._generate_traj_profile(target, end_time)
        
        cur_time = 0.0 
        while cur_time < end_time:
            # eval profile given cur t
            curr_point = self._eval_profile(joint_coeff, cur_time)
            
            # update buffer
            self._buffer_lock.acquire()
            self._buffer.push_data(curr_point)
            self._buffer_lock.release()
            
            cur_time += self.dt
            time.sleep(0.85*self.dt)
            
        # Add final point at end_time
        curr_point = self._eval_profile(joint_coeff, end_time)
        self._buffer_lock.acquire()
        self._buffer.push_data(curr_point)
        self._buffer_lock.release()
            
        self.trajectory_idle = True
        
    def _generate_traj_profile(self, target: TrajectoryState, end_time: float):
        joint_coeff = None
        
        if self._interpolation_type == "cubic":
            # Use cubic spline from base class
            waypoints = target._zero_order_values
            velocities = target._first_order_values
            _, positions, _, _ = self._plan_cubic_spline(
                waypoints, [0, end_time], velocities
            )
            raise NotImplementedError("Cubic spline needs coefficient extraction")
            
        elif self._interpolation_type == "quintic":
            joint_coeff = self._solve_quintic_coefficients(
                target._zero_order_values[0],
                target._zero_order_values[1],
                target._first_order_values[0],
                target._first_order_values[1],
                target._second_order_values[0],
                target._second_order_values[1],
                end_time
            )
            
        elif self._interpolation_type == "trapezoidal":
            raise NotImplementedError("Trapezoidal profile not implemented yet")
            
        return joint_coeff
        
    def _eval_profile(self, joint_coeff: np.ndarray, t: float) -> np.ndarray:
        if not self._enable_motion:
            curr_joints = self._eval_polynomial(joint_coeff, t, 0)
        else:
            curr_joints = self._get_traj_position(joint_coeff, t)
        
        return curr_joints
    
    def _get_traj_position(self, joint_coeff: np.ndarray, t: float) -> np.ndarray:
        curr_posi = self._eval_polynomial(joint_coeff, t, 0)
        curr_vel = self._eval_polynomial(joint_coeff, t, 1)
        curr_acc = self._eval_polynomial(joint_coeff, t, 2)
        curr_posi += curr_vel * self.dt + 0.5 * curr_acc * (self.dt**2)
        return curr_posi
    
    def _auto_generate_joint_end_time(self, start: np.ndarray, end: np.ndarray, 
                                     user_specified_time: float) -> float:
        """
        Auto-generate time based on joint angle differences and velocity limits.
        """
        angle_diff = np.abs(end - start)
        
        # Time based on max velocity for each joint
        vel_times = angle_diff / self._max_vel
        
        # Time based on acceleration limits (assuming triangular profile)
        acc_times = 2 * np.sqrt(angle_diff / self._max_acceleration)
        
        # Take the maximum time required
        auto_time = max(np.max(vel_times), np.max(acc_times))
        
        finish_time = max(auto_time, user_specified_time)
        
        if finish_time < 0.1 * self.dt:
            finish_time = -1
            
        return finish_time


if __name__ == '__main__':
    import yaml
    import os
    
    # Load config
    config = None
    cur_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cur_path, "config", "joint_polynomial_traj_cfg.yaml")
    print(f'cfg file name: {cfg_file}')
    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)
    
    # Test single arm trajectory
    print("\n=== Testing Single Arm Joint Trajectory ===")
    buffer = Buffer(config["buffer"]["size"], config["buffer"]["dim"])
    lock = threading.Lock()
    joint_traj = JointTrajectory(config["joint_polynomial"], buffer, lock)
    
    # Define joint angles (radians)
    start_joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # Home position
    end_joints = np.array([0.5, -0.5, 0.3, -2.0, -0.2, 1.8, 1.0])  # Target position
    
    target = TrajectoryState()
    target._zero_order_values = np.vstack((start_joints, end_joints))
    print(f'Joint positions: \n{target._zero_order_values}')
    target._first_order_values = np.zeros((2, 7))
    target._second_order_values = np.zeros((2, 7))
    
    joint_traj.plan_trajectory(target)
    print(f"Trajectory done! Buffer size: {buffer.size()}")
    print(f'Final joint angles: {joint_traj._buffer._data[-1]}')
    print(f'Expected final: {end_joints}')
    print(f'Error: {np.linalg.norm(joint_traj._buffer._data[-1] - end_joints):.6f}')
    
    # Test dual arm trajectory
    print("\n=== Testing Dual Arm Joint Trajectory ===")
    duo_buffer = Buffer(config["duo_buffer"]["size"], config["duo_buffer"]["dim"])
    duo_traj = JointTrajectory(config["joint_polynomial"], duo_buffer, lock)
    
    # Right arm joints
    start_joints_r = np.array([0.0, 0.785, 0.0, -2.356, 0.0, 1.571, -0.785])
    end_joints_r = np.array([-0.5, 0.5, -0.3, -2.0, 0.2, 1.8, -1.0])
    
    # Combine left and right arms
    start_all = np.hstack((start_joints, start_joints_r))
    end_all = np.hstack((end_joints, end_joints_r))
    
    target._zero_order_values = np.vstack((start_all, end_all))
    print(f'Dual arm positions: \n{target._zero_order_values}')
    target._first_order_values = np.zeros((2, 14))
    target._second_order_values = np.zeros((2, 14))
    
    duo_traj.plan_trajectory(target)
    print(f"Dual trajectory done! Buffer size: {duo_buffer.size()}")
    print(f'Final joint angles: {duo_traj._buffer._data[-1]}')
    print(f'Expected final: {end_all}')
    print(f'Error: {np.linalg.norm(duo_traj._buffer._data[-1] - end_all):.6f}')