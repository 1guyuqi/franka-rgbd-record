from __future__ import annotations

import os
import time
import yaml
import numpy as np
import threading

from panda_py import Panda
from panda_py.controllers import CartesianImpedance
from scipy.spatial.transform import Rotation as R

from repo_paths import CONFIG_DIR
from utils.utils import Buffer
from utils.trajectory_state import TrajectoryState
from cartesian_trajectory import CartessianTrajectory
from utils.ik import GaussianNewtonIK 

ROBOT_IP = os.environ.get("FRANKA_IP", "172.16.0.1")
CART_CFG = CONFIG_DIR / "cartesian_polynomial_traj_cfg.yaml"

# Target position in base frame (meters)
POS_TARGET = np.array([0.6, 0.2, 0.2], dtype=np.float64)

# Hold at the end of the nominal trajectory (seconds)
HOLD_TIME_S = 2.0


# OUTER-LOOP POSITION SERVO 

OUTER_MAX_ITERS = 30         # max outer iterations
POS_TOL_M = 0.0010           # stop when ||pos_err|| < 1 mm
STEP_MAX_M = 0.0030          # max command increment per iteration (3 mm)
OUTER_GAIN = 0.8             # 0~1, how aggressively to compensate error
OUTER_SETTLE_S = 0.25        # how long to hold each updated command (seconds)

def _rotate_z_axis(quat: np.ndarray) -> np.ndarray:
    """
    Given a quaternion, calculate the Z-axis direction of the end effector.
    """
    # Use scipy's Rotation to convert quaternion to rotation matrix
    r = R.from_quat(quat)
    rotation_matrix = r.as_matrix()
    return rotation_matrix[:, 2]  # Extract Z-axis direction from the rotation matrix

def move_up(pose: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """
    Move the position N cm along the end effector's Z axis.
    """
    # Get the Z-axis direction from the quaternion
    z_axis = _rotate_z_axis(quat)
    
    # Move along the Z-axis
    new_position = pose[:3] - 0.1 * z_axis  # 10cm along the Z-axis direction
    new_pose = np.hstack([new_position, pose[3:]])  # Keep the orientation the same
    return new_pose

def get_current_pose7(panda: Panda) -> np.ndarray:
    p = np.asarray(panda.get_position(), dtype=np.float64).reshape(3)
    q = np.asarray(panda.get_orientation(), dtype=np.float64).reshape(4)  # assume xyzw
    return np.hstack([p, q])


def _clip_step(vec3: np.ndarray, step_max: float) -> np.ndarray:
    n = float(np.linalg.norm(vec3))
    if n <= step_max or n <= 1e-12:
        return vec3
    return vec3 / n * step_max


def main():
    with open(CART_CFG, "r") as f:
        config = yaml.safe_load(f)

    buffer = Buffer(config["buffer"]["size"], config["buffer"]["dim"])  # dim=7
    lock = threading.Lock()
    cart_traj = CartessianTrajectory(config["cart_polynomial"], buffer, lock)

    panda = Panda(ROBOT_IP)
    panda.move_to_start()
    try:
        panda.set_speed_factor(0.2)
    except Exception:
        pass

    pose_start = get_current_pose7(panda)

    pose_target = pose_start.copy()
    pose_target[:3] = POS_TARGET
    pose_target[3:7] = pose_start[3:7]  # keep orientation
    pose_target = move_up(pose_target, pose_target[3:7])  # Move up by 10 cm
    target = TrajectoryState()
    target._zero_order_values = np.vstack((pose_start, pose_target))
    target._first_order_values = np.zeros((2, 7), dtype=np.float64)
    target._second_order_values = np.zeros((2, 7), dtype=np.float64)

    end_time = cart_traj._auto_generate_end_time(pose_start, pose_target, user_specified_time=5)
    print(f"[INFO] end_time = {end_time:.6f}")
    if end_time < 0:
        raise RuntimeError("end_time < 0, trajectory rejected")

    if cart_traj._interpolation_type != "quintic":
        raise RuntimeError(
            f"Your config interpolation_type={cart_traj._interpolation_type}, "
            f"but this trajectory class only implements quintic translation."
        )

    trans_coeff, slerp = cart_traj._generate_traj_profile(target, end_time)
    print(f"trans_coeff['single']: {trans_coeff['single']}")
    print(f"Shape of trans_coeff['single']: {trans_coeff['single'].shape}")
    ctrl = CartesianImpedance(filter_coeff=1.0)
    panda.start_controller(ctrl)

    dt = float(cart_traj.dt)
    dt = max(dt, 0.001)

    ik_solver = GaussianNewtonIK()  # Initialize IK solver

    try:
        # -------------------------
        # 1) Nominal trajectory
        # -------------------------
        t0 = time.perf_counter()
        next_tick = t0

        while True:
            now = time.perf_counter()
            t = now - t0
            if t >= end_time:
                break

            pose7 = cart_traj._eval_profile(trans_coeff["single"], slerp["single"], t).astype(np.float64)
            pos = pose7[:3]
            quat = pose7[3:7]

            ctrl.set_control(pos, quat)

            next_tick += dt
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

        # -------------------------
        # 2) Initial hold at pose_target
        # -------------------------
        quat_cmd = pose_target[3:7].copy()
        pos_cmd = pose_target[:3].copy()

        # -------------------------
        # 3) OUTER-LOOP ITERATIVE CONVERGENCE (NO IK, Iterative Position Update)
        # -------------------------
        settle_n = int(max(1, round(OUTER_SETTLE_S / dt)))

        for i in range(int(OUTER_MAX_ITERS)):
            pose_now = get_current_pose7(panda)
            pos_now = pose_now[:3]

            e = pose_target[:3] - pos_now
            err_norm = float(np.linalg.norm(e))

            print(f"[OUTER] iter={i:02d}  err_norm={err_norm:.6f} m  "
                  f"e=[{e[0]:+.6f},{e[1]:+.6f},{e[2]:+.6f}]")

            if not np.isfinite(err_norm):
                raise RuntimeError("Non-finite position error encountered.")

            if err_norm <= float(POS_TOL_M):
                print(f"[OUTER] Converged: err_norm <= {POS_TOL_M} m")
                break

            # Compute a small correction step (bounded)
            step = OUTER_GAIN * e
            step = _clip_step(step, float(STEP_MAX_M))

            # Integrate into command (this is what cancels steady-state bias)
            pos_cmd = pos_cmd + step

            # Hold the updated command for a short settle time
            for _ in range(settle_n):
                ctrl.set_control(pos_cmd, quat_cmd)
                time.sleep(dt)

        # -------------------------
        # 4) Final Hold at pose_target
        # -------------------------
        final_hold_n = int(max(1, round(HOLD_TIME_S / dt)))
        for _ in range(final_hold_n):
            ctrl.set_control(pos_cmd, quat_cmd)
            time.sleep(dt)

    finally:
        panda.stop_controller()

    pose_now = get_current_pose7(panda)
    print("\n[RESULT] pose_now    =", pose_now)
    print("[RESULT] pose_target =", pose_target)
    print("[RESULT] pos error   =", pose_now[:3] - pose_target[:3])


if __name__ == "__main__":
    main()
