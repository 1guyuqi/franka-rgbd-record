from __future__ import annotations

import os
import time
import yaml
import numpy as np
import threading

from panda_py import Panda
from panda_py.controllers import JointPosition

from repo_paths import CONFIG_DIR
from utils.utils import Buffer
from utils.trajectory_state import TrajectoryState
from joint_trajectory import JointTrajectory  

ROBOT_IP = os.environ.get("FRANKA_IP", "172.16.0.1")
JOINT_CFG = CONFIG_DIR / "joint_polynomial_traj_cfg.yaml"

q_target = np.array([3.07242423e-01,  4.99082031e-04,  4.88018274e-01,  9.99998562e-01,
 -4.78346824e-04,  1.53620648e-03, -5.36876634e-04], dtype=np.float64)


def main():
    with open(JOINT_CFG, "r") as f:
        config = yaml.safe_load(f)

    buffer = Buffer(config["buffer"]["size"], config["buffer"]["dim"])
    lock = threading.Lock()
    joint_traj = JointTrajectory(config["joint_polynomial"], buffer, lock)

    panda = Panda(ROBOT_IP)
    panda.move_to_start()
    try:
        panda.set_speed_factor(0.2) 
    except Exception:
        pass

    q_start = np.array(panda.get_state().q, dtype=np.float64)

    target = TrajectoryState()
    target._zero_order_values = np.vstack((q_start, q_target))
    target._first_order_values = np.zeros((2, 7))
    target._second_order_values = np.zeros((2, 7))

    end_time = joint_traj._auto_generate_joint_end_time(
        q_start, q_target, user_specified_time=5
    )
    print(f"[INFO] end_time = {end_time}")
    if end_time < 0:
        raise RuntimeError("end_time < 0, trajectory rejected")

    if joint_traj._interpolation_type != "quintic":
        raise RuntimeError(f"Your config interpolation_type={joint_traj._interpolation_type}, "
                           f"but only quintic is implemented in this file.")

    coeff = joint_traj._generate_traj_profile(target, end_time)

    ctrl = JointPosition(filter_coeff=1.0)
    panda.start_controller(ctrl)

    dt = float(joint_traj.dt)
    dt = max(dt, 0.001) 

    try:
        t0 = time.perf_counter()
        next_tick = t0

        while True:
            now = time.perf_counter()
            t = now - t0
            if t >= end_time:
                break

            q = joint_traj._eval_polynomial(coeff, t, 0).astype(np.float64)
            dq = joint_traj._eval_polynomial(coeff, t, 1).astype(np.float64)

            ctrl.set_control(q, dq)

            next_tick += dt
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

        for _ in range(int(0.2 / dt)):
            ctrl.set_control(q_target, np.zeros(7, dtype=np.float64))
            time.sleep(dt)

    finally:
        panda.stop_controller()

    q_now = np.array(panda.get_state().q, dtype=np.float64)
    err = q_now - q_target
    print("\n[RESULT] q_now   =", q_now)
    print("[RESULT] q_target=", q_target)
    print("[RESULT] per-joint error =", err)
    print("[RESULT] L2 error =", float(np.linalg.norm(err)))
    print("[RESULT] max|error| =", float(np.max(np.abs(err))))


if __name__ == "__main__":
    main()