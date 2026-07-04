import numpy as np

class TrajectoryState:
    # size: [num_state, dim_traj]
    _zero_order_values: np.ndarray # position
    _first_order_values: np.ndarray # vel
    _second_order_values: np.ndarray # acc

