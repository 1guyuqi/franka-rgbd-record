"""
Smoother Base Class
Provides abstract interface for joint command smoothers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np


class SmootherBase(ABC):
    """
    Base class for joint-command smoothers.
    Supports second-order filters, Ruckig, and other smoothing backends.
    """
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any], dof: int):
        """
        Initialize the smoother.

        Args:
            config: Smoother configuration dictionary.
            dof: Robot degrees of freedom.
        """
        self._config = config
        self._dof = dof
        self._is_running = False
    
    @abstractmethod
    def start(self, initial_positions: np.ndarray) -> None:
        """
        Start the smoother.

        Args:
            initial_positions: shape=(dof,) initial joint positions.

        Raises:
            RuntimeError: If the smoother is already running.
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the smoother thread."""
        pass
    
    @abstractmethod
    def update_target(self, joint_target: np.ndarray, immediate: bool = False) -> None:
        """
        Update the target joint positions.

        Args:
            joint_target: shape=(dof,) target joint angles.
            immediate: Jump immediately to the target (e.g. after reset).

        Returns:
            None (non-blocking).
        """
        pass
    
    @abstractmethod
    def get_command(self) -> Tuple[np.ndarray, bool]:
        """
        Get the smoothed command.

        Returns:
            (joint_positions, is_active): smoothed joints and active flag.
        """
        pass
    
    @abstractmethod
    def pause(self) -> None:
        """Pause smoothing while holding the current output."""
        pass
    
    @abstractmethod
    def resume(self, sync_to_current: bool = True) -> None:
        """
        Resume smoothing.

        Args:
            sync_to_current: Sync internal state to the current measured pose.
        """
        pass
    
    # === Optional Ruckig-specific hooks ===
    
    def set_velocity_limits(self, max_velocity: np.ndarray) -> None:
        """
        Set velocity limits (required by Ruckig).

        Args:
            max_velocity: shape=(dof,) maximum velocity (rad/s).
        """
        pass
    
    def set_acceleration_limits(self, max_acceleration: np.ndarray) -> None:
        """
        Set acceleration limits (required by Ruckig).

        Args:
            max_acceleration: shape=(dof,) maximum acceleration (rad/s^2).
        """
        pass
    
    def set_jerk_limits(self, max_jerk: np.ndarray) -> None:
        """
        Set jerk limits (Ruckig-specific).

        Args:
            max_jerk: shape=(dof,) maximum jerk (rad/s^3).
        """
        pass
    
    def get_motion_state(self) -> Dict[str, np.ndarray]:
        """
        Get the full motion state.

        Returns:
            Dict with position, velocity, and acceleration arrays.
        """
        return {
            'position': np.zeros(self._dof),
            'velocity': np.zeros(self._dof),
            'acceleration': np.zeros(self._dof)
        }
    
    def get_expected_duration(self) -> float:
        """
        Get expected time to reach the target (for trajectory planning).

        Returns:
            Expected duration in seconds.
        """
        return 0.0
    
    def is_trajectory_finished(self, tolerance: float = 0.001) -> bool:
        """
        Check whether the target has been reached.

        Args:
            tolerance: Position tolerance (rad).

        Returns:
            True if the trajectory is finished.
        """
        return True
