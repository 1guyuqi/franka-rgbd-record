"""
Ruckig-based S-curve trajectory smoother
Provides jerk-limited smooth trajectories using Ruckig library
"""

import numpy as np
import threading
import time,copy
import glog as log
from typing import Dict, Any, Tuple, Optional

try:
    from ruckig import Ruckig, InputParameter, OutputParameter, Result, Synchronization
except ImportError:
    raise ImportError("Ruckig library not found. Install with: pip install ruckig")

from .smoother_base import SmootherBase

class RuckigSmoother(SmootherBase):
    """
    Ruckig-based S-curve trajectory smoother with jerk limiting
    
    Provides optimal time trajectories with bounded jerk, acceleration, and velocity.
    Supports online trajectory generation and immediate target updates.
    """
    
    def __init__(self, config: Dict[str, Any], dof: int):
        """
        Initialize Ruckig smoother
        
        Args:
            config: Configuration dict containing:
                - control_frequency: Control loop frequency (Hz)
                - max_velocity: Maximum joint velocities (rad/s)
                - max_acceleration: Maximum joint accelerations (rad/s²)
                - max_jerk: Maximum joint jerks (rad/s³)
                - min_position: Minimum joint positions (rad)
                - max_position: Maximum joint positions (rad)
                - synchronization: Time synchronization mode
            dof: Degrees of freedom
        """
        super().__init__(config, dof)
        
        # Control parameters
        self._control_freq = config.get("control_frequency", 800.0)
        self._dt = 1.0 / self._control_freq
        
        # Ruckig instances
        self._ruckig = Ruckig(dof, self._dt)
        self._input = InputParameter(dof)
        self._output = OutputParameter(dof)
        
        # Dynamic limits (use config or defaults)
        self._max_velocity = np.array(config.get("max_velocity", [2.0] * dof))
        self._max_acceleration = np.array(config.get("max_acceleration", [10.0] * dof))
        self._max_jerk = np.array(config.get("max_jerk", [50.0] * dof))
        
        # Position limits (optional)
        if "min_position" in config and "max_position" in config:
            self._min_position = np.array(config["min_position"])
            self._max_position = np.array(config["max_position"])
            self._use_position_limits = True
        else:
            self._use_position_limits = False
        
        # Synchronization mode
        self._synchronization = config.get("synchronization", True)
        self._sync_mode = config.get("sync_mode", "phase")  # "phase", "time", or "none"
        
        # State variables
        self._current_position = np.zeros(dof)
        self._current_velocity = np.zeros(dof)
        self._current_acceleration = np.zeros(dof)
        self._target_position = np.zeros(dof)
        self._target_velocity = np.zeros(dof)  # Usually zero for position control
        self._target_acceleration = np.zeros(dof)  # Usually zero
        
        # Thread control
        self._lock = threading.Lock()
        self._output_lock = threading.Lock()
        self._pause_flag = False
        self._thread = None
        self._target_updated = False
        
        # Performance monitoring
        self._slow_loop_count = 0
        self._last_result = Result.Working
        self.max_time = 0

        # Apply initial configuration
        self._apply_limits()
        
        log.info(f"RuckigSmoother initialized: "
                   f"DOF={dof}, "
                   f"freq={self._control_freq}Hz, "
                   f"sync={'on' if self._synchronization else 'off'}, "
                   f"sync_mode={self._sync_mode}")
    
    def _apply_limits(self) -> None:
        """Apply dynamic and position limits to Ruckig input"""
        self._input.max_velocity = self._max_velocity.tolist()
        self._input.max_acceleration = self._max_acceleration.tolist()
        self._input.max_jerk = self._max_jerk.tolist()
        
        if self._use_position_limits:
            self._input.min_position = self._min_position.tolist()
            self._input.max_position = self._max_position.tolist()
        
        # Set synchronization based on config
        if not self._synchronization or self._sync_mode.lower() == "none":
            # No synchronization, each joint optimizes independently
            self._input.synchronization = Synchronization.No
            actual_sync = "No"
        elif self._sync_mode.lower() == "time":
            # Time synchronization - all joints finish at same time
            self._input.synchronization = Synchronization.Time
            actual_sync = "Time"
        elif self._sync_mode.lower() == "phase":
            # Phase synchronization - synchronize velocity profiles
            self._input.synchronization = Synchronization.Phase
            actual_sync = "Phase"
        else:
            # Default to phase synchronization
            log.warn(f"Unknown sync_mode '{self._sync_mode}', using Phase synchronization")
            self._input.synchronization = Synchronization.Phase
            actual_sync = "Phase (default)"
        
        # Log the actual synchronization mode set
        log.info(f"Ruckig synchronization set to: {actual_sync} (value={self._input.synchronization})")
    
    def start(self, initial_positions: np.ndarray) -> None:
        """
        Start smoother thread
        
        Args:
            initial_positions: Initial joint positions (rad)
        
        Raises:
            RuntimeError: If smoother already running
        """
        if self._is_running:
            raise RuntimeError("Smoother already running")
        
        assert initial_positions.shape == (self._dof,), \
            f"Expected shape ({self._dof},), got {initial_positions.shape}"
        
        # Initialize state
        with self._lock:
            self._current_position = initial_positions.copy()
            self._target_position = initial_positions.copy()
            
            # Initialize Ruckig input
            self._input.current_position = self._current_position.tolist()
            self._input.current_velocity = [0.0] * self._dof
            self._input.current_acceleration = [0.0] * self._dof
            
            self._input.target_position = self._target_position.tolist()
            self._input.target_velocity = [0.0] * self._dof
            self._input.target_acceleration = [0.0] * self._dof
            
            # Reset internal state
            self._current_velocity = np.zeros(self._dof)
            self._current_acceleration = np.zeros(self._dof)
            self._last_result = Result.Working
        
        # Start control thread
        self._is_running = True
        self._thread = threading.Thread(
            target=self._control_loop,
            daemon=True,
            name="RuckigSmootherLoop"
        )
        self._thread.start()
        log.info(f"Ruckig smoother thread started at {self._control_freq}Hz")
    
    def stop(self) -> None:
        """Stop smoother thread"""
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                log.warn("Ruckig smoother thread did not stop cleanly")
        log.info("Ruckig smoother stopped")
    
    def update_target(self, joint_target: np.ndarray, immediate: bool = False) -> None:
        """
        Update target joint positions
        
        Args:
            joint_target: Target joint positions (rad)
            immediate: If True, immediately jump to target (for reset)
        """
        assert joint_target.shape == (self._dof,), \
            f"Expected shape ({self._dof},), got {joint_target.shape}"
        
        with self._lock:
            self._target_position = joint_target.copy()
            
        if immediate:
            # Immediate jump mode - reset everything
            self._current_position = self._target_position.copy()
            self._current_velocity = np.zeros(self._dof)
            self._current_acceleration = np.zeros(self._dof)
            
            # Update Ruckig state
            self._input.current_position = self._current_position.tolist()
            self._input.current_velocity = [0.0] * self._dof
            self._input.current_acceleration = [0.0] * self._dof
            self._input.target_position = self._target_position.tolist()
            
            log.debug("Immediate jump to target")
        else:
            # Normal update - just set new target
            self._input.target_position = self._target_position.tolist()
            self._target_updated = True
    
    def get_command(self) -> Tuple[np.ndarray, bool]:
        """
        Get current smoothed joint command
        
        Returns:
            (joint_positions, is_active): Current positions and active flag
        """
        start = time.perf_counter()
        with self._output_lock:
            command = copy.deepcopy(self._current_position)
        used_time = (time.perf_counter() - start)*1000
        if used_time > self.max_time:
            self.max_time = used_time
            # log.info(f'max time get command {self.max_time:.2f}ms')
        need_pause = not self._pause_flag
        return command, need_pause
    
    def pause(self) -> None:
        """Pause smoother (maintains current output)"""
        with self._lock:
            self._pause_flag = True
        log.debug("Ruckig smoother paused")
    
    def resume(self, sync_to_current: bool = True) -> None:
        """
        Resume smoother after pause
        
        Args:
            sync_to_current: If True, sync target to current position
        """
        with self._lock:
            if sync_to_current:
                # Sync target to current position
                self._target_position = self._current_position.copy()
                self._input.target_position = self._target_position.tolist()
                # Keep current velocities and accelerations
            self._pause_flag = False
        log.debug("Ruckig smoother resumed")
    
    def set_velocity_limits(self, max_velocity: np.ndarray) -> None:
        """
        Update velocity limits
        
        Args:
            max_velocity: Maximum velocities per joint (rad/s)
        """
        assert max_velocity.shape == (self._dof,), \
            f"Expected shape ({self._dof},), got {max_velocity.shape}"
        
        with self._lock:
            self._max_velocity = np.clip(max_velocity, 0.1, 10.0)
            self._input.max_velocity = self._max_velocity.tolist()
        log.info(f"Velocity limits updated: {self._max_velocity}")
    
    def set_acceleration_limits(self, max_acceleration: np.ndarray) -> None:
        """
        Update acceleration limits
        
        Args:
            max_acceleration: Maximum accelerations per joint (rad/s²)
        """
        assert max_acceleration.shape == (self._dof,), \
            f"Expected shape ({self._dof},), got {max_acceleration.shape}"
        
        with self._lock:
            self._max_acceleration = np.clip(max_acceleration, 0.5, 50.0)
            self._input.max_acceleration = self._max_acceleration.tolist()
        log.info(f"Acceleration limits updated: {self._max_acceleration}")
    
    def set_jerk_limits(self, max_jerk: np.ndarray) -> None:
        """
        Update jerk limits
        
        Args:
            max_jerk: Maximum jerks per joint (rad/s³)
        """
        assert max_jerk.shape == (self._dof,), \
            f"Expected shape ({self._dof},), got {max_jerk.shape}"
        
        with self._lock:
            self._max_jerk = np.clip(max_jerk, 1.0, 500.0)
            self._input.max_jerk = self._max_jerk.tolist()
        log.info(f"Jerk limits updated: {self._max_jerk}")
    
    def get_motion_state(self) -> Dict[str, np.ndarray]:
        """
        Get complete motion state
        
        Returns:
            Dict with position, velocity, acceleration arrays
        """
        with self._lock:
            return {
                'position': self._current_position.copy(),
                'velocity': self._current_velocity.copy(),
                'acceleration': self._current_acceleration.copy()
            }
    
    def get_expected_duration(self) -> float:
        """
        Get expected time to reach target
        
        Returns:
            Expected duration in seconds
        """
        with self._lock:
            if hasattr(self._output, 'trajectory') and self._output.trajectory:
                return self._output.trajectory.duration
        return 0.0
    
    def is_trajectory_finished(self, tolerance: float = 0.001) -> bool:
        """
        Check if trajectory reached target
        
        Args:
            tolerance: Position tolerance (rad)
        
        Returns:
            True if within tolerance of target
        """
        with self._lock:
            if self._last_result == Result.Finished:
                return True
            
            error = np.linalg.norm(self._target_position - self._current_position)
            velocity = np.linalg.norm(self._current_velocity)
            return error < tolerance and velocity < tolerance
    
    def _control_loop(self) -> None:
        """Main control loop running in separate thread"""
        last_time = time.perf_counter()
        while self._is_running:
            loop_start = time.perf_counter()
            
            with self._lock:
                if self._pause_flag:
                    time.sleep(self._dt)
                    continue
                
            # Update Ruckig input with current state
            # Use saved acceleration from previous cycle
            self._input.current_position = self._current_position.tolist()
            self._input.current_velocity = self._current_velocity.tolist()
            self._input.current_acceleration = self._current_acceleration.tolist()
            
            # Compute next trajectory step (outside lock for performance)
            try:
                result = self._ruckig.update(self._input, self._output)
                
                # Update state with new trajectory point
                with self._output_lock:
                    self._current_position = np.array(self._output.new_position)
                    self._last_result = result
                self._current_velocity = np.array(self._output.new_velocity)
                self._current_acceleration = np.array(self._output.new_acceleration)

                with self._lock:
                    self._target_position = np.array(self._output.new_position)
                    
                # Pass output to input for next cycle
                self._output.pass_to_input(self._input)
                
                # Log trajectory completion
                if result == Result.Finished and self._target_updated:
                    log.debug(f"Trajectory finished in {self._output.trajectory.duration:.3f}s")
                    self._target_updated = False
                        
            except Exception as e:
                log.error(f"Ruckig update failed: {e}")
                # Keep current state on error
            
            # Timing management
            used_time = time.perf_counter() - loop_start
            # last_time = time.perf_counter()
            if used_time < self._dt:
                sleep_time = self._dt - used_time
                time.sleep(sleep_time)
            elif used_time > 1.2 * self._dt:
                # Performance warning
                self._slow_loop_count += 1
                if self._slow_loop_count % 300 == 0:
                    actual_dt = time.perf_counter() - loop_start
                    log.warn(f"Ruckig loop slow: {actual_dt*1000:.1f}ms "
                                 f"(target: {self._dt*1000:.1f}ms)")
                    