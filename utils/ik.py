# ik_gn_pinocchio.py
import os
import json
import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from threading import Lock
from typing import Optional, Tuple, Dict, Any

from repo_paths import DEFAULT_URDF, resolve_repo_path


DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config", "franka.json")
)
FRANKA_HOME_Q7 = np.array([0.0, -0.78539816339, 0.0, -2.35619449019, 0.0, 1.57079632679, 0.78539816339], dtype=float)

class _UrdfCache:
    """Thread-safe URDF model cache to avoid repeated parsing/build."""
    _lock = Lock()
    _cache: Dict[str, Tuple[pin.Model, Any, Any]] = {}

    @classmethod
    def load(cls, urdf_path: str):
        urdf_path = os.path.abspath(urdf_path)
        with cls._lock:
            if urdf_path in cls._cache:
                return cls._cache[urdf_path]
            package_dir = os.path.dirname(urdf_path)
            model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, package_dir)
            cls._cache[urdf_path] = (model, collision_model, visual_model)
            return cls._cache[urdf_path]


def _clip_finite(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Clip only finite joint limits (URDF may contain +/-inf)."""
    q = q.copy()
    m = np.isfinite(lo) & np.isfinite(hi)
    if np.any(m):
        q[m] = np.clip(q[m], lo[m], hi[m])
    return q


def _T_from_pos_quat_xyzw(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """pos(3,), quat(x,y,z,w) -> 4x4 homogeneous matrix."""
    pos = np.asarray(pos, dtype=float).reshape(3,)
    q = np.asarray(quat_xyzw, dtype=float).reshape(4,)
    x, y, z, w = q.tolist()
    R = pin.Quaternion(w, x, y, z).matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


@dataclass(frozen=True)
class GNConfig:
    """Static solver config (runtime targets are NOT included here)."""
    urdf: str = str(DEFAULT_URDF)
    ee_frame: str = "fr3_hand_tcp"
    tol: float = 1e-5
    max_iter: int = 500
    step_size: float = 1.0
    damping: float = 1e-6
    reference_frame: str = "LOCAL"  # "LOCAL" or "WORLD"
    verbose: bool = False


def load_or_create_config(path: str = DEFAULT_CONFIG_PATH) -> GNConfig:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(GNConfig().__dict__, f, ensure_ascii=False, indent=2)

    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    base = GNConfig()
    urdf = resolve_repo_path(str(d.get("urdf", base.urdf)))
    return GNConfig(
        urdf=str(urdf),
        ee_frame=str(d.get("ee_frame", base.ee_frame)),
        tol=float(d.get("tol", base.tol)),
        max_iter=int(d.get("max_iter", base.max_iter)),
        step_size=float(d.get("step_size", base.step_size)),
        damping=float(d.get("damping", base.damping)),
        reference_frame=str(d.get("reference_frame", base.reference_frame)).upper(),
        verbose=bool(d.get("verbose", base.verbose)),
    )


class GaussianNewtonIK:
    ...
    def __init__(self, cfg: Optional[GNConfig] = None, config_path: str = DEFAULT_CONFIG_PATH):
        self.cfg = cfg or load_or_create_config(config_path)

        self.model, _, _ = _UrdfCache.load(self.cfg.urdf)
        self.data = self.model.createData()

        if not self.model.existFrame(self.cfg.ee_frame):
            raise ValueError(f"EE frame '{self.cfg.ee_frame}' not found in model: {self.cfg.urdf}")

        self.ee_id = self.model.getFrameId(self.cfg.ee_frame)
        self.lo = self.model.lowerPositionLimit
        self.hi = self.model.upperPositionLimit

        # Default init (home pose padded to model.nq and clipped to limits).
        self.q_home = self._pack_q(FRANKA_HOME_Q7)

        if self.cfg.reference_frame == "LOCAL":
            self.ref = pin.LOCAL
        elif self.cfg.reference_frame == "WORLD":
            self.ref = pin.WORLD
        else:
            raise ValueError("reference_frame must be 'LOCAL' or 'WORLD'")

    def _pack_q(self, q_like: np.ndarray) -> np.ndarray:
        """Pad/trim to model.nq and clip to finite limits."""
        q_like = np.asarray(q_like, dtype=float).reshape(-1)
        q = np.zeros(self.model.nq, dtype=float)
        n = min(q.size, q_like.size)
        q[:n] = q_like[:n]
        return _clip_finite(q, self.lo, self.hi)
    
    def solve_T(self, target_T: np.ndarray, q0: Optional[np.ndarray] = None) -> Tuple[bool, np.ndarray, float]:
        if target_T.shape != (4, 4):
            raise ValueError("target_T must be 4x4")

        target = pin.SE3(target_T[:3, :3], target_T[:3, 3])

        # Init: use current_position if provided, otherwise use Franka home.
        q = self._pack_q(q0) if q0 is not None else self.q_home.copy()

        err_norm = float("inf")
        for it in range(self.cfg.max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            cur = self.data.oMf[self.ee_id]

            err = pin.log(cur.inverse() * target).vector
            err_norm = float(np.linalg.norm(err))
            if self.cfg.verbose:
                print(f"[GN] iter={it:03d} |err|={err_norm:.6e}")

            if err_norm < self.cfg.tol:
                return True, q, err_norm

            pin.computeJointJacobians(self.model, self.data, q)
            J = pin.getFrameJacobian(self.model, self.data, self.ee_id, self.ref)

            A = (J @ J.T) + self.cfg.damping * np.eye(6)
            dq = J.T @ np.linalg.solve(A, err)

            q = pin.integrate(self.model, q, self.cfg.step_size * dq)
            q = _clip_finite(q, self.lo, self.hi)

        return False, q, err_norm

    def solve(
        self,
        pos: np.ndarray,
        quat_xyzw: np.ndarray,
        current_position: Optional[np.ndarray] = None,
    ) -> Tuple[bool, np.ndarray, float]:
        return self.solve_T(_T_from_pos_quat_xyzw(pos, quat_xyzw), q0=current_position)    


if __name__ == "__main__":
    ik = GaussianNewtonIK()  # uses ./ik_gn_config.json (auto-created if missing)
    ok, q, err = ik.solve([
    0.6342867807034951,
    -0.09399095120436535,
    -0.0058887156320936995
  ], [
    1,
    0,
    0,
    0
  ])
    print(json.dumps({"ok": bool(ok), "final_err": float(err), "q": q.tolist()}, ensure_ascii=False))