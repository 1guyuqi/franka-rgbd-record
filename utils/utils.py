import numpy as np
import os
import open3d as o3d
import json
import cv2
import warnings
from collections import deque
from typing import Union, Optional, Tuple, List
from .trajectory_state import TrajectoryState
from scipy.spatial.transform import Rotation as R



def check_traj_size(traj: TrajectoryState, size: int) -> bool:
    if len(traj._zero_order_values[0]) != size:
        return False
    if len(traj._first_order_values[0]) != size:
        return False
    if len(traj._second_order_values[0]) != size:
        return False
    return True

class Buffer:
    _size: float
    _dim: float
    _data: deque
    _time_stamp: deque
    def __init__(self, size, dim):
        self._size = size
        self._dim = dim
        self._data = deque()
        self._time_stamp = deque()
        
    def push_data(self, data, stamp):
        if len(data) != self._dim:
            warnings.warn(f"The data dim: {len(data)} is not matched with buffer data dim {self._dim}")
            return False
        
        if len(self._data) == self._size:
            self.pop_data()
        self._data.append(data)
        self._time_stamp.append(stamp)
        return True
        
    def pop_data(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
            Pop the data from the buffer
            @returns
                bool for succeessfully poped or not
                np.array with size _dim
        """
        if len(self._data) == 0:
            return False, None, None
        
        poped_data = self._data.popleft()
        poped_stamp = self._time_stamp.popleft()
        return True, poped_data, poped_stamp
    
    def size(self):
        return len(self._data)
    
    def clear(self):
        self._data.clear()
        self._time_stamp.clear()
        
    def clear_outdated_data(self, cur_stamp):
        while True:
            success, data, stamp = self.pop_data()
            if not success: return
            criterion = (cur_stamp - stamp) < 0.01
            if criterion: return
            self.pop_data()


def quaternion_error(q1, q2):
    """
        @brief: compute the error between two quaternions
        @params:
            q1: the first quaternion
            q2: the second quaternion
        @return: the quaternion error
    """
    quat1 = R.from_quat(q1)
    quat2 = R.from_quat(q2)
    conjugate_quat2 = quat2.inv()
    quat_error = quat1 * conjugate_quat2
    return np.array(quat_error.as_quat())

def compute_pose_diff(pose1, pose2):
    """
        @brief: compute the difference between two poses (pose1 - pose2)
        @params:
            pose1 & pose2: format is in [x,y,z,qx,qy,qz,qw]
        @return: the 6D numpy array of two pose difference
    """
    diff = np.zeros(6)
    diff[:3] = pose1[:3] - pose2[:3]
    
    quat_error = quaternion_error(pose1[3:], pose2[3:])
    ori_error = np.array([quat_error[0], quat_error[1], quat_error[2]])
    ori_error = 2.0 * np.sign(quat_error[3]) * ori_error
    # @TODO: zyx: checking
    
    # norm = np.linalg.norm(ori_error)
    # if norm < 1e-15:
    #     ori_error = np.array([1, 0, 0])
    # else:
    #     ori_error = (1 / norm) * ori_error
    # angle = 2 * np.arctan2(norm, quat_error[3])
    # # angle = 2 * np.atan2(norm, quat_error[3])
    # if (angle > np.pi):
    #     angle -= 2 * np.pi
    # ori_error = angle * ori_error

    diff[3:] = ori_error
    return diff

def erode_mask(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask
    kernel = np.ones((k, k), dtype=np.uint8)
    out = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
    if out.sum() == 0:
        raise RuntimeError("mask became empty after erosion")
    return out


def depth_aligned_to_cloud(depth_u16, color_bgr, depth_scale, intr, zmin, zmax, max_points):
    fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)

    depth = depth_u16.astype(np.float32) * float(depth_scale)
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.reshape(-1)

    m = (z > float(zmin)) & (z < float(zmax)) & np.isfinite(z)
    u = u.reshape(-1)[m]
    v = v.reshape(-1)[m]
    z = z[m]

    x = (u - cx) / fx * z
    y = (v - cy) / fy * z

    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    cols = color_bgr.reshape(-1, 3)[m][:, ::-1].astype(np.float32) / 255.0  # RGB in [0,1]

    if int(max_points) > 0 and pts.shape[0] > int(max_points):
        idx = np.random.choice(pts.shape[0], int(max_points), replace=False)
        pts = pts[idx]
        cols = cols[idx]

    return pts, cols


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (pts @ T[:3, :3].T) + T[:3, 3]

def gripper_geom_to_points(gg_top1, sample_mesh_n: int = 1500) -> np.ndarray:
    """
    Convert top1 gripper Open3D geometry to a set of points (Nx3) for easy saving/inspection.
    Works for TriangleMesh / LineSet.
    """
    geoms = gg_top1.to_open3d_geometry_list()
    if not geoms:
        raise RuntimeError("to_open3d_geometry_list() returned empty for top1")

    g0 = geoms[0]

    if isinstance(g0, o3d.geometry.TriangleMesh):
        pcd = g0.sample_points_uniformly(number_of_points=int(sample_mesh_n))
        return np.asarray(pcd.points, dtype=np.float64)

    if isinstance(g0, o3d.geometry.LineSet):
        return np.asarray(g0.points, dtype=np.float64)

    # Fallback: try to read points attribute
    if hasattr(g0, "points"):
        return np.asarray(g0.points, dtype=np.float64)

    raise RuntimeError(f"Unsupported gripper geometry type: {type(g0)}")

def save_scene_ply_top1(
    out_dir: str,
    pts_cam: np.ndarray,
    cols_rgb01: np.ndarray,
    gg_top1,
    T_vis: np.ndarray,
    T_cam2base: np.ndarray | None = None,
    prefix: str = "scene",
):
    os.makedirs(out_dir, exist_ok=True)

    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    cols = np.asarray(cols_rgb01, dtype=np.float64)
    if cols.max() > 1.5:
        cols = cols / 255.0

    T_vis = np.asarray(T_vis, dtype=np.float64)
    if T_vis.shape != (4,4):
        raise ValueError(f"T_vis must be 4x4, got {T_vis.shape}")

    g_pts_cam = gripper_geom_to_points(gg_top1)

    # cam_vis
    pts_cam_vis = apply_T(T_vis, pts_cam)
    g_pts_cam_vis = apply_T(T_vis, g_pts_cam)

    g_cols = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (g_pts_cam_vis.shape[0], 1))

    scene_vis = o3d.geometry.PointCloud()
    scene_vis.points = o3d.utility.Vector3dVector(np.vstack([pts_cam_vis, g_pts_cam_vis]))
    scene_vis.colors = o3d.utility.Vector3dVector(np.vstack([cols, g_cols]))

    out_cam_vis = os.path.join(out_dir, f"{prefix}_cam_vis_top1.ply")
    o3d.io.write_point_cloud(out_cam_vis, scene_vis, write_ascii=False)
    print(f"[Saved] {out_cam_vis}")

    if T_cam2base is not None:
        T_cam2base = np.asarray(T_cam2base, dtype=np.float64)

        # base raw
        pts_base = apply_T(T_cam2base, pts_cam)
        g_pts_base = apply_T(T_cam2base, g_pts_cam)

        # base_vis (make it consistent with save_best_grasp)
        pts_base_vis = apply_T(T_vis, pts_base)
        g_pts_base_vis = apply_T(T_vis, g_pts_base)

        g_cols2 = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (g_pts_base_vis.shape[0], 1))

        scene_base_vis = o3d.geometry.PointCloud()
        scene_base_vis.points = o3d.utility.Vector3dVector(np.vstack([pts_base_vis, g_pts_base_vis]))
        scene_base_vis.colors = o3d.utility.Vector3dVector(np.vstack([cols, g_cols2]))

        out_base_vis = os.path.join(out_dir, f"{prefix}_base_vis_top1.ply")
        o3d.io.write_point_cloud(out_base_vis, scene_base_vis, write_ascii=False)
        print(f"[Saved] {out_base_vis}")

def grasp_to_matrix(g):
    if hasattr(g, "transformation_matrix"):
        T = np.asarray(g.transformation_matrix, dtype=np.float64)
        if T.shape == (4, 4):
            return T
    if hasattr(g, "rotation_matrix") and hasattr(g, "translation"):
        R = np.asarray(g.rotation_matrix, dtype=np.float64)
        t = np.asarray(g.translation, dtype=np.float64).reshape(3)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    raise RuntimeError("cannot convert grasp to 4x4 matrix")


def proj_uv(xyz_cam, fx, fy, cx, cy):
    x, y, z = float(xyz_cam[0]), float(xyz_cam[1]), float(xyz_cam[2])
    if (not np.isfinite(z)) or z <= 0:
        return None
    u = fx * x / z + cx
    v = fy * y / z + cy
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return int(round(u)), int(round(v))


def filter_grasps_by_mask(gg, mask, fx, fy, cx, cy):
    H, W = mask.shape
    keep = []
    for i in range(len(gg)):
        T = grasp_to_matrix(gg[i])
        uv = proj_uv(T[:3, 3], fx, fy, cx, cy)
        if uv is None:
            continue
        u, v = uv
        if 0 <= u < W and 0 <= v < H and bool(mask[v, u]):
            keep.append(i)

    if not keep:
        return gg[:0]

    try:
        return gg[keep]
    except Exception:
        return gg.__class__([gg[i] for i in keep])


def save_capture(out_dir, color_bgr, depth_u16, meta):
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "rgb.png"), color_bgr)
    cv2.imwrite(os.path.join(out_dir, "depth.png"), depth_u16)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def save_mask(out_dir, mask):
    cv2.imwrite(os.path.join(out_dir, "mask_obj.png"), (mask.astype(np.uint8) * 255))

def mirror_T(T: np.ndarray, S: np.ndarray) -> np.ndarray:
    # coordinate change by reflection: T' = S T S^{-1}, and S^{-1}=S
    return (S @ T @ S).astype(np.float64)

def save_best_grasp(out_dir, best_grasp, T_cam2base, meta):
    os.makedirs(out_dir, exist_ok=True)

    # Raw grasp pose from AnyGrasp (camera frame)
    T_cam_raw = grasp_to_matrix(best_grasp).astype(np.float64)
    T_base_raw = (np.asarray(T_cam2base, dtype=np.float64) @ T_cam_raw).astype(np.float64)

    # Fixed frame correction learned from your offline post-process:
    # apply +90deg rotation about local Y of the grasp frame: R' = R @ R_fix
    ang = np.pi / 2.0
    c = float(np.cos(ang))
    s = float(np.sin(ang))
    R_fix = np.array([[ c, 0.0,  s],
                      [0.0, 1.0, 0.0],
                      [-s, 0.0,  c]], dtype=np.float64)

    # Corrected poses (for control)
    T_cam = T_cam_raw.copy()
    T_cam[:3, :3] = T_cam[:3, :3] @ R_fix
    T_base = (np.asarray(T_cam2base, dtype=np.float64) @ T_cam).astype(np.float64)

    # scipy Rotation.as_quat() returns (x, y, z, w)
    q_cam_xyzw = R.from_matrix(T_cam[:3, :3]).as_quat()
    q_base_xyzw = R.from_matrix(T_base[:3, :3]).as_quat()

    out = {
        "score": float(getattr(best_grasp, "score", np.nan)),
        "width": float(getattr(best_grasp, "width", np.nan)),
        "depth": float(getattr(best_grasp, "depth", np.nan)),

        # Corrected (for control)
        "cam_xyz": [float(T_cam[0, 3]), float(T_cam[1, 3]), float(T_cam[2, 3])],
        "cam_quat_xyzw": [float(x) for x in q_cam_xyzw],

        "base_xyz": [float(T_base[0, 3]), float(T_base[1, 3]), float(T_base[2, 3])],
        "base_quat_xyzw": [float(x) for x in q_base_xyzw],

        "T_cam_grasp": T_cam.tolist(),
        "T_base_grasp": T_base.tolist(),

        # Debug: raw AnyGrasp output (no fixed correction)
        "T_cam_grasp_raw": T_cam_raw.tolist(),
        "T_base_grasp_raw": T_base_raw.tolist(),

        # Debug: what constant correction was applied
        "R_fix_local_y_90deg": R_fix.tolist(),

        "meta": meta,
    }

    with open(os.path.join(out_dir, "best_grasp.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def draw_top1_on_rgb(out_dir: str, color_bgr: np.ndarray, intr, grasp, name: str = "rgb_top1_mark.png"):
    """Save an RGB image with a marker showing the selected Top-1 grasp center (projected)."""
    os.makedirs(out_dir, exist_ok=True)

    T_cam = grasp_to_matrix(grasp).astype(np.float64)
    uv = proj_uv(
        T_cam[:3, 3],
        fx=float(intr.fx),
        fy=float(intr.fy),
        cx=float(intr.ppx),
        cy=float(intr.ppy),
    )

    img = color_bgr.copy()
    score = float(getattr(grasp, "score", np.nan))

    if uv is not None:
        u, v = uv
        cv2.drawMarker(
            img,
            (int(u), int(v)),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=28,
            thickness=2,
        )
        cv2.putText(
            img,
            f"TOP1 score={score:.4f}  uv=({u},{v})",
            (max(0, int(u) + 10), max(25, int(v) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            img,
            f"TOP1 score={score:.4f}  (proj failed)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    out_path = os.path.join(out_dir, name)
    cv2.imwrite(out_path, img)
    print(f"[Saved] {out_path}")


def save_top1_gripper_ply(out_dir: str, gg_top1, trans: np.ndarray, basename: str = "top1_gripper"):
    """
    Save the Top-1 gripper geometry as a .ply if possible.
    - If geometry is TriangleMesh -> write_triangle_mesh
    - If geometry is LineSet -> try write_line_set (if available), otherwise fallback:
      save its points as a PointCloud ply + save topology as npz.
    """
    os.makedirs(out_dir, exist_ok=True)

    geoms = gg_top1.to_open3d_geometry_list()
    if not geoms:
        print("[WARN] to_open3d_geometry_list() returned empty")
        return

    g0 = geoms[0]
    try:
        g0.transform(trans)
    except Exception as e:
        print("[WARN] gripper transform failed:", e)

    # Try best-effort save
    if isinstance(g0, o3d.geometry.TriangleMesh):
        ply_path = os.path.join(out_dir, f"{basename}.ply")
        ok = o3d.io.write_triangle_mesh(ply_path, g0)
        print(f"[Saved] {ply_path}  ok={ok}")
        return

    # LineSet case
    if isinstance(g0, o3d.geometry.LineSet):
        ply_path = os.path.join(out_dir, f"{basename}.ply")

        # Newer Open3D may have write_line_set
        if hasattr(o3d.io, "write_line_set"):
            ok = o3d.io.write_line_set(ply_path, g0)
            print(f"[Saved] {ply_path}  ok={ok}")
            return

        # Fallback: save as point cloud + topology separately
        pts = np.asarray(g0.points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        pcd_path = os.path.join(out_dir, f"{basename}_points.ply")
        ok = o3d.io.write_point_cloud(pcd_path, pcd, write_ascii=False)
        topo_path = os.path.join(out_dir, f"{basename}_topology.npz")
        np.savez_compressed(topo_path, points=pts, lines=np.asarray(g0.lines, dtype=np.int32))

        print(f"[Saved] {pcd_path}  ok={ok}")
        print(f"[Saved] {topo_path}  (lines saved separately; Open3D has no write_line_set in this build)")
        return

    # Unknown geometry type
    print(f"[WARN] Unsupported geometry type for saving: {type(g0)}")

def vis_o3d(points, colors, gg_pick, T_vis=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    if T_vis is None:
        T_vis = np.eye(4, dtype=np.float64)
    else:
        T_vis = np.asarray(T_vis, dtype=np.float64)
        assert T_vis.shape == (4,4)

    cloud.transform(T_vis)

    grippers = gg_pick.to_open3d_geometry_list()
    for g in grippers:
        g.transform(T_vis)

    o3d.visualization.draw_geometries([cloud, *grippers])