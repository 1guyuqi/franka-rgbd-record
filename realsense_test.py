from __future__ import annotations

import os
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d

WIDTH, HEIGHT, FPS = 1280, 720, 30
OUT_DIR = "./data"

# Camera -> Robot base (4x4)
T_CAM2BASE = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def depth_to_cloud_aligned(depth_u16, color_bgr, depth_scale, intr, zmin=0.05, zmax=2.0):
    """Back-project depth (already aligned to color) into a colorized point cloud (camera frame)."""
    fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)

    depth_m = depth_u16.astype(np.float32) * float(depth_scale)
    H, W = depth_m.shape

    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z = depth_m.reshape(-1)

    m = (z > zmin) & (z < zmax) & np.isfinite(z)
    u = u.reshape(-1)[m]
    v = v.reshape(-1)[m]
    z = z[m]

    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    pts_cam = np.stack([x, y, z], axis=1).astype(np.float64)

    rgb = color_bgr.reshape(-1, 3)[m][:, ::-1].astype(np.float64) / 255.0
    return pts_cam, rgb


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to Nx3 points."""
    return (pts @ T[:3, :3].T) + T[:3, 3][None, :]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    prof = pipe.start(cfg)

    dev = prof.get_device()
    depth_sensor = dev.first_depth_sensor()
    depth_sensor.set_option(rs.option.enable_auto_exposure, 1.0)
    depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
    depth_scale = depth_sensor.get_depth_scale()

    align = rs.align(rs.stream.color)  # depth -> color
    holefill = rs.hole_filling_filter()
    holefill.set_option(rs.option.holes_fill, 2.0)

    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)

    print(f"[RGB intrinsics] {intr.width}x{intr.height}")
    print(f"fx={fx:.6f} fy={fy:.6f} cx={cx:.6f} cy={cy:.6f}")
    print(f"[depth_scale] {depth_scale:.10f} m/unit")
    print(f"[depth AE] {int(depth_sensor.get_option(rs.option.enable_auto_exposure))}  [emitter] {int(depth_sensor.get_option(rs.option.emitter_enabled))}")
    print("[process] align depth->rgb, holefill=2, save rgb/depth/pointcloud(base)")

    for _ in range(10):
        pipe.wait_for_frames()

    frames = align.process(pipe.wait_for_frames())
    c = frames.get_color_frame()
    d = holefill.process(frames.get_depth_frame())

    color_bgr = np.asanyarray(c.get_data())
    depth_u16 = np.asanyarray(d.get_data())

    rgb_path = os.path.join(OUT_DIR, "rgb.png")
    depth_path = os.path.join(OUT_DIR, "depth.png")
    cv2.imwrite(rgb_path, color_bgr)
    cv2.imwrite(depth_path, depth_u16)

    pts_cam, cols = depth_to_cloud_aligned(depth_u16, color_bgr, depth_scale, intr, zmin=0.05, zmax=2.0)
    pts_base = transform_points(T_CAM2BASE, pts_cam)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_base)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    ply_path = os.path.join(OUT_DIR, "cloud_base.ply")
    o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False, compressed=False)

    print(f"[saved] {rgb_path}")
    print(f"[saved] {depth_path}")
    print(f"[saved] {ply_path}  (N={pts_base.shape[0]})")

    pipe.stop()


if __name__ == "__main__":
    main()