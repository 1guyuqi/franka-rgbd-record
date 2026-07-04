from __future__ import annotations

import os
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d

WIDTH, HEIGHT, FPS = 1280, 720, 30
OUT_DIR = "./data"

T_CAM2BASE = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def depth_aligned_to_cloud(depth_u16, color_bgr, depth_scale, intr, zmin=0.05, zmax=2.0):
    """Depth is aligned to color. Use COLOR intrinsics to back-project into a colored point cloud (camera frame)."""
    fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)

    depth = depth_u16.astype(np.float32) * float(depth_scale)
    h, w = depth.shape

    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.reshape(-1)

    m = (z > zmin) & (z < zmax) & np.isfinite(z)
    u = u.reshape(-1)[m]
    v = v.reshape(-1)[m]
    z = z[m]

    x = (u - cx) / fx * z
    y = (v - cy) / fy * z

    pts = np.stack([x, y, z], axis=1).astype(np.float64)
    rgb = color_bgr.reshape(-1, 3)[m][:, ::-1].astype(np.float64) / 255.0
    return pts, rgb


def apply_T(T, pts):
    """Apply 4x4 transform to Nx3 points."""
    return (pts @ T[:3, :3].T) + T[:3, 3]


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

    align = rs.align(rs.stream.color)              # depth -> color
    holefill = rs.hole_filling_filter()
    holefill.set_option(rs.option.holes_fill, 2.0)

    color_stream = prof.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    print(f"[RGB intrinsics] {intr.width}x{intr.height}  fx={intr.fx:.3f} fy={intr.fy:.3f} cx={intr.ppx:.3f} cy={intr.ppy:.3f}")
    print(f"[Depth scale] {depth_scale:.10f} m/unit")
    print(f"[Depth AE] {int(depth_sensor.get_option(rs.option.enable_auto_exposure))}  [Emitter] {int(depth_sensor.get_option(rs.option.emitter_enabled))}")
    print("[Keys] a=save(rgb/depth/cloud_base.ply)   q/ESC=quit")

    # warm-up
    for _ in range(10):
        pipe.wait_for_frames()

    try:
        while True:
            frames = align.process(pipe.wait_for_frames())
            cf = frames.get_color_frame()
            df = holefill.process(frames.get_depth_frame())

            color_bgr = np.asanyarray(cf.get_data())
            depth_u16 = np.asanyarray(df.get_data())

            # show depth using standard RealSense colorizer (simple & stable)
            depth_vis = np.asanyarray(rs.colorizer().colorize(df).get_data())

            cv2.imshow("RGB", color_bgr)
            cv2.imshow("Depth (aligned->RGB)", depth_vis)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break

            if k == ord("a"):
                rgb_path = os.path.join(OUT_DIR, "rgb.png")
                depth_path = os.path.join(OUT_DIR, "depth.png")
                ply_path = os.path.join(OUT_DIR, "cloud_base.ply")

                cv2.imwrite(rgb_path, color_bgr)
                cv2.imwrite(depth_path, depth_u16)

                pts_cam, cols = depth_aligned_to_cloud(depth_u16, color_bgr, depth_scale, intr)
                pts_base = apply_T(T_CAM2BASE, pts_cam)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts_base)
                pcd.colors = o3d.utility.Vector3dVector(cols)
                o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)

                print(f"[Saved] {rgb_path}")
                print(f"[Saved] {depth_path}")
                print(f"[Saved] {ply_path}  N={pts_base.shape[0]}")

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
