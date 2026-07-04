from __future__ import annotations

import os
import json
import cv2
import numpy as np
import pyrealsense2 as rs

# -------- Global config --------
WIDTH = 1280
HEIGHT = 720
FPS = 30
OUT_DIR = "./data"

# Camera -> Robot base extrinsic (4x4)
T_CAM2BASE = np.array(
    [[ 0.2575487 ,  0.64918405, -0.71570157,  1.06160604],
    [ 0.9540992 , -0.28802259,  0.08208357, -0.21999986],
    [-0.15285087, -0.70399081, -0.69356582,  0.66623525],
    [ 0.        ,  0.        ,  0.        ,  1.        ]],
    dtype=np.float64,
)


def start_realsense():
    """Start RealSense with color+depth streams and return runtime intrinsics/scale."""
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    profile = pipeline.start(cfg)

    dev = profile.get_device()

    # Enable auto exposure (color + depth) and IR projector (emitter)
    for s in dev.query_sensors():
        try:
            s.set_option(rs.option.enable_auto_exposure, 1.0)
        except Exception:
            pass
        try:
            s.set_option(rs.option.emitter_enabled, 1.0)
        except Exception:
            pass

    depth_sensor = dev.first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    # Align depth -> color
    align = rs.align(rs.stream.color)

    # Depth hole filling (level=2)
    holefill = rs.hole_filling_filter()
    try:
        holefill.set_option(rs.option.holes_fill, 2.0)
    except Exception:
        pass

    # Read RGB intrinsics (used for back-projection because depth is aligned to color)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)

    meta = {
        "rgb_intrinsics": {
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "model": str(intr.model),
            "coeffs": list(intr.coeffs),
        },
        "depth_scale_m_per_unit": depth_scale,
        "align": "depth_to_color",
        "hole_filling_level": 2,
        "auto_exposure": "enabled_if_supported",
        "emitter": "enabled_if_supported",
    }

    return pipeline, align, holefill, depth_scale, (fx, fy, cx, cy), meta


def grab_aligned_frame(pipeline, align, holefill):
    """Grab one depth-aligned-to-color frame."""
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    depth_frame = holefill.process(depth_frame)

    color_bgr = np.asanyarray(color_frame.get_data())
    depth_u16 = np.asanyarray(depth_frame.get_data())  # uint16 depth units
    return color_bgr, depth_u16


def save_capture(color_bgr, depth_u16, meta):
    """Save rgb/depth/meta to OUT_DIR."""
    os.makedirs(OUT_DIR, exist_ok=True)
    rgb_path = os.path.join(OUT_DIR, "rgb.png")
    depth_path = os.path.join(OUT_DIR, "depth.png")
    meta_path = os.path.join(OUT_DIR, "meta.json")

    cv2.imwrite(rgb_path, color_bgr)
    cv2.imwrite(depth_path, depth_u16)  # keep 16-bit PNG

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {rgb_path}")
    print(f"[Saved] {depth_path}")
    print(f"[Saved] {meta_path}")


def depth_u16_to_view(depth_u16):
    """Convert uint16 depth to a displayable 8-bit grayscale image."""
    d8 = cv2.normalize(depth_u16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(d8, cv2.COLOR_GRAY2BGR)


def pixel_to_range_m(u, v, depth_u16, depth_scale, fx, fy, cx, cy):
    """Compute depth (Z) and range (Euclidean distance to camera center) from a pixel."""
    z = float(depth_u16[v, u]) * float(depth_scale)  # meters
    xn = (float(u) - float(cx)) / float(fx)
    yn = (float(v) - float(cy)) / float(fy)
    r = z * float(np.sqrt(1.0 + xn * xn + yn * yn))
    return z, r

def pixel_to_xyz(u, v, depth_u16, depth_scale, fx, fy, cx, cy, T_cam2base=None):
    """Pixel -> (xyz_cam, xyz_base) using aligned depth-to-color and RGB intrinsics."""
    z = float(depth_u16[v, u]) * float(depth_scale)  # meters (along camera Z)
    x = (float(u) - float(cx)) / float(fx) * z
    y = (float(v) - float(cy)) / float(fy) * z
    xyz_cam = np.array([x, y, z], dtype=np.float64)

    xyz_base = None
    if T_cam2base is not None:
        R = T_cam2base[:3, :3]
        t = T_cam2base[:3, 3]
        xyz_base = R @ xyz_cam + t

    return xyz_cam, xyz_base

def pick_on_saved_images(color_bgr, depth_u16, depth_scale, fx, fy, cx, cy, T_cam2base):
    """Click on the saved frame and print depth/range."""
    H, W = depth_u16.shape
    depth_vis = depth_u16_to_view(depth_u16)

    view = np.hstack([color_bgr, depth_vis])

    scale = 0.75
    view_show = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    state = {"last_uv": None, "last_text": ""}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        xu = int(round(x / scale))
        yu = int(round(y / scale))

        if yu < 0 or yu >= H:
            return

        # Left panel = RGB, Right panel = depth_vis
        if xu < W:
            u, v = xu, yu
        else:
            u, v = xu - W, yu

        if u < 0 or u >= W:
            return

        z_m, r_m = pixel_to_range_m(u, v, depth_u16, depth_scale, fx, fy, cx, cy)
        xyz_cam, xyz_base = pixel_to_xyz(u, v, depth_u16, depth_scale, fx, fy, cx, cy, T_cam2base)

        state["last_uv"] = (u, v)
        state["last_text"] = (
            f"u={u} v={v}  z={z_m:.6f}m  r={r_m:.6f}m  "
            f"cam=[{xyz_cam[0]:.6f},{xyz_cam[1]:.6f},{xyz_cam[2]:.6f}]  "
            f"base=[{xyz_base[0]:.6f},{xyz_base[1]:.6f},{xyz_base[2]:.6f}]"
        )

        print(f"[Click] u={u} v={v} depth_u16={int(depth_u16[v,u])} z_m={z_m:.6f} r_m={r_m:.6f}")
        print(f"        xyz_cam  = [{xyz_cam[0]:.6f}, {xyz_cam[1]:.6f}, {xyz_cam[2]:.6f}]")
        print(f"        xyz_base = [{xyz_base[0]:.6f}, {xyz_base[1]:.6f}, {xyz_base[2]:.6f}]")

    win = "Pick point on saved frame (q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = view_show.copy()
        if state["last_uv"] is not None:
            u, v = state["last_uv"]
            x0 = int(round(u * scale))
            y0 = int(round(v * scale))
            cv2.circle(disp, (x0, y0), 6, (0, 255, 255), 2)
            cv2.putText(disp, state["last_text"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(win, disp)
        k = cv2.waitKey(10) & 0xFF
        if k in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


def main():
    pipeline, align, holefill, depth_scale, (fx, fy, cx, cy), meta = start_realsense()

    print("[RGB intrinsics]")
    print(f"  fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}")
    print(f"[Depth scale] {depth_scale:.10f} (meters per unit)")
    print("[Action] Type 'a' + Enter to capture & pick points, 'q' + Enter to quit.")

    # Warm-up
    for _ in range(15):
        pipeline.wait_for_frames()

    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "q":
                break
            if cmd != "a":
                continue

            color_bgr, depth_u16 = grab_aligned_frame(pipeline, align, holefill)
            save_capture(color_bgr, depth_u16, meta)

            # Pick points on the captured frame (not real-time)
            pick_on_saved_images(color_bgr, depth_u16, depth_scale, fx, fy, cx, cy, T_CAM2BASE)

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()