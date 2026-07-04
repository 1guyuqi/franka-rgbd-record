from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import torch
from PIL import Image

from repo_paths import ROOT
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from scipy.spatial.transform import Rotation as R
from utils.utils import (
    depth_aligned_to_cloud,
    apply_T,
    save_capture,
    erode_mask,
    save_mask,      
    filter_grasps_by_mask,
    save_best_grasp,
    draw_top1_on_rgb,
    save_scene_ply_top1,
    save_top1_gripper_ply,
    vis_o3d,
)


def _setup_anygrasp_import(sdk_root: str | None) -> None:
    root = sdk_root or os.environ.get("ANYGRASP_SDK_PATH", "")
    if not root:
        raise RuntimeError(
            "AnyGrasp SDK not found. Set ANYGRASP_SDK_PATH or anygrasp.sdk_path in config."
        )
    sdk_path = Path(root)
    if not sdk_path.is_absolute():
        sdk_path = ROOT / sdk_path
    grasp_detection = sdk_path / "grasp_detection"
    if not grasp_detection.is_dir():
        raise FileNotFoundError(f"AnyGrasp grasp_detection dir not found: {grasp_detection}")
    sys.path.insert(0, str(grasp_detection))


def _load_anygrasp(ag_cfg: dict):
    _setup_anygrasp_import(ag_cfg.get("sdk_path"))
    from gsnet import AnyGrasp

    return AnyGrasp(AnyGraspArgs(ag_cfg))

class AnyGraspArgs:
    """AnyGrasp expects an argparse-like object with these attributes."""
    def __init__(self, ag: dict):
        self.checkpoint_path = ag["checkpoint_path"]
        self.max_gripper_width = float(np.clip(float(ag["max_gripper_width"]), 0.0, 0.1))
        self.gripper_height = float(ag["gripper_height"])
        self.top_down_grasp = bool(ag["top_down_grasp"])
        self.debug = False
        self.dense_grasp = bool(ag["dense_grasp"])
        self.no_collision = (not bool(ag["collision_detection"]))


class Sam3TextSeg:
    def __init__(self, repo_path: str, ckpt_path: str, device: str):
        if not os.path.isdir(repo_path):
            raise FileNotFoundError(f"sam3 repo_path not found: {repo_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"sam3 checkpoint not found: {ckpt_path}")

        sys.path.append(repo_path)
        model = build_sam3_image_model(checkpoint_path=ckpt_path, load_from_HF=False)
        model.to(device).eval()
        self.proc = Sam3Processor(model)

    @torch.no_grad()
    def mask_from_text(self, rgb_u8: np.ndarray, prompt: str, thr: float) -> np.ndarray:
        state = self.proc.set_image(Image.fromarray(rgb_u8, mode="RGB"))
        out = self.proc.set_text_prompt(state=state, prompt=prompt)
        m = out["masks"]

        if not torch.is_tensor(m):
            m = torch.as_tensor(np.asarray(m))

        if m.ndim == 4:
            m2 = m[0, 0]
        elif m.ndim == 3:
            m2 = m[0]
        elif m.ndim == 2:
            m2 = m
        else:
            raise RuntimeError(f"unexpected masks ndim={m.ndim}, shape={tuple(m.shape)}")

        mask = (m2 > float(thr)).detach().cpu().numpy().astype(bool)
        if mask.sum() == 0:
            raise RuntimeError("SAM3 produced empty mask")
        return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    out_dir = cfg["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    T_cam2base = np.asarray(cfg["cam2base"], dtype=np.float64)
    if T_cam2base.shape != (4, 4):
        raise ValueError(f"cam2base must be 4x4, got {T_cam2base.shape}")

    rs_cfg = cfg["realsense"]
    WIDTH = int(rs_cfg["width"])
    HEIGHT = int(rs_cfg["height"])
    FPS = int(rs_cfg["fps"])
    WARMUP = int(rs_cfg["warmup_frames"])
    HOLEFILL = int(rs_cfg["hole_filling_level"])
    AE = bool(rs_cfg["auto_exposure"])
    EMITTER = bool(rs_cfg["emitter"])

    pipe = rs.pipeline()
    cfg_rs = rs.config()
    cfg_rs.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    cfg_rs.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    try:
        prof = pipe.start(cfg_rs)

        dev = prof.get_device()
        depth_sensor = dev.first_depth_sensor()

        if AE:
            try:
                depth_sensor.set_option(rs.option.enable_auto_exposure, 1.0)
            except Exception as e:
                print("[WARN] enable_auto_exposure failed:", e)

        if EMITTER:
            try:
                depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
            except Exception as e:
                print("[WARN] emitter_enabled failed:", e)

        depth_scale = float(depth_sensor.get_depth_scale())

        align = rs.align(rs.stream.color)
        holefill = rs.hole_filling_filter()
        holefill.set_option(rs.option.holes_fill, float(HOLEFILL))

        color_stream = prof.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()

        for _ in range(WARMUP):
            pipe.wait_for_frames(5000)

        frames = align.process(pipe.wait_for_frames(5000))
        cf = frames.get_color_frame()
        df = holefill.process(frames.get_depth_frame())

        if (not cf) or (not df):
            raise RuntimeError("Failed to get color/depth frame")

        color_bgr = np.asanyarray(cf.get_data()).copy()
        depth_u16 = np.asanyarray(df.get_data()).copy()

    finally:
        try:
            pipe.stop()
        except Exception as e:
            print("[WARN] pipe.stop failed:", e)
    # ==========================================================

    meta = {
        "rgb_intrinsics": {
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "cx": float(intr.ppx),
            "cy": float(intr.ppy),
            "model": str(intr.model),
            "coeffs": list(intr.coeffs),
        },
        "depth_scale_m_per_unit": float(depth_scale),
        "align": "depth_to_color",
        "hole_filling_level": int(HOLEFILL),
        "auto_exposure": bool(AE),
        "emitter": bool(EMITTER),
    }

    save_capture(out_dir, color_bgr, depth_u16, meta)

    # cloud
    cloud_cfg = cfg["cloud"]
    zmin = float(cloud_cfg["depth_min_m"])
    zmax = float(cloud_cfg["depth_max_m"])
    lims = list(cloud_cfg["workspace_lims"])
    max_points = int(cloud_cfg["max_points"])

    pts_cam, cols = depth_aligned_to_cloud(
        depth_u16=depth_u16,
        color_bgr=color_bgr,
        depth_scale=depth_scale,
        intr=intr,
        zmin=zmin,
        zmax=zmax,
        max_points=max_points,
    )
    if pts_cam.shape[0] < 2000:
        raise RuntimeError(f"too few valid points: {pts_cam.shape[0]}")

    # optional: save base cloud
    if bool(cfg["output"].get("save_cloud_base_ply", True)):
        ply_path = os.path.join(out_dir, "cloud_base.ply")
        pts_base = apply_T(T_cam2base, pts_cam.astype(np.float64))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_base)
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)
        print(f"[Saved] {ply_path}  N={pts_base.shape[0]}")

    # SAM3 (load AFTER capture, avoid RS timeout)
    sam_cfg = cfg["sam3"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam3 = Sam3TextSeg(sam_cfg["repo_path"], sam_cfg["checkpoint_path"], device)
    rgb_u8 = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    mask = sam3.mask_from_text(rgb_u8, sam_cfg["prompt"], float(sam_cfg["mask_threshold"]))
    mask = erode_mask(mask, int(sam_cfg["mask_erode"]))
    save_mask(out_dir, mask)

    # AnyGrasp (load AFTER capture, avoid RS timeout)
    ag_cfg = cfg["anygrasp"]
    anygrasp = _load_anygrasp(ag_cfg)
    anygrasp.load_net()

    gg, _ = anygrasp.get_grasp(
        pts_cam,
        cols,
        lims=lims,
        apply_object_mask=bool(ag_cfg["apply_object_mask"]),
        dense_grasp=bool(ag_cfg["dense_grasp"]),
        collision_detection=bool(ag_cfg["collision_detection"]),
    )
    if len(gg) == 0:
        raise RuntimeError("AnyGrasp: no grasps")

    gg = gg.nms().sort_by_score()

    gg = filter_grasps_by_mask(
        gg,
        mask,
        fx=float(intr.fx),
        fy=float(intr.fy),
        cx=float(intr.ppx),
        cy=float(intr.ppy),
    )
    if len(gg) == 0:
        raise RuntimeError("MaskFilter: all grasps filtered out")

    topk = min(20, len(gg))
    gg_pick = gg[:topk]
    print(np.array(gg_pick.scores, dtype=np.float64))
    print("best grasp score:", float(gg_pick[0].score))

    # ---- Save top1 gripper geometry / vis transform (define ONCE)
    T_vis = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0,-1, 0],
                      [0, 0, 0, 1]], dtype=np.float64)

    grasp_meta = {
        "depth_scale_m_per_unit": float(depth_scale),
        "depth_min": float(zmin),
        "depth_max": float(zmax),
        "lims": lims,
        "sam3_prompt": sam_cfg["prompt"],
        "sam3_threshold": float(sam_cfg["mask_threshold"]),
        "sam3_erode": int(sam_cfg["mask_erode"]),
        "rgb_intrinsics": meta["rgb_intrinsics"],
    }

    save_best_grasp(
        out_dir,
        gg_pick[0],
        T_cam2base,
        meta=grasp_meta,
    )

    draw_top1_on_rgb(out_dir, color_bgr, intr, gg_pick[0])

    save_scene_ply_top1(
        out_dir=out_dir,
        pts_cam=pts_cam,
        cols_rgb01=cols,
        gg_top1=gg_pick[:1],
        T_vis=T_vis,
        T_cam2base=T_cam2base,
        prefix="scene",
    )

    if bool(cfg["output"].get("vis_o3d", True)):
        vis_o3d(pts_cam, cols, gg_pick, T_vis=T_vis)
        vis_o3d(pts_cam, cols, gg_pick[:1], T_vis=T_vis)

if __name__ == "__main__":
    main()
