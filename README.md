# Franka + RealSense Data Collection

RGB-D capture and Franka motion utilities for building manipulation datasets.

## Features

- **RealSense RGB-D**: aligned color + depth, intrinsics, `meta.json`
- **Point cloud**: colored PLY in robot **base** frame (`cam2base` extrinsic)
- **Franka motion**: joint / Cartesian trajectory modules + `panda_py` examples
- **Optional** `demo.py`: single-frame SAM3 segmentation + AnyGrasp (checkpoints not included)

> **Note:** SpaceMouse teleop and multi-frame episode recording are on the [Roadmap](#roadmap).

## Hardware

- Franka Emika (FR3 / Panda) + Franka Hand
- Intel RealSense (D435 / D435i class)
- CUDA workstation (optional, for `demo.py` only)

## Install

```bash
conda create -n robo_record python=3.10
conda activate robo_record

pip install numpy opencv-python pyrealsense2 open3d pyyaml scipy
pip install panda-py
```

Copy local configs from examples:

```bash
cp config/anygrasp_realsense_config.example.json config/anygrasp_realsense_config.json
cp config/franka.example.json config/franka.json   # optional; default config already works
```

Edit `cam2base` (4×4 camera-to-base transform) before saving base-frame point clouds.

## Quick start

**1. Single-frame RGB-D + point cloud**

```bash
python realsense_video_test.py
```

Keys: `a` = save `rgb.png`, `depth.png`, `cloud_base.ply` · `q` = quit

**2. Depth inspection**

```bash
python realsense_depth.py
```

**3. Franka Cartesian motion**

```bash
export FRANKA_IP=your.robot.ip
python test_cartesian.py
```

**4. Joint trajectory test**

```bash
export FRANKA_IP=your.robot.ip
python test.py
```

**5. Optional grasp demo** (requires AnyGrasp + SAM3)

```bash
export ANYGRASP_SDK_PATH=/path/to/anygrasp_sdk
python demo.py --config config/anygrasp_realsense_config.json
```

## Repository layout

```
├── realsense_video_test.py   # RGB-D preview + single-frame capture
├── realsense_depth.py        # Interactive depth inspection
├── joint_trajectory.py       # Joint-space trajectories
├── cartesian_trajectory.py   # Cartesian end-effector trajectories
├── test_cartesian.py         # Franka Cartesian control example
├── demo.py                   # Optional: SAM3 + AnyGrasp single-frame demo
├── repo_paths.py             # Repo-root paths (no machine-specific hardcoding)
├── config/                   # Trajectory and camera configs
├── model/                    # FR3 URDF
└── assets/                   # Robot meshes
```

## Calibration

Provide camera → base extrinsic `T_cam2base` (4×4):

- Edit `T_CAM2BASE` in `realsense_video_test.py`, or
- Set `"cam2base"` in `config/anygrasp_realsense_config.json`, or
- Save as `cam2base.npy` / `cam2base.txt` (gitignored)

Re-calibrate whenever the camera mount changes.

## Output layout

```
data/
├── rgb.png
├── depth.png
├── meta.json           # intrinsics, depth_scale, align settings
└── cloud_base.ply      # colored cloud in base frame
```

## Environment variables

| Variable | Description |
|----------|-------------|
| `FRANKA_IP` | Franka controller IP address |
| `ANYGRASP_SDK_PATH` | Root of AnyGrasp SDK (for `demo.py`) |

## Roadmap

- [ ] SpaceMouse Cartesian teleoperation (velocity / delta, enable + e-stop)
- [ ] Episode recorder: start / stop / discard at fixed rate (e.g. 30 Hz)
- [ ] Per-frame sync: joint `q` (7-DOF), end-effector pose, timestamps
- [ ] HOI4D-style export folders

## Safety

- Keep the hardware **e-stop** within reach
- Use low `set_speed_factor` during first tests
- Verify workspace limits in `config/` before motion

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)
- [Open3D](https://www.open3d.org/)
- [panda_py](https://github.com/JanMeyerBerling/panda-py)
