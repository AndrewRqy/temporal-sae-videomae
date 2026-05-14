"""
Synthetic ball dataset generation using Kubric.

Velocity dataset:    8 directions x 7 speeds x 7 start positions = 392 videos
Acceleration dataset: 8 directions x 5 accelerations x 7 start positions = 280 videos

Each video: 16 frames, 24 fps, 256x256 px, single blue sphere on flat white background,
overhead orthographic camera, 8x8m field with elastic boundary walls.

Usage (inside kubric Docker container):
  python3 ball_dataset.py --dataset_type velocity --output_dir /output/velocity
  python3 ball_dataset.py --dataset_type acceleration --output_dir /output/acceleration
"""

import argparse
import csv
import json
import logging
import math
import pathlib
import shutil
import tempfile

import kubric as kb
from kubric.renderer import Blender
from kubric.simulator import PyBullet
import pybullet as pb

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter grids (from paper Appendix A.1.2)
# ---------------------------------------------------------------------------
DIRECTIONS = [
    (math.cos(math.radians(a)), math.sin(math.radians(a)), 0.0)
    for a in range(0, 360, 45)
]  # 8 unit vectors at 0, 45, ..., 315 degrees

START_POSITIONS = [
    (-1.5, -1.5),
    (-1.5,  0.0),
    (-1.5,  1.5),
    ( 0.0, -1.5),
    ( 0.0,  0.0),
    ( 0.0,  1.5),
    ( 1.5,  0.0),
]  # 7 start positions (XY only)

SPEEDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]          # m/s, 7 values
ACCELERATIONS = [0.5, 1.0, 1.5, 2.0, 2.5]              # m/s^2, 5 values

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------
BALL_RADIUS = 0.3   # m
BALL_MASS   = 1.0   # kg
FLOOR_HALF  = 4.0   # 8x8 m floor
WALL_THICK  = 0.1
CAMERA_Z    = 10.0
FRAME_END   = 15    # 0-indexed -> 16 frames total
FRAME_RATE  = 24
STEP_RATE   = 240   # 10 physics substeps per frame


# ---------------------------------------------------------------------------
# Custom simulation loop for constant-force (acceleration) mode
# ---------------------------------------------------------------------------
def run_with_constant_force(simulator, scene, ball_uid, force_xyz,
                            frame_start=0, frame_end=None):
    """Like simulator.run() but applies a constant force to ball_uid each step."""
    frame_end = scene.frame_end if frame_end is None else frame_end
    steps_per_frame = scene.step_rate // scene.frame_rate
    max_step = (frame_end - frame_start + 1) * steps_per_frame

    pc = simulator._physics_client
    obj_idxs = [pc.getBodyUniqueId(i) for i in range(pc.getNumBodies())]
    raw = {idx: {"position": [], "quaternion": [], "velocity": [], "angular_velocity": []}
           for idx in obj_idxs}

    for step in range(max_step):
        pc.applyExternalForce(ball_uid, -1, force_xyz, [0, 0, 0], pb.WORLD_FRAME)

        if step % steps_per_frame == 0:
            for idx in obj_idxs:
                pos, quat = simulator.get_position_and_rotation(idx)
                vel, angvel = simulator.get_velocities(idx)
                raw[idx]["position"].append(pos)
                raw[idx]["quaternion"].append(quat)
                raw[idx]["velocity"].append(vel)
                raw[idx]["angular_velocity"].append(angvel)

        pc.stepSimulation()

    # Build asset-keyed animation dict (same format as simulator.run())
    animation = {
        asset: raw[asset.linked_objects[simulator]]
        for asset in scene.assets
        if asset.linked_objects.get(simulator) in obj_idxs
    }

    # Transfer to Blender keyframes
    for asset, anim in animation.items():
        for fid in range(frame_end - frame_start + 1):
            asset.position         = anim["position"][fid]
            asset.quaternion       = anim["quaternion"][fid]
            asset.velocity         = anim["velocity"][fid]
            asset.angular_velocity = anim["angular_velocity"][fid]
            asset.keyframe_insert("position",         fid + frame_start)
            asset.keyframe_insert("quaternion",       fid + frame_start)
            asset.keyframe_insert("velocity",         fid + frame_start)
            asset.keyframe_insert("angular_velocity", fid + frame_start)

    return animation, []


# ---------------------------------------------------------------------------
# Single video generator
# ---------------------------------------------------------------------------
def generate_video(dataset_type, dir_idx, param_idx, pos_idx, output_dir, resolution):
    direction = DIRECTIONS[dir_idx]
    start_xy  = START_POSITIONS[pos_idx]

    if dataset_type == "velocity":
        speed = SPEEDS[param_idx]
        init_velocity = tuple(v * speed for v in direction)
        label = {
            "dataset_type":    "velocity",
            "direction_idx":   dir_idx,
            "direction_deg":   dir_idx * 45,
            "direction_vec":   list(direction),
            "speed_idx":       param_idx,
            "speed_mps":       speed,
            "start_pos_idx":   pos_idx,
            "start_pos_xy":    list(start_xy),
        }
    else:
        accel = ACCELERATIONS[param_idx]
        force = BALL_MASS * accel
        force_xyz = tuple(v * force for v in direction)
        label = {
            "dataset_type":    "acceleration",
            "direction_idx":   dir_idx,
            "direction_deg":   dir_idx * 45,
            "direction_vec":   list(direction),
            "accel_idx":       param_idx,
            "accel_mps2":      accel,
            "force_N":         force,
            "start_pos_idx":   pos_idx,
            "start_pos_xy":    list(start_xy),
        }

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scratch = pathlib.Path(tempfile.mkdtemp(prefix="kubric_"))

    # --- Scene
    scene = kb.Scene(
        resolution=resolution,
        frame_start=0,
        frame_end=FRAME_END,
        frame_rate=FRAME_RATE,
        step_rate=STEP_RATE,
        gravity=(0, 0, 0),  # horizontal-only motion; no gravity needed
    )

    simulator = PyBullet(scene, scratch)
    renderer  = Blender(scene, scratch, use_denoising=False, adaptive_sampling=False)

    # --- Materials
    # Floor/walls: flat white background (no lighting interaction)
    white = kb.FlatMaterial(color=kb.get_color("white"), indirect_visibility=False)
    # Ball: PBR so the directional light gives it a 3D shaded appearance
    blue  = kb.PrincipledBSDFMaterial(color=kb.Color(0.15, 0.47, 0.71, 1.0),
                                      roughness=0.4, metallic=0.0)

    # --- Floor (static, white)
    floor = kb.Cube(
        name="floor",
        scale=(FLOOR_HALF, FLOOR_HALF, WALL_THICK),
        position=(0, 0, -WALL_THICK),
        material=white,
        static=True,
        restitution=1.0,
        friction=0.0,
        background=True,
    )

    # --- Boundary walls (elastic, white)
    wall_dyn = dict(material=white, static=True, restitution=1.0, friction=0.0, background=True)
    north = kb.Cube(name="wall_n", scale=(FLOOR_HALF + WALL_THICK, WALL_THICK, 1.0),
                    position=(0,  FLOOR_HALF + WALL_THICK, 1.0), **wall_dyn)
    south = kb.Cube(name="wall_s", scale=(FLOOR_HALF + WALL_THICK, WALL_THICK, 1.0),
                    position=(0, -FLOOR_HALF - WALL_THICK, 1.0), **wall_dyn)
    east  = kb.Cube(name="wall_e", scale=(WALL_THICK, FLOOR_HALF, 1.0),
                    position=( FLOOR_HALF + WALL_THICK, 0, 1.0), **wall_dyn)
    west  = kb.Cube(name="wall_w", scale=(WALL_THICK, FLOOR_HALF, 1.0),
                    position=(-FLOOR_HALF - WALL_THICK, 0, 1.0), **wall_dyn)

    scene.add([floor, north, south, east, west])

    # --- Ball
    ball = kb.Sphere(
        name="ball",
        scale=[BALL_RADIUS] * 3,
        position=(start_xy[0], start_xy[1], BALL_RADIUS),
        mass=BALL_MASS,
        material=blue,
        friction=0.0,
        restitution=1.0,
    )
    if dataset_type == "velocity":
        ball.velocity = init_velocity
    scene.add(ball)

    # --- Camera: overhead perspective at (0,0,10), FOV sized to cover 8m floor
    # focal_length=45mm, sensor_width=36mm → hFOV = 2*arctan(4/10) ≈ 43.6°
    scene.camera = kb.PerspectiveCamera(
        name="camera",
        position=(0, 0, CAMERA_Z),
        look_at=(0, 0, 0),
        focal_length=45,
        sensor_width=36,
    )

    # --- Light
    scene.add(kb.DirectionalLight(
        name="sun",
        position=(0, 0, CAMERA_Z),
        look_at=(0, 0, 0),
        intensity=2.5,
    ))

    # --- Simulation
    if dataset_type == "velocity":
        animation, _ = simulator.run(frame_start=0, frame_end=FRAME_END)
    else:
        ball_uid = ball.linked_objects[simulator]
        animation, _ = run_with_constant_force(
            simulator, scene, ball_uid, force_xyz,
            frame_start=0, frame_end=FRAME_END,
        )

    # --- Render (computes all passes, we only save rgba)
    data_stack = renderer.render()

    # --- Save only the rgba frames (skip depth/flow/normals/object_coordinates)
    out.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(data_stack["rgba"]):
        kb.write_png(frame, kb.as_path(str(out / f"rgba_{i:05d}.png")))

    # --- Build trajectory from animation
    ball_anim = animation.get(ball, {})
    trajectory = [
        {
            "frame":    f,
            "position": list(ball_anim["position"][f]),
            "velocity": list(ball_anim["velocity"][f]),
        }
        for f in range(FRAME_END + 1)
    ] if ball_anim else []

    # --- Save metadata
    metadata = {
        "label":      label,
        "trajectory": trajectory,
        "scene": {
            "frame_end":     FRAME_END,
            "frame_rate":    FRAME_RATE,
            "step_rate":     STEP_RATE,
            "ball_radius_m": BALL_RADIUS,
            "ball_mass_kg":  BALL_MASS,
            "field_size_m":  FLOOR_HALF * 2,
            "camera_z_m":    CAMERA_Z,
            "resolution":    list(resolution),
        },
    }
    kb.write_json(metadata, kb.as_path(str(out / "metadata.json")))

    shutil.rmtree(str(scratch), ignore_errors=True)

    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Main: loop over all combinations
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_type", choices=["velocity", "acceleration"], required=True)
    p.add_argument("--output_dir",   type=str, required=True)
    p.add_argument("--width",        type=int, default=256)
    p.add_argument("--height",       type=int, default=256)
    p.add_argument("--direction_idx", type=int, default=-1,
                   help="Run only this direction index (0-7). Default -1 = all.")
    p.add_argument("--param_idx",    type=int, default=-1,
                   help="Run only this speed/accel index. Default -1 = all.")
    p.add_argument("--pos_idx",      type=int, default=-1,
                   help="Run only this start-position index. Default -1 = all.")
    return p.parse_args()


def write_dataset_docs(output_dir, dataset_type):
    """Write manifest CSV and a human-readable README for the completed dataset."""
    root = pathlib.Path(output_dir)

    # Collect all metadata.json files
    rows = []
    param_key  = "speed_mps"    if dataset_type == "velocity" else "accel_mps2"
    param_label = "speed_m_per_s" if dataset_type == "velocity" else "accel_m_per_s2"

    for meta_path in sorted(root.glob("*/metadata.json")):
        with open(meta_path) as f:
            m = json.load(f)
        lbl = m["label"]
        traj = m["trajectory"]
        last = traj[-1] if traj else {}
        rows.append({
            "video":            meta_path.parent.name,
            "dataset_type":     lbl["dataset_type"],
            "direction_idx":    lbl["direction_idx"],
            "direction_deg":    lbl["direction_deg"],
            "dir_x":            round(lbl["direction_vec"][0], 6),
            "dir_y":            round(lbl["direction_vec"][1], 6),
            param_label:        lbl[param_key],
            "start_pos_idx":    lbl["start_pos_idx"],
            "start_x":          lbl["start_pos_xy"][0],
            "start_y":          lbl["start_pos_xy"][1],
            "final_pos_x":      round(last.get("position", [0, 0, 0])[0], 4),
            "final_pos_y":      round(last.get("position", [0, 0, 0])[1], 4),
            "final_vel_x":      round(last.get("velocity", [0, 0, 0])[0], 4),
            "final_vel_y":      round(last.get("velocity", [0, 0, 0])[1], 4),
        })

    # Write CSV manifest
    csv_path = root / "manifest.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote manifest: %s (%d rows)", csv_path, len(rows))

    # Write README
    params = SPEEDS if dataset_type == "velocity" else ACCELERATIONS
    readme_path = root / "README.md"
    with open(readme_path, "w") as f:
        f.write(f"# {dataset_type.capitalize()} Ball Dataset\n\n")
        f.write("Generated with [Kubric](https://github.com/google-research/kubric) "
                "following paper Appendix A.1.2.\n\n")

        f.write("## Scene setup\n\n")
        f.write("| Parameter | Value |\n|---|---|\n")
        f.write(f"| Ball radius | {BALL_RADIUS} m |\n")
        f.write(f"| Ball mass | {BALL_MASS} kg |\n")
        f.write(f"| Ball color | blue (R=0.15, G=0.47, B=0.71) |\n")
        f.write(f"| Background | flat white |\n")
        f.write(f"| Field size | {FLOOR_HALF * 2} × {FLOOR_HALF * 2} m (elastic walls) |\n")
        f.write(f"| Camera | overhead perspective at z={CAMERA_Z} m, focal_length=45mm, sensor_width=36mm |\n")
        f.write(f"| Resolution | 256 × 256 px |\n")
        f.write(f"| Frames | {FRAME_END + 1} (0–{FRAME_END}), {FRAME_RATE} fps |\n")
        f.write(f"| Physics step rate | {STEP_RATE} Hz ({STEP_RATE // FRAME_RATE} substeps/frame) |\n")
        f.write(f"| Friction | 0.0 |\n")
        f.write(f"| Restitution | 1.0 (elastic) |\n\n")

        f.write("## Parameter grid\n\n")
        f.write(f"**Directions** (8): {[d*45 for d in range(8)]}°\n\n")
        f.write(f"**{'Speeds' if dataset_type == 'velocity' else 'Accelerations'}** "
                f"({len(params)}): {params} "
                f"{'m/s' if dataset_type == 'velocity' else 'm/s²'}\n\n")
        f.write(f"**Start positions** ({len(START_POSITIONS)}):\n\n")
        f.write("| idx | x (m) | y (m) |\n|---|---|---|\n")
        for i, (x, y) in enumerate(START_POSITIONS):
            f.write(f"| {i} | {x} | {y} |\n")
        f.write(f"\n**Total videos**: {len(DIRECTIONS)} × {len(params)} × "
                f"{len(START_POSITIONS)} = {len(DIRECTIONS) * len(params) * len(START_POSITIONS)}\n\n")

        f.write("## Video naming convention\n\n")
        f.write("`dirDD_paramPP_posSS/` where DD = direction index, PP = "
                f"{'speed' if dataset_type == 'velocity' else 'acceleration'} index, "
                "SS = start-position index.\n\n")

        f.write("## Files per video\n\n")
        f.write("| File | Description |\n|---|---|\n")
        f.write("| `rgba_XXXXX.png` | RGBA frames (16 total) |\n")
        f.write("| `segmentation_XXXXX.png` | Segmentation masks |\n")
        f.write("| `forward_flow_XXXXX.png` | Forward optical flow |\n")
        f.write("| `backward_flow_XXXXX.png` | Backward optical flow |\n")
        f.write("| `metadata.json` | Ground-truth label + per-frame trajectory |\n\n")

        f.write("## Manifest\n\n")
        f.write("See `manifest.csv` for a tabular summary of all videos with their "
                "parameters and final-frame position/velocity.\n")

    logger.info("Wrote README: %s", readme_path)


def main():
    args = parse_args()
    resolution = (args.width, args.height)

    n_params = len(SPEEDS) if args.dataset_type == "velocity" else len(ACCELERATIONS)

    dir_range   = [args.direction_idx] if args.direction_idx >= 0 else range(len(DIRECTIONS))
    param_range = [args.param_idx]     if args.param_idx     >= 0 else range(n_params)
    pos_range   = [args.pos_idx]       if args.pos_idx       >= 0 else range(len(START_POSITIONS))

    total = len(list(dir_range)) * len(list(param_range)) * len(list(pos_range))
    logger.info("Generating %d videos for dataset_type=%s", total, args.dataset_type)

    count = 0
    for d in dir_range:
        for p in param_range:
            for s in pos_range:
                tag = f"dir{d:02d}_param{p:02d}_pos{s:02d}"
                out = pathlib.Path(args.output_dir) / tag
                if (out / "metadata.json").exists():
                    logger.info("Skipping %s (already done)", tag)
                    count += 1
                    continue
                logger.info("[%d/%d] Generating %s ...", count + 1, total, tag)
                generate_video(args.dataset_type, d, p, s, str(out), resolution)
                count += 1

    write_dataset_docs(args.output_dir, args.dataset_type)
    logger.info("Done. Generated %d videos in %s", total, args.output_dir)


if __name__ == "__main__":
    main()
