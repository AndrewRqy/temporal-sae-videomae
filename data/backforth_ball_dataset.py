"""
Back-and-forth ball dataset.

50 videos: 8 directions × 6-7 starting positions.
Ball moves at FORWARD_SPEED for frames 0-7, then reverses direction for frames 8-15.
Net displacement at video end ≈ 0 m; path length ≈ 2 × 1.0 m = 2.0 m.

Ablation purpose: dissociate three competing hypotheses for what SAE features encode:
  1. Instantaneous speed  → expect FLAT trajectory (speed is constant = FORWARD_SPEED)
  2. Path length (∫|v|dt) → expect MONOTONICALLY RISING trajectory
  3. Net displacement     → expect inverted-V (rises first half, falls back second half)

Usage (inside kubric Docker container):
  python3 backforth_ball_dataset.py --output_dir /output/backforth
"""

import argparse
import colorsys
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

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directions (same as all other datasets)
# ---------------------------------------------------------------------------
DIRECTIONS = [
    (math.cos(math.radians(a)), math.sin(math.radians(a)), 0.0)
    for a in range(0, 360, 45)
]

START_POSITIONS = [
    (-1.5, -1.5),
    (-1.5,  0.0),
    (-1.5,  1.5),
    ( 0.0, -1.5),
    ( 0.0,  0.0),
    ( 0.0,  1.5),
    ( 1.5,  0.0),
]

# ---------------------------------------------------------------------------
# Back-and-forth speed profile
# ---------------------------------------------------------------------------
FORWARD_SPEED  = 3.0   # m/s  (~4 px/frame at 256×256)
N_FORWARD      = 8     # frames moving forward  (0–7)
N_REVERSE      = 8     # frames moving backward (8–15)

# Displacement per half: 8 * 3.0 / 24 ≈ 1.0 m  → ball travels 1.0 m out and 1.0 m back
HALF_DISP_M    = N_FORWARD * FORWARD_SPEED / 24

# ---------------------------------------------------------------------------
# Dataset size: 50 videos (same distribution as nonmono)
# ---------------------------------------------------------------------------
N_VIDEOS = 50
N_DIRS   = len(DIRECTIONS)
N_EXTRA  = N_VIDEOS % N_DIRS   # 2

def n_positions_for_dir(dir_idx: int) -> int:
    base = N_VIDEOS // N_DIRS
    return base + (1 if dir_idx < N_EXTRA else 0)

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
def _make_ball_colors(n: int) -> list:
    colors = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.88)
        colors.append(kb.Color(r, g, b, 1.0))
    return colors

def _make_bg_colors(n: int) -> list:
    base, variation = 0.92, 0.07
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        r = base + variation * math.cos(2 * math.pi * t)
        g = base + variation * math.cos(2 * math.pi * t + 2.09)
        b = base + variation * math.cos(2 * math.pi * t + 4.19)
        colors.append((
            max(0.85, min(1.0, r)),
            max(0.85, min(1.0, g)),
            max(0.85, min(1.0, b)),
        ))
    return colors

BALL_COLORS = _make_ball_colors(N_VIDEOS)
BG_COLORS   = _make_bg_colors(N_VIDEOS)

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------
BALL_RADIUS = 0.3
BALL_MASS   = 1.0
FLOOR_HALF  = 4.0
WALL_THICK  = 0.1
CAMERA_Z    = 10.0
FRAME_END   = 15
FRAME_RATE  = 24
STEP_RATE   = 240


# ---------------------------------------------------------------------------
# Simulation: forward for frames 0-7, reversed for frames 8-15
# ---------------------------------------------------------------------------
def run_backforth_speed(simulator, scene, ball, direction_xyz):
    pc       = simulator._physics_client
    ball_uid = ball.linked_objects[simulator]
    dx, dy, dz = direction_xyz

    steps_per_frame = scene.step_rate // scene.frame_rate
    obj_idxs = [pc.getBodyUniqueId(i) for i in range(pc.getNumBodies())]

    raw = {
        idx: {"position": [], "quaternion": [], "velocity": [], "angular_velocity": []}
        for idx in obj_idxs
    }

    for frame_id in range(scene.frame_end + 1):
        # Forward during first half, reversed during second half
        sign = 1.0 if frame_id < N_FORWARD else -1.0

        for idx in obj_idxs:
            pos,  quat   = simulator.get_position_and_rotation(idx)
            vel,  angvel = simulator.get_velocities(idx)
            raw[idx]["position"].append(pos)
            raw[idx]["quaternion"].append(quat)
            raw[idx]["velocity"].append(vel)
            raw[idx]["angular_velocity"].append(angvel)

        pc.resetBaseVelocity(
            ball_uid,
            linearVelocity=[sign * dx * FORWARD_SPEED,
                             sign * dy * FORWARD_SPEED,
                             sign * dz * FORWARD_SPEED],
            angularVelocity=[0, 0, 0],
        )

        for _ in range(steps_per_frame):
            pc.stepSimulation()

    animation = {
        asset: raw[asset.linked_objects[simulator]]
        for asset in scene.assets
        if asset.linked_objects.get(simulator) in obj_idxs
    }

    for asset, anim in animation.items():
        for fid in range(scene.frame_end + 1):
            asset.position         = anim["position"][fid]
            asset.quaternion       = anim["quaternion"][fid]
            asset.velocity         = anim["velocity"][fid]
            asset.angular_velocity = anim["angular_velocity"][fid]
            asset.keyframe_insert("position",         fid)
            asset.keyframe_insert("quaternion",       fid)
            asset.keyframe_insert("velocity",         fid)
            asset.keyframe_insert("angular_velocity", fid)

    return animation, []


# ---------------------------------------------------------------------------
# Single video generator
# ---------------------------------------------------------------------------
def generate_video(dir_idx, pos_idx, video_idx, output_dir, resolution):
    direction  = DIRECTIONS[dir_idx]
    start_xy   = START_POSITIONS[pos_idx]
    ball_color = BALL_COLORS[video_idx]
    bg_rgb     = BG_COLORS[video_idx]

    label = {
        "dataset_type":      "backforth",
        "direction_idx":     dir_idx,
        "direction_deg":     dir_idx * 45,
        "direction_vec":     list(direction),
        "pos_idx":           pos_idx,
        "start_pos_xy":      list(start_xy),
        "video_idx":         video_idx,
        "ball_color_rgba":   [round(c, 4) for c in
                              [ball_color.r, ball_color.g, ball_color.b, ball_color.a]],
        "forward_speed_mps": FORWARD_SPEED,
        "n_forward_frames":  N_FORWARD,
        "n_reverse_frames":  N_REVERSE,
        "half_disp_m":       round(HALF_DISP_M, 4),
        # Speed profile (instantaneous magnitude, constant throughout)
        "speed_profile":     [FORWARD_SPEED] * (FRAME_END + 1),
    }

    out     = pathlib.Path(output_dir)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="kubric_backforth_"))

    scene = kb.Scene(
        resolution=resolution,
        frame_start=0,
        frame_end=FRAME_END,
        frame_rate=FRAME_RATE,
        step_rate=STEP_RATE,
        gravity=(0, 0, 0),
    )

    simulator = PyBullet(scene, scratch)
    renderer  = Blender(scene, scratch, use_denoising=False, adaptive_sampling=False)

    bg_color = kb.Color(bg_rgb[0], bg_rgb[1], bg_rgb[2], 1.0)
    bg_mat   = kb.FlatMaterial(color=bg_color, indirect_visibility=False)

    floor = kb.Cube(
        name="floor",
        scale=(FLOOR_HALF, FLOOR_HALF, WALL_THICK),
        position=(0, 0, -WALL_THICK),
        material=bg_mat, static=True,
        restitution=1.0, friction=0.0, background=True,
    )
    wall_kw = dict(material=bg_mat, static=True, restitution=1.0,
                   friction=0.0, background=True)
    walls = [
        kb.Cube(name="wall_n",
                scale=(FLOOR_HALF + WALL_THICK, WALL_THICK, 1.0),
                position=(0,  FLOOR_HALF + WALL_THICK, 1.0), **wall_kw),
        kb.Cube(name="wall_s",
                scale=(FLOOR_HALF + WALL_THICK, WALL_THICK, 1.0),
                position=(0, -FLOOR_HALF - WALL_THICK, 1.0), **wall_kw),
        kb.Cube(name="wall_e",
                scale=(WALL_THICK, FLOOR_HALF, 1.0),
                position=( FLOOR_HALF + WALL_THICK, 0, 1.0), **wall_kw),
        kb.Cube(name="wall_w",
                scale=(WALL_THICK, FLOOR_HALF, 1.0),
                position=(-FLOOR_HALF - WALL_THICK, 0, 1.0), **wall_kw),
    ]
    scene.add([floor] + walls)

    ball_mat = kb.PrincipledBSDFMaterial(color=ball_color, roughness=0.4, metallic=0.0)
    ball = kb.Sphere(
        name="ball",
        scale=[BALL_RADIUS] * 3,
        position=(start_xy[0], start_xy[1], BALL_RADIUS),
        mass=BALL_MASS,
        material=ball_mat,
        friction=0.0,
        restitution=1.0,
    )
    scene.add(ball)

    scene.camera = kb.PerspectiveCamera(
        name="camera",
        position=(0, 0, CAMERA_Z),
        look_at=(0, 0, 0),
        focal_length=45,
        sensor_width=36,
    )
    scene.add(kb.DirectionalLight(
        name="sun",
        position=(0, 0, CAMERA_Z),
        look_at=(0, 0, 0),
        intensity=2.5,
    ))

    animation, _ = run_backforth_speed(simulator, scene, ball, direction)

    data_stack = renderer.render()
    out.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(data_stack["rgba"]):
        kb.write_png(frame, kb.as_path(str(out / f"rgba_{i:05d}.png")))

    ball_anim  = animation.get(ball, {})
    trajectory = [
        {"frame": f, "position": list(ball_anim["position"][f]),
         "velocity": list(ball_anim["velocity"][f])}
        for f in range(FRAME_END + 1)
    ] if ball_anim else []

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
# Manifest
# ---------------------------------------------------------------------------
def write_dataset_docs(output_dir):
    root = pathlib.Path(output_dir)
    rows = []
    for meta_path in sorted(root.glob("*/metadata.json")):
        with open(meta_path) as f:
            m = json.load(f)
        lbl  = m["label"]
        traj = m["trajectory"]
        last = traj[-1] if traj else {}
        rows.append({
            "video":         meta_path.parent.name,
            "video_idx":     lbl["video_idx"],
            "direction_idx": lbl["direction_idx"],
            "direction_deg": lbl["direction_deg"],
            "pos_idx":       lbl["pos_idx"],
            "start_x":       lbl["start_pos_xy"][0],
            "start_y":       lbl["start_pos_xy"][1],
            "final_pos_x":   round(last.get("position", [0,0,0])[0], 4),
            "final_pos_y":   round(last.get("position", [0,0,0])[1], 4),
        })
    csv_path = root / "manifest.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote manifest: %s", csv_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--width",         type=int, default=256)
    p.add_argument("--height",        type=int, default=256)
    p.add_argument("--direction_idx", type=int, default=-1,
                   help="Generate only this direction (0-7). Default -1 = all.")
    p.add_argument("--pos_idx",       type=int, default=-1,
                   help="Generate only this position index. Default -1 = all.")
    return p.parse_args()


def main():
    args = parse_args()
    resolution = (args.width, args.height)

    global_idx_map = {}
    idx = 0
    for d in range(N_DIRS):
        for p in range(n_positions_for_dir(d)):
            global_idx_map[(d, p)] = idx
            idx += 1

    pairs = []
    for d in range(N_DIRS):
        if args.direction_idx >= 0 and args.direction_idx != d:
            continue
        n_pos = n_positions_for_dir(d)
        pos_range = [args.pos_idx] if args.pos_idx >= 0 else range(n_pos)
        for p in pos_range:
            pairs.append((d, p))

    logger.info("Generating %d videos in %s", len(pairs), args.output_dir)

    for count, (d, p) in enumerate(pairs, 1):
        tag = f"dir{d:02d}_pos{p:02d}"
        out = pathlib.Path(args.output_dir) / tag
        if (out / "metadata.json").exists():
            logger.info("Skipping %s (already done)", tag)
            continue
        video_idx = global_idx_map[(d, p)]
        logger.info("[%d/%d] %s  dir=%d°  pos=%s  video_idx=%d",
                    count, len(pairs), tag, d*45, START_POSITIONS[p], video_idx)
        generate_video(d, p, video_idx, str(out), resolution)

    write_dataset_docs(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
