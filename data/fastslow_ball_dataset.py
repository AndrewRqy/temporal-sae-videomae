"""
Fast-slow-fast (non-monotonic) ball dataset generation using Kubric.

50 videos: 8 directions × 6-7 starting positions.
Speed profile is fixed and non-monotonic for every video:
  frames  0–4  : fast  (6.0 m/s  ≈ 8 px/frame at 256×256)
  frames  5–10 : slow  (1.5 m/s  ≈ 2 px/frame at 256×256)
  frames 11–15 : fast  (6.0 m/s)

Variables across videos:
  direction     — 8 evenly-spaced angles (0°, 45°, …, 315°)
  start_position — 7 grid positions in the 8×8 m field
  ball_color     — varied per video (HSV rainbow)
  background     — subtle light-neutral tint per video

Usage (inside kubric Docker container):
  python3 fastslow_ball_dataset.py --output_dir /output/fastslow
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
# Directions (same as velocity/acceleration/nonmono datasets)
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
]  # 7 positions — all verified valid for every direction (max displacement 2.06 m)

# ---------------------------------------------------------------------------
# Fast-slow-fast speed profile (inverse of nonmono)
# ---------------------------------------------------------------------------
SLOW_SPEED = 1.5   # m/s  ≈ 2 px/frame at 256×256
FAST_SPEED = 6.0   # m/s  ≈ 8 px/frame at 256×256

# Speed at each of the 16 frames (0-indexed)
#   frames  0– 4: fast  (5 frames)
#   frames  5–10: slow  (6 frames)
#   frames 11–15: fast  (5 frames)
FRAME_SPEEDS = [FAST_SPEED]*5 + [SLOW_SPEED]*6 + [FAST_SPEED]*5

# Total displacement (m): 5*(6.0/24) + 6*(1.5/24) + 5*(6.0/24) = 1.25+0.375+1.25 = 2.875
TOTAL_DISP_M = sum(s / 24 for s in FRAME_SPEEDS)  # ≈ 2.875 m

# ---------------------------------------------------------------------------
# Dataset size: 50 videos
#   dirs 0-1: 7 positions each  (14)
#   dirs 2-7: 6 positions each  (36)
#   total = 50
# ---------------------------------------------------------------------------
N_VIDEOS   = 50
N_DIRS     = len(DIRECTIONS)    # 8
N_EXTRA    = N_VIDEOS % N_DIRS  # 2  — first N_EXTRA dirs get one extra position

def n_positions_for_dir(dir_idx: int) -> int:
    base = N_VIDEOS // N_DIRS  # 6
    return base + (1 if dir_idx < N_EXTRA else 0)

# ---------------------------------------------------------------------------
# Color palettes (50 videos)
# ---------------------------------------------------------------------------
def _make_ball_colors(n: int) -> list[kb.Color]:
    """n evenly-spaced HSV hues, high saturation & brightness."""
    colors = []
    for i in range(n):
        h = i / n                       # hue 0-1
        r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.88)
        colors.append(kb.Color(r, g, b, 1.0))
    return colors

def _make_bg_colors(n: int) -> list[tuple[float, float, float]]:
    """n subtly different light-neutral floor colours (RGB 0-1)."""
    base = 0.92
    variation = 0.07
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        # Cycle through warm/cool/neutral tints
        r = base + variation * math.cos(2 * math.pi * t)
        g = base + variation * math.cos(2 * math.pi * t + 2.09)   # +120°
        b = base + variation * math.cos(2 * math.pi * t + 4.19)   # +240°
        colors.append((
            max(0.85, min(1.0, r)),
            max(0.85, min(1.0, g)),
            max(0.85, min(1.0, b)),
        ))
    return colors

BALL_COLORS = _make_ball_colors(N_VIDEOS)
BG_COLORS   = _make_bg_colors(N_VIDEOS)

# ---------------------------------------------------------------------------
# Scene constants (match velocity/acceleration/nonmono datasets)
# ---------------------------------------------------------------------------
BALL_RADIUS = 0.3
BALL_MASS   = 1.0
FLOOR_HALF  = 4.0
WALL_THICK  = 0.1
CAMERA_Z    = 10.0
FRAME_END   = 15      # 16 frames (0-indexed)
FRAME_RATE  = 24
STEP_RATE   = 240     # 10 substeps/frame


# ---------------------------------------------------------------------------
# Custom simulation: non-monotonic speed via direct velocity control
# ---------------------------------------------------------------------------
def run_nonmono_speed(simulator, scene, ball, direction_xyz):
    """
    Simulate fast-slow-fast speed by calling resetBaseVelocity at each frame.
    Records positions for Blender keyframes identically to simulator.run().
    """
    pc      = simulator._physics_client
    ball_uid = ball.linked_objects[simulator]
    dx, dy, dz = direction_xyz

    steps_per_frame = scene.step_rate // scene.frame_rate
    obj_idxs = [pc.getBodyUniqueId(i) for i in range(pc.getNumBodies())]

    raw = {
        idx: {"position": [], "quaternion": [], "velocity": [], "angular_velocity": []}
        for idx in obj_idxs
    }

    for frame_id in range(scene.frame_end + 1):  # 0–15
        speed = FRAME_SPEEDS[frame_id]

        # Record state at the start of this frame (before physics steps)
        for idx in obj_idxs:
            pos,  quat   = simulator.get_position_and_rotation(idx)
            vel,  angvel = simulator.get_velocities(idx)
            raw[idx]["position"].append(pos)
            raw[idx]["quaternion"].append(quat)
            raw[idx]["velocity"].append(vel)
            raw[idx]["angular_velocity"].append(angvel)

        # Set exact velocity for this frame (no inertia, no drift)
        pc.resetBaseVelocity(
            ball_uid,
            linearVelocity=[dx * speed, dy * speed, dz * speed],
            angularVelocity=[0, 0, 0],
        )

        for _ in range(steps_per_frame):
            pc.stepSimulation()

    # Build asset-keyed animation dict (same format as simulator.run())
    animation = {
        asset: raw[asset.linked_objects[simulator]]
        for asset in scene.assets
        if asset.linked_objects.get(simulator) in obj_idxs
    }

    # Transfer to Blender keyframes
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
        "dataset_type":    "fastslow",
        "direction_idx":   dir_idx,
        "direction_deg":   dir_idx * 45,
        "direction_vec":   list(direction),
        "pos_idx":         pos_idx,
        "start_pos_xy":    list(start_xy),
        "video_idx":       video_idx,
        "ball_color_rgba": [round(c, 4) for c in [ball_color.r, ball_color.g,
                                                    ball_color.b, ball_color.a]],
        "slow_speed_mps":  SLOW_SPEED,
        "fast_speed_mps":  FAST_SPEED,
        "speed_profile":   FRAME_SPEEDS,
        "total_disp_m":    round(TOTAL_DISP_M, 4),
    }

    out     = pathlib.Path(output_dir)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="kubric_fastslow_"))

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

    # Floor and walls with per-video background tint
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

    # Ball with per-video colour (PBR so the directional light gives 3D shading)
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

    # Overhead perspective camera (same as velocity/acceleration/nonmono datasets)
    scene.camera = kb.PerspectiveCamera(
        name="camera",
        position=(0, 0, CAMERA_Z),
        look_at=(0, 0, 0),
        focal_length=45,
        sensor_width=36,
    )

    # Directional light (same position / intensity as other datasets)
    scene.add(kb.DirectionalLight(
        name="sun",
        position=(0, 0, CAMERA_Z),
        look_at=(0, 0, 0),
        intensity=2.5,
    ))

    # Run fast-slow-fast speed simulation
    animation, _ = run_nonmono_speed(simulator, scene, ball, direction)

    # Render and save only rgba frames
    data_stack = renderer.render()
    out.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(data_stack["rgba"]):
        kb.write_png(frame, kb.as_path(str(out / f"rgba_{i:05d}.png")))

    # Trajectory log
    ball_anim = animation.get(ball, {})
    trajectory = [
        {
            "frame":    f,
            "position": list(ball_anim["position"][f]),
            "velocity": list(ball_anim["velocity"][f]),
        }
        for f in range(FRAME_END + 1)
    ] if ball_anim else []

    metadata = {
        "label":      label,
        "trajectory": trajectory,
        "scene": {
            "frame_end":      FRAME_END,
            "frame_rate":     FRAME_RATE,
            "step_rate":      STEP_RATE,
            "ball_radius_m":  BALL_RADIUS,
            "ball_mass_kg":   BALL_MASS,
            "field_size_m":   FLOOR_HALF * 2,
            "camera_z_m":     CAMERA_Z,
            "resolution":     list(resolution),
        },
    }
    kb.write_json(metadata, kb.as_path(str(out / "metadata.json")))

    shutil.rmtree(str(scratch), ignore_errors=True)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Manifest + README
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
            "video":           meta_path.parent.name,
            "video_idx":       lbl["video_idx"],
            "direction_idx":   lbl["direction_idx"],
            "direction_deg":   lbl["direction_deg"],
            "dir_x":           round(lbl["direction_vec"][0], 6),
            "dir_y":           round(lbl["direction_vec"][1], 6),
            "pos_idx":         lbl["pos_idx"],
            "start_x":         lbl["start_pos_xy"][0],
            "start_y":         lbl["start_pos_xy"][1],
            "final_pos_x":     round(last.get("position", [0,0,0])[0], 4),
            "final_pos_y":     round(last.get("position", [0,0,0])[1], 4),
        })

    csv_path = root / "manifest.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote manifest: %s (%d rows)", csv_path, len(rows))

    readme_path = root / "README.md"
    with open(readme_path, "w") as f:
        f.write("# Fast-Slow-Fast Ball Dataset\n\n")
        f.write("Generated with Kubric. Fixed fast-slow-fast speed profile; "
                "direction and start-position vary.\n\n")
        f.write("## Speed profile\n\n")
        f.write("| Phase | Frames | Speed |\n|---|---|---|\n")
        f.write(f"| Fast | 0–4   | {FAST_SPEED} m/s (~8 px/frame at 256×256) |\n")
        f.write(f"| Slow | 5–10  | {SLOW_SPEED} m/s (~2 px/frame at 256×256) |\n")
        f.write(f"| Fast | 11–15 | {FAST_SPEED} m/s |\n\n")
        f.write(f"Total displacement: ≈ {TOTAL_DISP_M:.3f} m per video.\n\n")
        f.write("## Scene setup\n\n")
        f.write("Identical to velocity/acceleration/nonmono datasets: PerspectiveCamera at z=10 m, "
                "PrincipledBSDFMaterial ball, DirectionalLight, 256×256 px output.\n\n")
        f.write("## Varied per video\n\n")
        f.write("- Ball colour (HSV rainbow across 50 videos)\n")
        f.write("- Background floor colour (subtle neutral tint)\n")
        f.write("- Direction (8 angles × 0°–315°)\n")
        f.write("- Starting position (6–7 grid positions)\n\n")
        f.write("## Naming convention\n\n")
        f.write("`dirDD_posPP/` where DD = direction index (0-7), PP = position index (0-6).\n")

    logger.info("Wrote README: %s", readme_path)


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

    # Build the ordered list of (dir_idx, pos_idx) pairs for 50 videos
    pairs = []
    for d in range(N_DIRS):
        n_pos = n_positions_for_dir(d)
        pos_range = [args.pos_idx] if args.pos_idx >= 0 else range(n_pos)
        dir_range  = [d] if (args.direction_idx >= 0 and args.direction_idx != d) else [d]
        if args.direction_idx >= 0 and args.direction_idx != d:
            continue
        for p in pos_range:
            pairs.append((d, p))

    total = len(pairs)
    logger.info("Generating %d videos in %s", total, args.output_dir)

    # Assign global video_idx by full ordering (dir 0 pos 0 = 0, dir 0 pos 1 = 1, …)
    global_idx_map = {}
    idx = 0
    for d in range(N_DIRS):
        for p in range(n_positions_for_dir(d)):
            global_idx_map[(d, p)] = idx
            idx += 1

    for count, (d, p) in enumerate(pairs, 1):
        tag = f"dir{d:02d}_pos{p:02d}"
        out = pathlib.Path(args.output_dir) / tag
        if (out / "metadata.json").exists():
            logger.info("Skipping %s (already done)", tag)
            continue
        video_idx = global_idx_map[(d, p)]
        logger.info("[%d/%d] %s  dir=%d°  pos=%s  video_idx=%d",
                    count, total, tag, d*45, START_POSITIONS[p], video_idx)
        generate_video(d, p, video_idx, str(out), resolution)

    write_dataset_docs(args.output_dir)
    logger.info("Done. %d videos in %s", total, args.output_dir)


if __name__ == "__main__":
    main()
