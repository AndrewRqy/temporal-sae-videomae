"""
Static ball dataset: ball is completely stationary throughout each video.

30 videos: 6 starting positions × 5 colour variants.
Speed = 0 m/s for all 16 frames.

Ablation purpose:
  If a temporal SAE feature still ramps up to t=7 on a motionless ball,
  it is tracking elapsed frame time, NOT motion or displacement.
  If it stays flat, it was motion-dependent.

Usage (inside kubric Docker container):
  python3 static_ball_dataset.py --output_dir /output/static
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
# Start positions (no direction — ball never moves)
# ---------------------------------------------------------------------------
START_POSITIONS = [
    (-1.5, -1.5),
    (-1.5,  0.0),
    (-1.5,  1.5),
    ( 0.0, -1.5),
    ( 0.0,  0.0),
    ( 0.0,  1.5),
]
N_POSITIONS = len(START_POSITIONS)   # 6
N_VARIANTS  = 5                      # colour/bg repeats per position
N_VIDEOS    = N_POSITIONS * N_VARIANTS  # 30

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
# Scene constants (identical to other datasets)
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
# Simulation: zero velocity at every frame
# ---------------------------------------------------------------------------
def run_static(simulator, scene, ball):
    pc       = simulator._physics_client
    ball_uid = ball.linked_objects[simulator]

    steps_per_frame = scene.step_rate // scene.frame_rate
    obj_idxs = [pc.getBodyUniqueId(i) for i in range(pc.getNumBodies())]

    raw = {
        idx: {"position": [], "quaternion": [], "velocity": [], "angular_velocity": []}
        for idx in obj_idxs
    }

    for frame_id in range(scene.frame_end + 1):
        for idx in obj_idxs:
            pos,  quat   = simulator.get_position_and_rotation(idx)
            vel,  angvel = simulator.get_velocities(idx)
            raw[idx]["position"].append(pos)
            raw[idx]["quaternion"].append(quat)
            raw[idx]["velocity"].append(vel)
            raw[idx]["angular_velocity"].append(angvel)

        # Hold ball perfectly still every frame
        pc.resetBaseVelocity(ball_uid,
                             linearVelocity=[0, 0, 0],
                             angularVelocity=[0, 0, 0])

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
def generate_video(pos_idx, variant_idx, video_idx, output_dir, resolution):
    start_xy   = START_POSITIONS[pos_idx]
    ball_color = BALL_COLORS[video_idx]
    bg_rgb     = BG_COLORS[video_idx]

    label = {
        "dataset_type":    "static",
        "pos_idx":         pos_idx,
        "variant_idx":     variant_idx,
        "start_pos_xy":    list(start_xy),
        "video_idx":       video_idx,
        "ball_color_rgba": [round(c, 4) for c in
                            [ball_color.r, ball_color.g, ball_color.b, ball_color.a]],
        "speed_mps":       0.0,
        "speed_profile":   [0.0] * (FRAME_END + 1),
    }

    out     = pathlib.Path(output_dir)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="kubric_static_"))

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

    animation, _ = run_static(simulator, scene, ball)

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
        lbl = m["label"]
        rows.append({
            "video":       meta_path.parent.name,
            "video_idx":   lbl["video_idx"],
            "pos_idx":     lbl["pos_idx"],
            "variant_idx": lbl["variant_idx"],
            "start_x":     lbl["start_pos_xy"][0],
            "start_y":     lbl["start_pos_xy"][1],
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
    p.add_argument("--output_dir", required=True)
    p.add_argument("--width",      type=int, default=256)
    p.add_argument("--height",     type=int, default=256)
    p.add_argument("--pos_idx",    type=int, default=-1,
                   help="Generate only this position index (0-5). Default -1 = all.")
    p.add_argument("--variant_idx", type=int, default=-1,
                   help="Generate only this variant index (0-4). Default -1 = all.")
    return p.parse_args()


def main():
    args = parse_args()
    resolution = (args.width, args.height)

    pos_range = [args.pos_idx]     if args.pos_idx     >= 0 else range(N_POSITIONS)
    var_range = [args.variant_idx] if args.variant_idx >= 0 else range(N_VARIANTS)

    # Build global video_idx map: pos 0 var 0 = 0, pos 0 var 1 = 1, ...
    global_idx_map = {}
    idx = 0
    for p in range(N_POSITIONS):
        for v in range(N_VARIANTS):
            global_idx_map[(p, v)] = idx
            idx += 1

    pairs = [(p, v) for p in pos_range for v in var_range]
    logger.info("Generating %d videos in %s", len(pairs), args.output_dir)

    for count, (p, v) in enumerate(pairs, 1):
        tag = f"pos{p:02d}_v{v:02d}"
        out = pathlib.Path(args.output_dir) / tag
        if (out / "metadata.json").exists():
            logger.info("Skipping %s (already done)", tag)
            continue
        video_idx = global_idx_map[(p, v)]
        logger.info("[%d/%d] %s  start=%s  variant=%d  video_idx=%d",
                    count, len(pairs), tag, START_POSITIONS[p], v, video_idx)
        generate_video(p, v, video_idx, str(out), resolution)

    write_dataset_docs(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
