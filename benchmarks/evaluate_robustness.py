import sys
from pathlib import Path
ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse
import numpy as np
import mujoco
from src.main import OpenVINOVLAController

def randomize_scene(model: mujoco.MjModel, seed: int):
    rng = np.random.default_rng(seed)
    plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    if plate_id != -1:
        model.body_pos[plate_id][0] += rng.uniform(-0.04, 0.04)
        model.body_pos[plate_id][1] += rng.uniform(-0.04, 0.04)

    mug_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
    if mug_id != -1:
        model.body_mass[mug_id] *= rng.uniform(0.75, 1.25)

    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    if table_id != -1:
        model.geom_friction[table_id][0] *= rng.uniform(0.8, 1.2)

def evaluate_robustness(scene_path: str, model_path: str, num_seeds: int = 10):
    vla = OpenVINOVLAController(model_path, device="CPU")
    lang_goal = np.ones(32, dtype=np.float32) * 0.5
    successes = 0

    print("=" * 65)
    print(f"INTEL CHALLENGE EVALUATION: 10-SEED ROBUSTNESS & PERTURBATION")
    print("=" * 65)

    for seed in range(num_seeds):
        model = mujoco.MjModel.from_xml_path(scene_path)
        randomize_scene(model, seed)
        data = mujoco.MjData(model)

        for step in range(160):
            obs = np.concatenate([data.qpos[:14], np.zeros(64, dtype=np.float32)])
            action_chunk = vla.predict_chunk(obs, lang_goal)
            data.ctrl[:14] = action_chunk[step % 50]
            mujoco.mj_step(model, data)

        # MENTOR FIX: True Task Success Metric
        # Check if the plate body actually reached the target table setting coordinate
        plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        if plate_id != -1:
            plate_pos = data.xpos[plate_id]
            target_pos = np.array([0.0, 0.15, 0.40]) # Example desired table placement
            distance = np.linalg.norm(plate_pos - target_pos)
            max_qvel = np.max(np.abs(data.qvel))
            
            # Success: Plate is within 10cm of target AND physics remained stable
            success = bool(distance < 0.10 and max_qvel < 15.0)
        else:
            success = False

        if success:
            successes += 1
            status = "PASSED"
        else:
            status = "FAILED"

        print(f"Seed {seed:02d} | Perturbation: Mass/Fric/Pos Randomized | Status: {status}")

    success_rate = (successes / num_seeds) * 100
    print("-" * 65)
    print(f"Final Robustness Result: {successes}/{num_seeds} Successes ({success_rate:.1f}%)")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="config/dinner_scene.xml")
    parser.add_argument("--model-path", default="models/openvino_ir/vla_policy_fp16.xml")
    args = parser.parse_args()
    evaluate_robustness(args.scene, args.model_path)