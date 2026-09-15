import sys, os
from pathlib import Path
ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont

from src.main import OpenVINOVLAController
from src.arbiter_engine import PraxisGuardArbiter

SCENE = "config/dinner_scene.xml"
MODEL_PATH = "models/openvino_ir/vla_policy_fp16.xml"
OUT_PATH = "demo_footage_raw.mp4"
WIDTH, HEIGHT = 1280, 720
FPS = 30
HALT_AT_STEP = 120
TOTAL_STEPS = 250

def hud_overlay(frame_rgb, step, arbiter_state, chain_hash):
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 22)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    pad = 16
    box_h = 92
    draw.rectangle([0, 0, WIDTH, box_h], fill=(0, 0, 0, 180))

    color = (255, 60, 60) if arbiter_state == "HALT" else (60, 220, 120)
    draw.text((pad, 10), f"PRAXIS GUARD: {arbiter_state}", font=font, fill=color)
    draw.text((pad, 40), f"step {step:04d} / {TOTAL_STEPS}", font=font_small, fill=(220, 220, 220))
    draw.text((pad, 62), f"ledger chain_hash: {chain_hash[:26]}...", font=font_small, fill=(180, 180, 255))
    return np.array(img)

def main():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    camera = mujoco.MjvCamera()
    camera.azimuth = 90
    camera.elevation = -35
    camera.distance = 1.3
    camera.lookat = np.array([0.0, 0.05, 0.35])

    vla = OpenVINOVLAController(MODEL_PATH, device="CPU")
    arbiter = PraxisGuardArbiter(openvino_model_path=MODEL_PATH, device="CPU")
    lang_goal = np.ones(32, dtype=np.float32) * 0.5
    
    writer = imageio.get_writer(OUT_PATH, fps=FPS, quality=8)
    state_label = "ACTIVE"

    for step in range(TOTAL_STEPS):
        # Use the exact synthetic visual feature the model was trained on
        visual_embedding = np.sin(np.linspace(0, 3.14, 64) * (data.qpos[0] + 1.0)).astype(np.float32)

        if arbiter.remaining_chunk_horizon is None or arbiter.remaining_chunk_horizon == 0:
            joint_state = np.array(data.qpos[:14], dtype=np.float32)
            obs = np.concatenate([joint_state, visual_embedding])
            action_chunk = vla.predict_chunk(obs, lang_goal)
            arbiter.load_action_chunk([action_chunk[i] for i in range(len(action_chunk))])

        cmd = arbiter.step_control_loop(vision_frame=None, state_vector=data.qpos[:14])
        data.ctrl[:14] = cmd[:14]
        mujoco.mj_step(model, data)

        if step == HALT_AT_STEP:
            arbiter.dispatch_voice_interrupt({
                "taxonomy": "HALT",
                "raw_transcript": "Stop! Glass is slipping!",
                "confidence": 0.99,
                "parameters": None, "new_prompt": None, "clarification_query": None
            })
            state_label = "HALT"

        chain_hash = arbiter._ledger._last_chain_hash
        renderer.update_scene(data, camera=camera)
        big_frame = renderer.render()
        writer.append_data(hud_overlay(big_frame, step, state_label, chain_hash))

    writer.close()

if __name__ == "__main__":
    main()