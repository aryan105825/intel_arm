import sys, os
from pathlib import Path
ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont

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
    ledger_path = Path("benchmarks/raw/compliance_ledger.jsonl")
    if ledger_path.exists():
        ledger_path.unlink()

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    camera = mujoco.MjvCamera()
    camera.azimuth = 90
    camera.elevation = -35
    camera.distance = 1.3
    camera.lookat = np.array([0.0, 0.05, 0.35])

    arbiter = PraxisGuardArbiter(openvino_model_path=MODEL_PATH, device="CPU")
    writer = imageio.get_writer(OUT_PATH, fps=FPS, quality=8)
    
    state_label = "ACTIVE"

    for step in range(TOTAL_STEPS):
        # DIRECT KINEMATIC OVERRIDE - Guarantees visual movement for the demo
        if not arbiter.emergency_stop_flag:
            progress = step / HALT_AT_STEP
            ctrl = np.zeros(14)
            # Sweep both arms inward and downward toward the objects
            ctrl[0] = 0.5 * np.sin(progress * np.pi / 2)
            ctrl[1] = 0.8 * np.sin(progress * np.pi / 2)
            ctrl[2] = -0.9 * np.sin(progress * np.pi / 2)
            
            ctrl[7] = -0.5 * np.sin(progress * np.pi / 2)
            ctrl[8] = 0.8 * np.sin(progress * np.pi / 2)
            ctrl[9] = -0.9 * np.sin(progress * np.pi / 2)
            
            data.ctrl[:14] = ctrl

        # TRIGGER HALT INTERRUPT
        if step == HALT_AT_STEP:
            arbiter.dispatch_voice_interrupt({
                "taxonomy": "HALT",
                "raw_transcript": "Stop! Glass is slipping!",
                "confidence": 0.99,
                "parameters": None, "new_prompt": None, "clarification_query": None
            })
            state_label = "HALT"
            # Once emergency_stop_flag flips to True, data.ctrl stops updating and holds its last position

        mujoco.mj_step(model, data)

        chain_hash = arbiter._ledger._last_chain_hash
        renderer.update_scene(data, camera=camera)
        big_frame = renderer.render()
        writer.append_data(hud_overlay(big_frame, step, state_label, chain_hash))

    writer.close()
    print(f"Saved {OUT_PATH} — {TOTAL_STEPS} frames at {FPS}fps")

if __name__ == "__main__":
    main()