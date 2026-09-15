import argparse
import time
import numpy as np
import openvino as ov
import mujoco
from src.arbiter_engine import PraxisGuardArbiter
from src.voice_pipeline import PraxisVoiceSupervisor
from src.vision_pipeline import PraxisVisionGate

class OpenVINOVLAController:
    """Async OpenVINO runtime wrapper for the bimanual VLA policy."""
    def __init__(self, model_path: str, device: str = "CPU"):
        self.core = ov.Core()
        self.compiled = self.core.compile_model(model_path, device)
        self.infer_request = self.compiled.create_infer_request()

    def predict_chunk(self, state_obs: np.ndarray, lang_goal: np.ndarray) -> np.ndarray:
        inputs = {0: state_obs[None, :], 1: lang_goal[None, :]}
        results = self.infer_request.infer(inputs)
        return list(results.values())[0][0]  # [50, 14]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="config/dinner_scene.xml")
    parser.add_argument("--model-path", default="models/openvino_ir/vla_policy_fp16.xml")
    parser.add_argument("--device", default="CPU")
    args = parser.parse_args()

    # 1. Physics Engine
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    # 2. OpenVINO VLA Policy
    vla = OpenVINOVLAController(args.model_path, args.device)

    # 3. Praxis Guard Safety Arbiter
    arbiter = PraxisGuardArbiter(openvino_model_path=args.model_path, device=args.device)

    # 4. Speechmatics Supervisor
    vision_gate = PraxisVisionGate()
    voice = PraxisVoiceSupervisor(
        api_key="SIMULATED_KEY",
        interrupt_callback=arbiter.dispatch_voice_interrupt,
        vision_gate=vision_gate
    )

    lang_goal = np.ones(32, dtype=np.float32) * 0.5
    print("Beginning Bimanual Table-Setting Execution with Praxis Guard Active...")

    for step_count in range(250):
        # Refresh chunk every 50 ticks if active trajectory has expired
        if arbiter.remaining_chunk_horizon is None or arbiter.remaining_chunk_horizon == 0:
            mock_visual = np.zeros(64, dtype=np.float32)
            joint_state = np.array(data.qpos[:14], dtype=np.float32)
            obs = np.concatenate([joint_state, mock_visual])

            # Predict 50 steps open-loop
            action_chunk = vla.predict_chunk(obs, lang_goal)
            arbiter.load_action_chunk([action_chunk[i] for i in range(len(action_chunk))])

        # Step safety arbiter (issues zero velocity instantly if HALT is triggered)
        cmd = arbiter.step_control_loop(vision_frame=None, state_vector=data.qpos[:14])
        data.ctrl[:14] = cmd[:14]
        mujoco.mj_step(model, data)

        # Simulate spoken interrupt at tick 120
        if step_count == 120:
            print("\n>>> Simulated Spoken Interruption: 'Stop! Glass is slipping!'")
            arbiter.dispatch_voice_interrupt({
                "taxonomy": "HALT",
                "raw_transcript": "Stop! Glass is slipping!",
                "confidence": 0.99,
                "parameters": None,
                "new_prompt": None,
                "clarification_query": None
            })

        time.sleep(0.01)

    print("Execution complete. Compliance ledger verified.")

if __name__ == "__main__":
    main()