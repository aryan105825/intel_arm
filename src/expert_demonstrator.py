import argparse
import numpy as np
import mujoco
import os

TOTAL_JOINTS = 14  

def generate_smooth_trajectory(start: np.ndarray, goal: np.ndarray, steps: int) -> np.ndarray:
    t = np.linspace(0, 1, steps)[:, None]
    s = 10 * (t**3) - 15 * (t**4) + 6 * (t**5)
    return start + s * (goal - start)

def run_demonstration_episode(model_path: str, seed: int):
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    q0 = np.zeros(TOTAL_JOINTS)
    data.ctrl[:] = q0

    noise1 = rng.uniform(-0.05, 0.05, TOTAL_JOINTS)
    noise2 = rng.uniform(-0.05, 0.05, TOTAL_JOINTS)
    
    w1 = np.array([-0.3, 0.4, -0.6, 0.0, 0.3, 0.0, 0.02,   0.3, 0.4, -0.6, 0.0, 0.3, 0.0, 0.02]) + noise1
    w2 = np.array([-0.4, 0.7, -0.8, 0.0, 0.5, 0.0, -0.01,  0.4, 0.7, -0.8, 0.0, 0.5, 0.0, -0.01]) + noise2
    w3 = np.array([0.0, 0.3, -0.3, 0.0, 0.2, 0.0, -0.01,   0.1, 0.3, -0.3, 0.0, 0.2, 0.0, -0.01]) + noise1
    w4 = np.array([0.0, 0.1, -0.1, 0.0, 0.0, 0.0, 0.02,    0.0, 0.1, -0.1, 0.0, 0.0, 0.0, 0.02])

    seg1 = generate_smooth_trajectory(q0, w1, 40)
    seg2 = generate_smooth_trajectory(w1, w2, 40)
    seg3 = generate_smooth_trajectory(w2, w3, 50)
    seg4 = generate_smooth_trajectory(w3, w4, 30)

    full_plan = np.vstack([seg1, seg2, seg3, seg4])
    
    observations = []
    actions = []

    for step_idx in range(len(full_plan)):
        target_ctrl = full_plan[step_idx]
        data.ctrl[:] = target_ctrl
        mujoco.mj_step(model, data)

        mock_visual_feat = np.sin(np.linspace(0, 3.14, 64) * (data.qpos[0] + 1.0)).astype(np.float32)
        joint_state = np.array(data.qpos[:TOTAL_JOINTS], dtype=np.float32)

        observations.append(np.concatenate([joint_state, mock_visual_feat]))
        actions.append(target_ctrl.astype(np.float32))

    return np.array(observations), np.array(actions)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--output", type=str, default="data/demonstrations.npz")
    parser.add_argument("--scene", type=str, default="config/dinner_scene.xml")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    all_obs = []
    all_acts = []
    print(f"Generating {args.num_episodes} varied expert demonstrations...")
    for ep in range(args.num_episodes):
        obs, acts = run_demonstration_episode(args.scene, seed=ep)
        all_obs.append(obs)
        all_acts.append(acts)

    np.savez_compressed(args.output, observations=np.array(all_obs), actions=np.array(all_acts))
    print(f"Saved {args.num_episodes} diverse demonstrations to {args.output}")

if __name__ == "__main__":
    main()