# Praxis Guard: Deterministic Edge Safety & Compliance Control Plane for VLA Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenVINO](https://img.shields.io/badge/Intel-OpenVINO_2026.3-blue.svg)](https://docs.openvino.ai/)
[![Speechmatics](https://img.shields.io/badge/Speechmatics-Streaming--RT-red.svg)](https://www.speechmatics.com/)
[![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo_3.1+-green.svg)](https://mujoco.org/)

**Tracks:**

* **Primary:** Intel Physical AI Online Challenge
* **Bonus Award:** Best Use of Speechmatics

---

## The Problem: The Action-Horizon Blind Spot

Modern manipulation policies such as ACT, SmolVLA, and Pi0.5 predict motor trajectories in chunks:

$$
a = [a_t, \dots, a_{t+k}]
$$

typically with $k = 50$ steps.

While action chunking stabilizes high-frequency multi-joint control, it introduces an **action-horizon blind spot**: once a trajectory chunk has been generated, the robot can continue executing it even when a dynamic workspace hazard appears during execution.

**Praxis Guard** addresses this gap.

We do not compete on generating the largest foundation model. Instead, we build the **safety infrastructure required to supervise and interrupt VLA systems running on edge hardware**.

---

## Technical Scope: Safety First

Praxis Guard decouples **trajectory generation** from **deterministic safety arbitration**.

A lightweight OpenVINO-optimized proxy policy drives coordinated dual SO-101 manipulation in MuJoCo, while the Praxis Guard control plane supervises execution and intercepts spoken natural-language corrections through Speechmatics streaming ASR.

### Core Capabilities

* **Speechmatics Streaming Integration:** A WebSocket streaming client interfaces with the Speechmatics real-time ASR endpoint for low-latency speech recognition.

* **Deterministic Voice-Interrupt Gate:** A priority-ordered intervention taxonomy is implemented:

  ```text
  HALT > CLARIFY > REDIRECT > MODIFY
  ```

  with negation awareness:

  ```text
  "Don't stop now"                → No HALT trigger

  "Stop! The glass is slipping"  → HALT
  ```

* **Instant Action Flush:** Safety-critical spoken hazards bypass normal policy inference, invalidate the active action chunk, and inject a zero-velocity hold.

* **Cryptographic Compliance Ledger:** Safety interventions are signed and hash-chained:

  $$
  H_i = \operatorname{SHA256}(H_{i-1} \parallel S_i)
  $$

  creating an append-only, tamper-evident event history for auditing and reproducibility.

* **Baseline Verification Policy:** The safety plane is evaluated using a lightweight, **10-seed domain-randomized behavioral-cloning policy** compiled to OpenVINO Intermediate Representation (IR).

  > **Scope note:** Vision and language inputs are represented as embedded heuristic proxies in this build. This intentionally isolates the evaluation of inference latency, safety arbitration, action interruption, and manipulation robustness rather than foundation-model-scale semantic generalization.

---

## System Architecture

```text
                         ┌─────────────────────────┐
                         │   Goal / Task Context   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   OpenVINO Proxy VLA    │
                         │   Behavioral Cloning    │
                         └────────────┬────────────┘
                                      │
                                      │ Action Chunk
                                      │ k = 50
                                      ▼
              ┌──────────────────────────────────────────────┐
              │               PRAXIS GUARD                   │
              │                                              │
              │       Deterministic Safety Arbiter           │
              │                                              │
              │  HALT > CLARIFY > REDIRECT > MODIFY          │
              └──────────────┬───────────────────┬───────────┘
                             │                   │
                   Safe Action│                   │Voice Interrupt
                             │                   │
                             ▼                   ▼
                   ┌─────────────────┐    ┌──────────────────┐
                   │  Motor Command  │    │ Speechmatics ASR │
                   │     Path        │    │  Streaming Input │
                   └────────┬────────┘    └────────┬─────────┘
                            │                      │
                            │                      │ Intent
                            │                      ▼
                            │             ┌──────────────────┐
                            │             │ Intent Classifier│
                            │             └────────┬─────────┘
                            │                      │
                            └──────────┬───────────┘
                                       ▼
                         ┌─────────────────────────┐
                         │   Dual SO-101 MuJoCo   │
                         │     Manipulation Sim    │
                         │                         │
                         │ Plate · Mug · Fork     │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │  Cryptographic Audit    │
                         │        Ledger            │
                         └─────────────────────────┘
```

---

## Safety Control Flow

Praxis Guard operates as an independent supervisory layer between trajectory generation and actuation.

```text
                    VLA generates action chunk
                              │
                              ▼
                     ┌─────────────────┐
                     │  Praxis Guard   │
                     │     Arbiter     │
                     └────────┬────────┘
                              │
                ┌─────────────┼──────────────┐
                │             │              │
                ▼             ▼              ▼
              SAFE          MODIFY         REDIRECT
                │             │              │
                │             └──────┬───────┘
                │                    │
                ▼                    ▼
          Execute action       Updated action
                                          
                              ┌──────────────┐
                              │   CLARIFY    │
                              └──────┬───────┘
                                     │
                                     ▼
                              Pause / Context

                              ┌──────────────┐
                              │     HALT     │
                              └──────┬───────┘
                                     │
                                     ▼
                              Flush action chunk
                                     │
                                     ▼
                             Zero-velocity hold
                                     │
                                     ▼
                               Audit event
```

The critical property is that **HALT does not require another VLA inference**.

The currently executing trajectory is invalidated at the supervisory layer, allowing the safety mechanism to operate independently of the policy's action-generation cadence.

---

## The Action-Horizon Problem

At a 100 Hz control frequency:

$$
T_{\text{step}} = \frac{1}{100} = 10\text{ ms}
$$

For an action chunk containing 50 steps:

$$
T_{\text{chunk}} = 50 \times 10\text{ ms} = 500\text{ ms}
$$

This creates a potential **500 ms action horizon**.

```text
Without Praxis Guard

Policy ──► [ Action Chunk: 50 steps ] ───────────────────►
                         500 ms
                             
                         Hazard!
                            │
                            X
                     Policy may still
                     be executing chunk


With Praxis Guard

Policy ──► [ Action Chunk: 50 steps ] ───────────────────►
                         │
                         │ Hazard detected
                         ▼
                    Praxis Guard
                         │
                         ▼
                   FLUSH CHUNK
                         │
                         ▼
                ZERO-VELOCITY HOLD
```

The safety layer therefore addresses a structural property of chunked action execution rather than attempting to improve the underlying policy's semantic intelligence.

---

## Empirical Benchmark & Hardware Profile

### Hardware Portability Note

Benchmarks were captured directly using:

```text
benchmarks/intel_benchmark.py
```

Empirical execution was recorded on an **x86 host environment** using the OpenVINO 2026.3 runtime.

The policy is compiled into OpenVINO Intermediate Representation:

```text
.xml + .bin
```

This model representation is designed for deployment across supported OpenVINO execution devices, including Intel Core Ultra CPU, GPU/iGPU, and NPU targets.

> **Important:** The measurements below are host-runtime measurements. They are **not measurements taken directly on Intel Core Ultra hardware**. Intel NPU latency should be validated on the target Core Ultra system.

### OpenVINO Inference & Compute Headroom

| Metric                              | Measured Baseline Proxy Result | Intel Core Ultra Deployment Target |
| ----------------------------------- | -----------------------------: | ---------------------------------- |
| **Baseline Policy Inference (p50)** |                   **0.058 ms** | < 3.0 ms target                    |
| **Baseline Policy Inference (p99)** |                   **0.128 ms** | < 6.0 ms target                    |
| **Throughput**                      |    **17,252.4 inferences/sec** | High-efficiency edge serving       |
| **Control Loop Margin**             |                  **499.94 ms** | > 490 ms target                    |
| **Interrupt Response**              |      **Immediate chunk flush** | < 1.0 ms deterministic HALT target |

---

## 10-Seed Domain Perturbation & Robustness

The manipulation system is evaluated over 10 randomized seeds using:

* Dynamic object mass: $\pm25%$
* Surface friction: $\pm20%$
* Initial coordinate / object-placement perturbation

The evaluation is not based on a stationary or zero-control baseline.

**Pass criteria require the robot to actively move the plate beyond its starting-coordinate tolerance and successfully reach the target destination.**

### Result

**Overall Task Success Rate: 100.0% — 10 / 10 seeds passed**

| Metric               |     Result |
| -------------------- | ---------: |
| Evaluation Seeds     |     **10** |
| Successful Seeds     |     **10** |
| Failed Seeds         |      **0** |
| Overall Task Success | **100.0%** |

---

## Quickstart

### 1. Setup Environment

```bash
git clone https://github.com/<your-org>/praxis-guard.git
cd praxis-guard

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the End-to-End System

Execute the complete bimanual orchestration loop with the active safety supervisor:

```bash
python src/main.py \
  --scene config/dinner_scene.xml \
  --model-path models/openvino_ir/vla_policy_fp16.xml \
  --device CPU
```

For supported Intel hardware, OpenVINO's device selection can be configured for the available execution backend.

For example:

```bash
python src/main.py \
  --scene config/dinner_scene.xml \
  --model-path models/openvino_ir/vla_policy_fp16.xml \
  --device AUTO
```

### 3. Reproduce the Robustness Benchmark

Run the 10-seed domain-randomization evaluation:

```bash
python benchmarks/evaluate_robustness.py \
  --scene config/dinner_scene.xml \
  --model-path models/openvino_ir/vla_policy_fp16.xml
```

### 4. Run the OpenVINO Benchmark

Run the latency and throughput benchmark:

```bash
python benchmarks/intel_benchmark.py \
  --model-path models/openvino_ir/vla_policy_fp16.xml \
  --device CPU
```

---

## Repository Structure

```text
praxis-guard/
├── config/
│   └── dinner_scene.xml
│       # Dual SO-101 MuJoCo manipulation scene
│
├── models/
│   └── openvino_ir/
│       ├── vla_policy_fp16.xml
│       └── vla_policy_fp16.bin
│       # OpenVINO IR behavioral-cloning policy
│
├── src/
│   ├── main.py
│   │   # End-to-end VLA + Praxis Guard + MuJoCo runtime
│   │
│   ├── arbiter_engine.py
│   │   # Deterministic safety arbitration and action interruption
│   │
│   ├── audit_logger.py
│   │   # SHA-256 hash-chained compliance ledger
│   │
│   ├── expert_demonstrator.py
│   │   # Demonstration / trajectory generation
│   │
│   ├── train.py
│   │   # Behavioral cloning training
│   │
│   ├── vla_model.py
│   │   # Lightweight proxy policy
│   │
│   ├── voice_pipeline.py
│   │   # Speechmatics streaming ASR + intent classification
│   │
│   └── vision_pipeline.py
│       # Visual-state processing and heuristic proxy inputs
│
├── benchmarks/
│   ├── evaluate_robustness.py
│   │   # 10-seed domain-randomized evaluation
│   │
│   └── intel_benchmark.py
│       # OpenVINO latency and throughput benchmark
│
├── requirements.txt
└── LICENSE
```

---

## Technology Stack

| Component              | Technology                                 |
| ---------------------- | ------------------------------------------ |
| **Simulation**         | MuJoCo                                     |
| **Robot**              | Dual SO-101                                |
| **Policy**             | Lightweight Behavioral Cloning / VLA Proxy |
| **Inference Runtime**  | OpenVINO 2026.3                            |
| **Model Precision**    | FP16                                       |
| **Speech Recognition** | Speechmatics Streaming ASR                 |
| **Safety Arbitration** | Praxis Guard                               |
| **Control Frequency**  | 100 Hz                                     |
| **Action Chunk**       | 50 steps                                   |
| **Audit Mechanism**    | SHA-256 Hash Chaining                      |
| **Runtime**            | Python                                     |

---

## Design Principle

Praxis Guard deliberately separates **intelligence** from **authority**.

```text
┌───────────────────────────────────────────────────────┐
│                  TRAJECTORY GENERATION                │
│                                                       │
│  VLA / Policy                                         │
│                                                       │
│  "What should the robot do?"                          │
│                                                       │
│  Generates action chunks                              │
└──────────────────────────┬────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────┐
│                    SAFETY AUTHORITY                    │
│                                                       │
│  Praxis Guard                                         │
│                                                       │
│  "Is it safe to continue?"                            │
│                                                       │
│  • Deterministic arbitration                          │
│  • Voice interruption                                 │
│  • Action-chunk invalidation                           │
│  • Zero-velocity hold                                 │
│  • Cryptographic audit                                │
└──────────────────────────┬────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────┐
│                      ACTUATION                         │
│                                                       │
│  Dual SO-101 MuJoCo Environment                       │
└───────────────────────────────────────────────────────┘
```

The VLA decides **what to do**.

Praxis Guard retains authority over **whether execution should continue**.

That separation is the core design of the system.

---

## License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.
