# SafeVLA: Deterministic Edge Safety & Interruption Control Plane for Bimanual VLA Manipulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenVINO](https://img.shields.io/badge/Intel-OpenVINO_2026.3-blue.svg)](https://docs.openvino.ai/)
[![Speechmatics](https://img.shields.io/badge/Speechmatics-Streaming--RT-red.svg)](https://www.speechmatics.com/)
[![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo_3.1+-green.svg)](https://mujoco.org/)

**Tracks:**

* **Primary:** Intel Physical AI Online Challenge — Bimanual VLA Manipulation
* **Bonus Award:** Best Use of Speechmatics

---

## Executive Summary

Modern manipulation policies such as ACT, SmolVLA, and Pi0.5 predict motor trajectories in chunks:

$$
a = [a_t, \dots, a_{t+k}]
$$

typically with $k = 50$ steps.

While action chunking stabilizes high-frequency multi-joint control, it introduces an **action-horizon blind spot**: the robot can continue executing a previously generated chunk even when a dynamic workspace hazard appears during execution.

**SafeVLA** decouples *trajectory generation* from *deterministic safety arbitration*.

A lightweight OpenVINO-optimized bimanual Vision-Language-Action (VLA) baseline drives coordinated dual SO-101 manipulation in MuJoCo, while the **Praxis Guard** control plane intercepts spoken natural-language corrections through Speechmatics streaming ASR.

Safety-critical commands can immediately invalidate the active action chunk and inject a zero-velocity hold. Every intervention is cryptographically signed and recorded through a SHA-256 hash-chained append-only compliance ledger.

---

## Honest Technical Scope & Capabilities

Rather than deploying a massive foundation model, this submission focuses on a **highly optimized edge-safety architecture** built explicitly around Intel Core Ultra and Speechmatics integration.

### Deterministic Voice-Interrupt Gate

A priority-ordered intervention taxonomy is implemented:

```text
HALT > CLARIFY > REDIRECT > MODIFY
```

The voice pipeline includes negation awareness:

```text
"Don't stop now"                  → No HALT trigger

"Stop! The glass is slipping"    → HALT
```

Spoken safety hazards bypass normal policy inference and directly trigger the safety arbitration path, flushing the currently active action chunk and commanding a zero-velocity hold.

### Cryptographic Compliance Ledger

Every safety intervention is signed and hash-chained:

$$
H_i = \operatorname{SHA256}(H_{i-1} \parallel S_i)
$$

where each event incorporates the cryptographic state of the previous event.

The resulting append-only ledger provides a tamper-evident record of safety interventions for post-run auditing and reproducibility.

### OpenVINO Edge Optimization

The behavioral-cloning policy is converted and compiled to **OpenVINO Intermediate Representation (IR)** using FP16 precision.

The resulting `.xml` / `.bin` model representation is designed for deployment across supported OpenVINO execution devices, including Intel Core Ultra CPU, GPU/iGPU, and NPU targets where supported.

### Baseline VLA Simulation

To isolate and evaluate the safety plane, the system uses a lightweight **50-seed domain-randomized behavioral-cloning policy**.

> **Scope note:** Vision and language inputs are represented by embedded heuristic proxies in this build. The primary objective is therefore deterministic safety arbitration, interruption control, edge inference, and reproducible manipulation evaluation rather than foundation-model-scale semantic generalization.

---

## System Architecture

```text
                         [ Synthesized Goal Vector ]
                                    │
                                    ▼
        [ Simulated State Feed ] ──►┌──────────────────────────────────────┐
                                    │        OpenVINO VLA Baseline         │
        [ 14-DOF Joint Telemetry ]─►│   Multi-Modal Behavioral Cloning    │
                                    └─────────────────┬────────────────────┘
                                                      │
                                                      │ Predicted Action Chunk
                                                      │ k = 50
                                                      ▼
 [ Speechmatics Streaming ASR ] ──►┌──────────────────────────────────────┐
     "Stop! Glass is slipping!"    │         Praxis Guard Arbiter         │
                                   │    Deterministic Safety Interceptor  │
                                   └─────────────────┬────────────────────┘
                                                     │
                                                     │ Safe Motor Commands
                                                     │ 100 Hz
                                                     ▼
                                    ┌──────────────────────────────────────┐
                                    │        Dual SO-101 MuJoCo Sim        │
                                    │                                      │
                                    │ Plate · Mug · Fork · Shared Table   │
                                    └──────────────────────────────────────┘
```

---

## Safety Control Flow

The central control path is:

```text
                    ┌──────────────────┐
                    │   OpenVINO VLA   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Action Chunk    │
                    │    k = 50        │
                    └────────┬─────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │     Praxis Guard       │
                 │ Deterministic Arbiter  │
                 └───────────┬────────────┘
                             │
              ┌──────────────┼───────────────┐
              │              │               │
              ▼              ▼               ▼
           SAFE           MODIFY          REDIRECT
              │              │               │
              │              └──────┬────────┘
              │                     │
              ▼                     ▼
         Execute action        Updated action
                                   
                             ┌───────────────┐
                             │   CLARIFY     │
                             └───────┬───────┘
                                     │
                                     ▼
                              Pause / Context

                             ┌───────────────┐
                             │     HALT      │
                             └───────┬───────┘
                                     │
                                     ▼
                             Flush action chunk
                                     │
                                     ▼
                           Zero-velocity hold
                                     │
                                     ▼
                              Audit Ledger
```

The key architectural property is that **Praxis Guard is an independent deterministic supervisory layer**, rather than another generative model.

---

## Empirical Benchmark & Hardware Profile

### Hardware Context & Portability Note

Benchmarks were captured directly using the test harness:

```text
benchmarks/intel_benchmark.py
```

Empirical execution was recorded on an **x86 host environment** using the OpenVINO 2026.3 execution runtime.

The policy is compiled into OpenVINO Intermediate Representation:

```text
.xml + .bin
```

This representation is intended for portable deployment across supported OpenVINO devices. The same model artifact can be targeted toward supported Intel Core Ultra CPU, GPU/iGPU, or NPU execution backends.

> **Important:** The host measurements below are OpenVINO runtime measurements and should not be interpreted as measurements performed on Intel Core Ultra hardware. Intel-specific NPU latency should be validated directly on the target Core Ultra system.

### OpenVINO Inference & Compute Headroom

| Metric                                 | Measured Result — OpenVINO IR Runtime | Intel Core Ultra Deployment Target |
| -------------------------------------- | ------------------------------------: | ---------------------------------- |
| **VLA Policy Inference Latency (p50)** |                          **0.058 ms** | < 3.0 ms NPU target                |
| **VLA Policy Inference Latency (p99)** |                          **0.128 ms** | < 6.0 ms target                    |
| **Throughput**                         |           **17,252.4 inferences/sec** | High-efficiency edge serving       |
| **Action Chunk**                       |                          **50 steps** | 50 steps                           |
| **Control Frequency**                  |                            **100 Hz** | 100 Hz                             |
| **Action Horizon**                     |                            **500 ms** | 500 ms                             |
| **Compute Headroom**                   |                         **499.94 ms** | > 490 ms target                    |
| **Interrupt Response**                 |         **Instantaneous chunk flush** | < 1 ms deterministic HALT target   |

### Action-Horizon Calculation

The control loop operates at 100 Hz:

$$
T_{\text{step}} = \frac{1}{100} = 10\text{ ms}
$$

For a 50-step action chunk:

$$
T_{\text{chunk}} = 50 \times 10\text{ ms} = 500\text{ ms}
$$

Therefore, the policy can generate an action horizon spanning approximately **500 ms**.

Praxis Guard operates outside this trajectory-generation horizon. A safety intervention can invalidate the active chunk rather than waiting for the next policy inference.

---

## 10-Seed Domain Perturbation & Robustness

The manipulation system was evaluated across **10 randomized seeds** using:

* Object mass perturbation: $\pm25%$
* Surface friction perturbation: $\pm20%$
* Initial coordinate / object-placement perturbation
* Active manipulation rather than a stationary-control baseline

The pass criteria require the robot to **actively move the plate beyond its starting-coordinate tolerance and reach the target destination**.

|        Seed | Mass | Friction | Placement  | Active Manipulation | Result     |
| ----------: | ---: | -------: | ---------- | ------------------- | ---------- |
|      **00** | ±25% |     ±20% | Randomized | Passed              | **PASSED** |
|      **01** | ±25% |     ±20% | Randomized | Passed              | **PASSED** |
|      **02** | ±25% |     ±20% | Randomized | Passed              | **PASSED** |
|      **03** | ±25% |     ±20% | Randomized | Passed              | **PASSED** |
|      **04** | ±25% |     ±20% | Randomized | Passed              | **PASSED** |
|      **05** | ±25% |     ±20% | Randomized | Passed              | **PASSED** |
|      **06** | ±25% |     ±20% | Randomized | Passed              | **PASSED** |
|      **07** | ±25% |     ±20% | Randomized | Passed              | **PASSED** |
|      **08** | ±25% |     ±20% | Randomized | Passed              | **PASSED** |
|      **09** | ±25% |     ±20% | Randomized | Passed              | **PASSED** |
| **Overall** |    — |        — | —          | **10 / 10**         | **100.0%** |

### Result

**Overall Task Success Rate: 100.0% — 10 / 10 seeds passed**

---

## Quickstart

### 1. Setup Environment

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/<your-org>/safe-vla.git
cd safe-vla

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the End-to-End System

Execute the full bimanual orchestration loop with the active safety supervisor:

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

### 4. Run the OpenVINO Latency Benchmark

Run the inference latency and throughput benchmark:

```bash
python benchmarks/intel_benchmark.py \
  --model-path models/openvino_ir/vla_policy_fp16.xml \
  --device CPU
```

---

## Repository Structure

```text
safe-vla/
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
│   │   # Deterministic safety arbitration and interrupt state machine
│   │
│   ├── audit_logger.py
│   │   # SHA-256 hash-chained append-only audit ledger
│   │
│   ├── expert_demonstrator.py
│   │   # Demonstration / trajectory generation
│   │
│   ├── train.py
│   │   # Behavioral cloning training
│   │
│   ├── vla_model.py
│   │   # Lightweight multimodal policy
│   │
│   ├── voice_pipeline.py
│   │   # Speechmatics streaming ASR + intent classification
│   │
│   └── vision_pipeline.py
│       # Visual-state processing and heuristic proxy inputs
│
├── benchmarks/
│   ├── evaluate_robustness.py
│   │   # Domain-randomized manipulation evaluation
│   │
│   └── intel_benchmark.py
│       # OpenVINO latency and throughput benchmark
│
├── requirements.txt
└── LICENSE
```

---

## Technology Stack

| Layer                  | Technology                           |
| ---------------------- | ------------------------------------ |
| **Simulation**         | MuJoCo                               |
| **Robot**              | Dual SO-101                          |
| **Policy**             | Lightweight VLA / Behavioral Cloning |
| **Inference Runtime**  | OpenVINO 2026.3                      |
| **Model Precision**    | FP16                                 |
| **Speech Recognition** | Speechmatics Streaming ASR           |
| **Safety Arbitration** | Praxis Guard                         |
| **Control Frequency**  | 100 Hz                               |
| **Audit Mechanism**    | SHA-256 Hash Chaining                |
| **Runtime**            | Python                               |

---

## Core Design Principle

SafeVLA separates two fundamentally different responsibilities:

```text
┌─────────────────────────────────────────────────────────┐
│                    TRAJECTORY LAYER                     │
│                                                         │
│  VLA Policy                                              │
│  "What should the robot do?"                            │
│                                                         │
│  Generates action chunks                                │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    SAFETY LAYER                         │
│                                                         │
│  Praxis Guard                                            │
│  "Is it safe to continue executing?"                    │
│                                                         │
│  Deterministic arbitration                              │
│  Voice interruption                                     │
│  Chunk invalidation                                      │
│  Zero-velocity hold                                      │
│  Cryptographic audit                                     │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    ACTUATION LAYER                      │
│                                                         │
│  Dual SO-101 MuJoCo Environment                          │
└─────────────────────────────────────────────────────────┘
```

This separation allows a lightweight generative manipulation policy to operate normally while retaining an independent, deterministic mechanism capable of overriding its execution when a safety-critical intervention occurs.

---

## License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for the complete license text.
