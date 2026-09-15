import argparse
import time
import numpy as np
import openvino as ov

def run_intel_benchmark(model_path: str, device: str = "CPU", iterations: int = 200):
    core = ov.Core()
    print("=" * 65)
    print("INTEL CORE ULTRA HARDWARE ACCELERATION & INFERENCE BENCHMARK")
    print("=" * 65)
    print(f"OpenVINO Version : {ov.__version__}")
    print(f"Selected Device  : {device}")
    print(f"Target Hardware  : Intel Core Ultra Series 2/3 (NPU/iGPU/CPU Architecture)")
    print(f"Model Under Test : {model_path}")
    print("-" * 65)

    compiled = core.compile_model(model_path, device)
    infer_request = compiled.create_infer_request()

    dummy_state = np.random.randn(1, 78).astype(np.float32)
    dummy_lang = np.random.randn(1, 32).astype(np.float32)
    inputs = {0: dummy_state, 1: dummy_lang}

    # Warmup
    for _ in range(20):
        infer_request.infer(inputs)

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        infer_request.infer(inputs)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    throughput = 1000.0 / p50

    print(f"Inference Latency (p50) : {p50:.3f} ms")
    print(f"Inference Latency (p95) : {p95:.3f} ms")
    print(f"Inference Latency (p99) : {p99:.3f} ms")
    print(f"Throughput              : {throughput:.1f} inferences/sec")
    print(f"Action Chunk Horizon    : 50 steps (500 ms autonomous execution)")
    print(f"Control Loop Margin     : {(500.0 - p50):.2f} ms compute headroom")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/openvino_ir/vla_policy_fp16.xml")
    parser.add_argument("--device", default="CPU")
    args = parser.parse_args()
    run_intel_benchmark(args.model_path, args.device)