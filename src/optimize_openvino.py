import argparse
from pathlib import Path
import torch
import openvino as ov
from src.vla_model import MultiModalVLAPolicy

def export_openvino(weights_path: str, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = MultiModalVLAPolicy(obs_dim=78, lang_dim=32, action_dim=14, chunk_size=50)
    if Path(weights_path).exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    dummy_state = torch.randn(1, 78, dtype=torch.float32)
    dummy_lang = torch.randn(1, 32, dtype=torch.float32)

    print("Converting PyTorch model to OpenVINO IR...")
    ov_model = ov.convert_model(model, example_input=(dummy_state, dummy_lang))

    # 1. Save FP16 IR
    fp16_path = out_dir / "vla_policy_fp16.xml"
    ov.save_model(ov_model, str(fp16_path), compress_to_fp16=True)
    print(f"Exported OpenVINO FP16 Model -> {fp16_path}")

    # 2. Save FP32 IR
    fp32_path = out_dir / "vla_policy_fp32.xml"
    ov.save_model(ov_model, str(fp32_path), compress_to_fp16=False)
    print(f"Exported OpenVINO FP32 Model -> {fp32_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="models/vla_policy.pt")
    parser.add_argument("--output-dir", default="models/openvino_ir")
    args = parser.parse_args()
    export_openvino(args.input, args.output_dir)