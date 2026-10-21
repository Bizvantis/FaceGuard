"""Convert the CVPR 2024 workshop PAD checkpoints (He et al., arXiv:2404.08450) to ONNX.

Run once, in a separate environment with PyTorch + timm (the FaceGuard venv needs only onnxruntime):

    uv venv convert_env --python 3.12
    uv pip install --python convert_env torch torchvision --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python convert_env timm onnx onnxscript onnxruntime numpy
    git clone https://github.com/Xianhua-He/cvpr2024-face-anti-spoofing-challenge <repo>
    convert_env/Scripts/python.exe pad_convert.py <repo>
    convert_env/Scripts/python.exe pad_convert.py --macs        (multiply-accumulates and parameters)

Reads data/pad/full_resnet50.pth and data/pad/face_swin_v2_base.pth and writes data/pad/*.onnx, whose single
output is the softmax over (live, spoof).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PAD = Path(__file__).resolve().parent / "data" / "pad"
MODELS = {"full_resnet50": "resnet50", "face_swin_v2_base": "swin_v2_b"}


class Softmax(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        logits = out[1] if isinstance(out, (tuple, list)) else out  # the repo's nets return (features, logits)
        return torch.softmax(logits, dim=-1)


def load(arch: str, path: Path):
    from nets.utils import get_model

    model = get_model(arch, 2)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get("state_dict_ema") is not None:
        state = dict(ckpt["state_dict_ema"])
        state.pop("n_averaged", None)
        print("  using the EMA weights")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    clean = {}
    for k, v in state.items():
        for prefix in ("module.module.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        clean[k] = v
    print(" ", model.load_state_dict(clean, strict=True))
    return Softmax(model).eval()


def main(repo: str) -> None:
    import types

    sys.path.insert(0, repo)
    sys.modules.setdefault("thop", types.SimpleNamespace(profile=None))  # imported but unused by nets/shufflenetv2
    torch.Tensor.cuda = lambda self, *a, **k: self  # nets/swin_transformer_v2 moves a constant to the GPU
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    only = sys.argv[2:] or list(MODELS)
    for name, arch in MODELS.items():
        if name not in only:
            continue
        src = PAD / f"{name}.pth"
        if not src.exists():
            print(f"missing {src}")
            continue
        print(name)
        model = load(arch, src)
        x = torch.from_numpy(rng.normal(size=(1, 3, 224, 224)).astype(np.float32))
        dst = PAD / f"{name}.onnx"
        torch.onnx.export(model, (x,), str(dst), input_names=["input"], output_names=["prob"], opset_version=17,
                          dynamo=False)
        with torch.no_grad():
            ref = model(x).numpy()
        out = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
        print(f"  {dst.name}: {dst.stat().st_size / 1e6:.1f} MB, max |torch - onnx| = {np.abs(ref - out).max():.2e}")
        print(f"  parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")


def macs(path: Path) -> tuple[float, float]:
    """Multiply-accumulates of Conv / Gemm / MatMul nodes (from ONNX shape inference) and parameter count."""
    import onnx
    from onnx import numpy_helper, shape_inference

    model = shape_inference.infer_shapes(onnx.load(str(path)))
    g = model.graph
    shapes = {}
    for v in list(g.value_info) + list(g.input) + list(g.output):
        dims = [d.dim_value if d.HasField("dim_value") else 1 for d in v.type.tensor_type.shape.dim]
        shapes[v.name] = dims
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for k, a in inits.items():
        shapes[k] = list(a.shape)
    total = 0.0
    for n in g.node:
        if n.op_type == "Conv":
            w = shapes[n.input[1]]
            out = shapes.get(n.output[0])
            if out:
                total += float(np.prod(out)) * w[1] * float(np.prod(w[2:]))
        elif n.op_type in ("Gemm", "MatMul"):
            a, b = shapes.get(n.input[0]), shapes.get(n.input[1])
            out = shapes.get(n.output[0])
            if a and out:
                k = a[-1]
                total += float(np.prod(out)) * k
    params = sum(float(np.prod(a.shape)) for a in inits.values())
    return total, params


def report_macs() -> None:
    for f in ("MiniFASNetV2.onnx", "MiniFASNetV1SE.onnx", "anti-spoof-mn3.onnx", "full_resnet50.onnx",
              "face_swin_v2_base.onnx"):
        m, p = macs(PAD / f)
        print(f"{f:26s} {m / 1e9:7.3f} GMACs  {p / 1e6:7.2f} M parameters")


if __name__ == "__main__":
    if sys.argv[1] == "--macs":  # pad_convert.py --macs : cost of every PAD model (needs the onnx package)
        report_macs()
    else:
        main(sys.argv[1])
