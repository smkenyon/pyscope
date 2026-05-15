"""Object-detection inference under pyscope with clear preprocess / infer / postprocess scopes.

Uses a torchvision-pretrained Faster R-CNN (no real images — we feed synthetic
tensors so the example works offline). What you should see in the summary:

  preprocess     — short, mostly CPU
  inference_loop — bulk of GPU energy/util; 100 segments named `infer_one`
                   under it (parent_id = inference_loop's id), each ~10–30 ms
  postprocess    — short, CPU-only

Run with:
    uv sync --extra cpu-gpu-bench
    uv run pyscope --output ./out examples/torchvision_detector.py
"""

from __future__ import annotations

import sys

import pyscope


def _require_deps():
    try:
        import torch
        import torchvision
    except ImportError:
        print(
            "This example needs torch + torchvision. "
            "Install: uv sync --extra cpu-gpu-bench",
            file=sys.stderr,
        )
        sys.exit(1)
    return torch, torchvision


def build_detector(torch, torchvision):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    # No weights download needed if your env has them cached; otherwise pass
    # `weights=None` and accept random weights (this is a perf demo, not an
    # accuracy demo).
    try:
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    except Exception:
        model = fasterrcnn_resnet50_fpn(weights=None)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def synthesize_batch(torch):
    """Return a [3, 480, 640] float image in [0, 1] on the right device."""
    img = torch.rand(3, 480, 640, dtype=torch.float32)
    if torch.cuda.is_available():
        img = img.cuda(non_blocking=True)
    return img


def preprocess(torch, n: int) -> list:
    """Generate `n` synthetic image tensors and normalize them."""
    out = []
    with pyscope.scope("preprocess", batch_count=n):
        for i in range(n):
            with pyscope.scope("preprocess_one", i=i):
                t = synthesize_batch(torch)
                # mimic a normalize step
                t = (t - 0.485) / 0.229
                out.append(t)
    return out


def postprocess(detections: list) -> list:
    """Collapse detections into a small summary structure."""
    summarized = []
    with pyscope.scope("postprocess", n=len(detections)):
        for i, det in enumerate(detections):
            with pyscope.scope("postprocess_one", i=i):
                boxes = det.get("boxes")
                scores = det.get("scores")
                if boxes is None or scores is None:
                    summarized.append({"n": 0})
                    continue
                k = min(int(boxes.shape[0]), 5)
                summarized.append(
                    {
                        "n": int(boxes.shape[0]),
                        "topk_scores": scores.detach().cpu().tolist()[:k],
                    }
                )
    return summarized


def main() -> None:
    torch, torchvision = _require_deps()
    pyscope.annotate("detector_demo_start", cuda=torch.cuda.is_available())

    with pyscope.scope("model_load"):
        model = build_detector(torch, torchvision)

    inputs = preprocess(torch, n=100)

    detections: list = []
    with pyscope.scope("inference_loop", count=len(inputs)):
        with torch.no_grad():
            for i, img in enumerate(inputs):
                with pyscope.scope("infer_one", i=i):
                    out = model([img])
                    detections.append(out[0])
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    _ = postprocess(detections)

    pyscope.annotate("detector_demo_done", batches=len(inputs))


if __name__ == "__main__":
    main()
