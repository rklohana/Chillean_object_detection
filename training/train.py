#!/usr/bin/env python3
"""Fine-tune YOLO11s with the fixed project training recipe."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path("runs/license_plate"))
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device != "cpu":
        raise RuntimeError("CUDA is not available. Use a Colab GPU runtime or pass --device cpu explicitly.")

    args.project.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (args.project / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    model = YOLO("yolo11s.pt")
    model.train(
        data=str(args.data.resolve()),
        imgsz=640,
        epochs=150,
        patience=25,
        batch=-1,
        seed=42,
        deterministic=True,
        device=args.device,
        project=str(args.project),
        name="yolo11s_plates",
        exist_ok=True,
        pretrained=True,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.35,
        degrees=5.0,
        translate=0.08,
        scale=0.35,
        perspective=0.0005,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.5,
        close_mosaic=10,
        plots=True,
    )


if __name__ == "__main__":
    main()
