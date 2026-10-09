#!/usr/bin/env python3
"""Select detector confidence/NMS thresholds on validation, then score test once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


CONFIDENCE_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
IOU_GRID = (0.40, 0.45, 0.50, 0.55, 0.60)


def metric_summary(metrics: object) -> dict[str, float]:
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/thresholds.json"))
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(str(args.weights.resolve()))
    trials: list[dict[str, float]] = []
    for confidence in CONFIDENCE_GRID:
        for iou in IOU_GRID:
            metrics = model.val(
                data=str(args.data.resolve()),
                split="val",
                imgsz=640,
                conf=confidence,
                iou=iou,
                device=args.device,
                plots=False,
                verbose=False,
            )
            trials.append({"confidence": confidence, "nms_iou": iou, **metric_summary(metrics)})

    selected = max(trials, key=lambda row: (row["f1"], row["map50"], row["recall"]))
    test_operating_point = model.val(
        data=str(args.data.resolve()),
        split="test",
        imgsz=640,
        conf=selected["confidence"],
        iou=selected["nms_iou"],
        device=args.device,
        plots=True,
        verbose=False,
    )
    test_ap = model.val(
        data=str(args.data.resolve()),
        split="test",
        imgsz=640,
        conf=0.001,
        iou=selected["nms_iou"],
        device=args.device,
        plots=False,
        verbose=False,
    )
    test = metric_summary(test_operating_point)
    test["map50"] = float(test_ap.box.map50)
    test["map50_95"] = float(test_ap.box.map)
    report = {
        "selection_rule": "maximum validation F1; ties use mAP50 then recall",
        "selected": selected,
        "test": test,
        "test_ap_confidence": 0.001,
        "validation_trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "test": test}, indent=2))


if __name__ == "__main__":
    main()
