#!/usr/bin/env python3
"""Export and validate the fixed-shape YOLO11s ONNX deployment model."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_extensions
from onnx import TensorProto, helper, numpy_helper
from onnxruntime_extensions.tools import add_pre_post_processing_to_model
from PIL import Image
from ultralytics import YOLO


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_center_boxes_to_corners(model_path: Path) -> None:
    """Change the Extensions helper output from cx/cy/w/h to x1/y1/x2/y2."""
    graph = onnx.load(model_path)
    source_output = graph.graph.output[0].name
    starts_xy = numpy_helper.from_array(np.array([0], dtype=np.int64), "starts_xy")
    ends_xy = numpy_helper.from_array(np.array([2], dtype=np.int64), "ends_xy")
    starts_wh = numpy_helper.from_array(np.array([2], dtype=np.int64), "starts_wh")
    ends_wh = numpy_helper.from_array(np.array([4], dtype=np.int64), "ends_wh")
    starts_tail = numpy_helper.from_array(np.array([4], dtype=np.int64), "starts_tail")
    ends_tail = numpy_helper.from_array(np.array([6], dtype=np.int64), "ends_tail")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "box_axes")
    two = numpy_helper.from_array(np.array([2.0], dtype=np.float32), "two")
    graph.graph.initializer.extend([starts_xy, ends_xy, starts_wh, ends_wh, starts_tail, ends_tail, axes, two])
    graph.graph.node.extend(
        [
            helper.make_node("Slice", [source_output, "starts_xy", "ends_xy", "box_axes"], ["box_center"]),
            helper.make_node("Slice", [source_output, "starts_wh", "ends_wh", "box_axes"], ["box_size"]),
            helper.make_node("Slice", [source_output, "starts_tail", "ends_tail", "box_axes"], ["score_class"]),
            helper.make_node("Div", ["box_size", "two"], ["half_size"]),
            helper.make_node("Sub", ["box_center", "half_size"], ["top_left"]),
            helper.make_node("Add", ["box_center", "half_size"], ["bottom_right"]),
            helper.make_node("Concat", ["top_left", "bottom_right", "score_class"], ["detections"], axis=1),
        ]
    )
    graph.graph.ClearField("output")
    graph.graph.output.append(helper.make_tensor_value_info("detections", TensorProto.FLOAT, [None, 6]))
    onnx.checker.check_model(graph)
    onnx.save(graph, model_path)


def build_byte_input_model(tensor_model: Path, output: Path) -> tuple[list[int], str]:
    add_pre_post_processing_to_model.yolo_detection(
        tensor_model,
        output,
        onnx_opset=18,
        num_classes=1,
        input_shape=[640, 640],
        output_as_image=False,
    )
    convert_center_boxes_to_corners(output)
    options = ort.SessionOptions()
    options.register_custom_ops_library(onnxruntime_extensions.get_library_path())
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    blank = Image.new("RGB", (960, 720), color=(114, 114, 114))
    encoded = io.BytesIO()
    blank.save(encoded, format="JPEG")
    data = np.frombuffer(encoded.getvalue(), dtype=np.uint8)
    result = session.run(None, {session.get_inputs()[0].name: data})[0]
    if result.ndim != 2 or result.shape[1] != 6:
        raise RuntimeError(f"Unexpected byte-input output shape: {result.shape}")
    return list(result.shape), session.get_outputs()[0].name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--with-byte-model", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=640,
            batch=1,
            dynamic=False,
            simplify=True,
            opset=18,
            nms=False,
        )
    )
    target = args.output_dir / "plate_detector_fp32.onnx"
    shutil.copy2(exported, target)

    graph = onnx.load(target)
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    sample = np.zeros((1, 3, 640, 640), dtype=np.float32)
    output = session.run(None, {input_meta.name: sample})[0]
    confidence_threshold = 0.25
    nms_iou_threshold = 0.45
    if args.thresholds:
        threshold_report = json.loads(args.thresholds.read_text(encoding="utf-8"))
        confidence_threshold = float(threshold_report["selected"]["confidence"])
        nms_iou_threshold = float(threshold_report["selected"]["nms_iou"])

    metadata = {
        "model": "yolo11s-license-plate",
        "version": 1,
        "sha256": digest(target),
        "input": {
            "name": input_meta.name,
            "shape": input_meta.shape,
            "dtype": "float32",
            "layout": "NCHW",
            "color": "RGB",
            "normalization": "0..1",
            "letterbox": 640,
        },
        "output": {"name": output_meta.name, "shape": list(output.shape), "layout": "raw_yolo11"},
        "classes": ["license_plate"],
        "confidence_threshold": confidence_threshold,
        "nms_iou_threshold": nms_iou_threshold,
        "license": "AGPL-3.0-or-later or a separate Ultralytics Enterprise licence",
    }
    if args.with_byte_model:
        byte_model = args.output_dir / "plate_detector_bytes.onnx"
        byte_shape, byte_output_name = build_byte_input_model(target, byte_model)
        metadata["byte_input_model"] = {
            "filename": byte_model.name,
            "sha256": digest(byte_model),
            "input": "encoded JPEG or PNG uint8 bytes",
            "output": {
                "name": byte_output_name,
                "shape": byte_shape,
                "columns": ["x1", "y1", "x2", "y2", "score", "class_id"],
            },
        }
    (args.output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
