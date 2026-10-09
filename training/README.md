# Training and export

The original images stay read-only. `prepare_dataset.py` validates them, removes exact duplicates from the manifest, groups near-duplicates and repeated plate identities, and creates a fixed 70/15/15 split.

For detector training, open `yolo_finetune.ipynb` in a Colab GPU runtime, update its two Google Drive paths, and run every cell. OCR transcriptions are not required to train YOLO; blank values in `transcriptions.csv` can be completed later for OCR evaluation.

The notebook creates:

- `runs/license_plate/yolo11s_plates/weights/best.pt`
- `artifacts/thresholds.json` with validation-selected confidence/NMS thresholds and held-out test metrics
- `artifacts/models/plate_detector_fp32.onnx`
- `artifacts/models/model_metadata.json`

Download or preserve the entire `artifacts` directory. Later, the FP32 ONNX file will be copied into the React Native Android assets folder.

The app refuses to initialize detection when the real exported model is absent. No placeholder detector is shipped as if it were trained.
