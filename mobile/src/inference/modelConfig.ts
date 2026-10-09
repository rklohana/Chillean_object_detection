export const MODEL_CONFIG = {
  assetPath: 'models/plate_detector_fp32.onnx',
  cacheName: 'plate_detector_fp32.onnx',
  inputSize: 640,
  confidenceThreshold: 0.25,
  nmsIouThreshold: 0.45,
} as const;
