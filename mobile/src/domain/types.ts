export type BoundingBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type Detection = {
  box: BoundingBox;
  confidence: number;
};

export type OcrFormat = 'modern' | 'legacy' | 'uncertain';

export type OcrResult = {
  rawText: string;
  normalizedText: string;
  format: OcrFormat;
};

export type PlateResult = Detection & {
  ocr?: OcrResult;
};

export type ImageSize = {
  width: number;
  height: number;
};

export type PipelineOutput = {
  source: ImageSize;
  plates: PlateResult[];
  elapsedMs: number;
};

export type HistoryEntry = {
  id: string;
  text: string;
  detectedAt: number;
  confidence: number;
};

export interface PlateDetector {
  initialize(): Promise<void>;
  detectImage(uri: string): Promise<{ detections: Detection[]; source: ImageSize }>;
  detectTensor(data: Float32Array): Promise<Detection[]>;
  release(): Promise<void>;
}

export interface PlateRecognizer {
  recognize(uri: string): Promise<OcrResult>;
}
