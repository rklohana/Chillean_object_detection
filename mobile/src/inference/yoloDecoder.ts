import {intersectionOverUnion} from '../domain/geometry';
import type {Detection, ImageSize} from '../domain/types';

export type LetterboxTransform = {
  source: ImageSize;
  scale: number;
  padX: number;
  padY: number;
  inputSize: number;
};

type Candidate = Detection & {score: number};

function candidateToSource(
  cx: number,
  cy: number,
  width: number,
  height: number,
  score: number,
  transform: LetterboxTransform,
): Candidate {
  const x1 = (cx - width / 2 - transform.padX) / transform.scale;
  const y1 = (cy - height / 2 - transform.padY) / transform.scale;
  const x2 = (cx + width / 2 - transform.padX) / transform.scale;
  const y2 = (cy + height / 2 - transform.padY) / transform.scale;
  const clampedX1 = Math.max(0, Math.min(transform.source.width, x1));
  const clampedY1 = Math.max(0, Math.min(transform.source.height, y1));
  const clampedX2 = Math.max(clampedX1, Math.min(transform.source.width, x2));
  const clampedY2 = Math.max(clampedY1, Math.min(transform.source.height, y2));
  return {
    box: {
      x: clampedX1,
      y: clampedY1,
      width: clampedX2 - clampedX1,
      height: clampedY2 - clampedY1,
    },
    confidence: score,
    score,
  };
}

export function nonMaximumSuppression(
  detections: Detection[],
  iouThreshold: number,
  limit = 20,
): Detection[] {
  const sorted = [...detections].sort((left, right) => right.confidence - left.confidence);
  const kept: Detection[] = [];
  while (sorted.length > 0 && kept.length < limit) {
    const best = sorted.shift()!;
    kept.push(best);
    for (let index = sorted.length - 1; index >= 0; index -= 1) {
      if (intersectionOverUnion(best.box, sorted[index].box) > iouThreshold) {
        sorted.splice(index, 1);
      }
    }
  }
  return kept;
}

export function decodeYolo11(
  data: Float32Array,
  dims: readonly number[],
  transform: LetterboxTransform,
  confidenceThreshold: number,
  iouThreshold: number,
): Detection[] {
  if (dims.length !== 3 || dims[0] !== 1) {
    throw new Error(`Unexpected YOLO output shape: ${dims.join('x')}`);
  }
  const channelFirst = dims[1] <= 16;
  const channels = channelFirst ? dims[1] : dims[2];
  const count = channelFirst ? dims[2] : dims[1];
  if (channels < 5) {
    throw new Error(`YOLO output has ${channels} channels; expected at least five`);
  }
  const at = (candidate: number, channel: number) =>
    channelFirst ? data[channel * count + candidate] : data[candidate * channels + channel];
  const candidates: Detection[] = [];
  for (let index = 0; index < count; index += 1) {
    const score = at(index, 4);
    if (score < confidenceThreshold) {
      continue;
    }
    const candidate = candidateToSource(
      at(index, 0),
      at(index, 1),
      at(index, 2),
      at(index, 3),
      score,
      transform,
    );
    if (candidate.box.width > 1 && candidate.box.height > 1) {
      candidates.push(candidate);
    }
  }
  return nonMaximumSuppression(candidates, iouThreshold);
}
