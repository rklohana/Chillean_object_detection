import {intersectionOverUnion} from './geometry';
import {chooseConsensus} from './ocr';
import type {Detection, OcrResult, PlateResult} from './types';

export type DetectionTrack = {
  id: number;
  detection: Detection;
  stableFrames: number;
  missedFrames: number;
  readings: OcrResult[];
};

export function updateTracks(
  tracks: DetectionTrack[],
  detections: Detection[],
  nextId: () => number,
): DetectionTrack[] {
  const unmatched = new Set(detections.map((_, index) => index));
  const updated = tracks.map(track => {
    let bestIndex = -1;
    let bestIou = 0.5;
    for (const index of unmatched) {
      const overlap = intersectionOverUnion(track.detection.box, detections[index].box);
      if (overlap > bestIou) {
        bestIndex = index;
        bestIou = overlap;
      }
    }
    if (bestIndex < 0) {
      return {...track, missedFrames: track.missedFrames + 1};
    }
    unmatched.delete(bestIndex);
    return {
      ...track,
      detection: detections[bestIndex],
      stableFrames: track.stableFrames + 1,
      missedFrames: 0,
    };
  });
  for (const index of unmatched) {
    updated.push({
      id: nextId(),
      detection: detections[index],
      stableFrames: 1,
      missedFrames: 0,
      readings: [],
    });
  }
  return updated.filter(track => track.missedFrames <= 2);
}

export function trackResults(tracks: DetectionTrack[]): PlateResult[] {
  return tracks
    .filter(track => track.missedFrames === 0)
    .map(track => ({...track.detection, ocr: chooseConsensus(track.readings)}));
}
