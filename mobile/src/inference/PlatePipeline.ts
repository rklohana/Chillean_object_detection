import ImageEditor from '@react-native-community/image-editor';
import {FileSystem} from 'react-native-file-access';
import {padBox} from '../domain/geometry';
import {chooseConsensus} from '../domain/ocr';
import {trackResults, updateTracks, type DetectionTrack} from '../domain/tracker';
import type {
  BoundingBox,
  ImageSize,
  PipelineOutput,
  PlateDetector,
  PlateRecognizer,
  PlateResult,
} from '../domain/types';

export class PlatePipeline {
  private busy = false;
  private tracks: DetectionTrack[] = [];
  private id = 0;

  constructor(
    private readonly detector: PlateDetector,
    private readonly recognizer: PlateRecognizer,
  ) {}

  initialize() {
    return this.detector.initialize();
  }

  async processGallery(uri: string): Promise<PipelineOutput | null> {
    return this.exclusive(async () => {
      const startedAt = Date.now();
      const {detections, source} = await this.detector.detectImage(uri);
      const plates: PlateResult[] = [];
      for (const detection of detections) {
        plates.push({...detection, ocr: await this.recognizeBox(uri, detection.box, source)});
      }
      return {source, plates, elapsedMs: Date.now() - startedAt};
    });
  }

  async processLive(uri: string): Promise<PipelineOutput | null> {
    return this.exclusive(async () => {
      const startedAt = Date.now();
      const {detections, source} = await this.detector.detectImage(uri);
      this.tracks = updateTracks(this.tracks, detections, () => ++this.id);
      for (const track of this.tracks) {
        if (track.missedFrames !== 0 || track.stableFrames < 3 || track.readings.length >= 3) {
          continue;
        }
        const consensus = chooseConsensus(track.readings);
        if (consensus && consensus.format !== 'uncertain') {
          continue;
        }
        track.readings.push(await this.recognizeBox(uri, track.detection.box, source));
      }
      return {source, plates: trackResults(this.tracks), elapsedMs: Date.now() - startedAt};
    });
  }

  resetTracking() {
    this.tracks = [];
  }

  release() {
    return this.detector.release();
  }

  private async recognizeBox(uri: string, box: BoundingBox, source: ImageSize) {
    const crop = padBox(box, source);
    const scale = crop.height < 96 ? 96 / Math.max(crop.height, 1) : 1;
    const result = await ImageEditor.cropImage(uri, {
      offset: {x: Math.floor(crop.x), y: Math.floor(crop.y)},
      size: {width: Math.max(1, Math.floor(crop.width)), height: Math.max(1, Math.floor(crop.height))},
      displaySize:
        scale > 1
          ? {width: Math.min(1280, Math.round(crop.width * scale)), height: Math.round(crop.height * scale)}
          : undefined,
      format: 'jpeg',
      quality: 0.95,
    });
    try {
      return await this.recognizer.recognize(result.uri);
    } finally {
      if (await FileSystem.exists(result.path)) {
        await FileSystem.unlink(result.path);
      }
    }
  }

  private async exclusive<T>(work: () => Promise<T>): Promise<T | null> {
    if (this.busy) {
      return null;
    }
    this.busy = true;
    try {
      return await work();
    } finally {
      this.busy = false;
    }
  }
}
