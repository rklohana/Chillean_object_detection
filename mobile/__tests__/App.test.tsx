import {clampBox, intersectionOverUnion, mapBoxToView, padBox} from '../src/domain/geometry';
import {addHistoryResults} from '../src/domain/history';
import {chooseConsensus, normalizePlateText} from '../src/domain/ocr';
import {trackResults, updateTracks, type DetectionTrack} from '../src/domain/tracker';
import {decodeYolo11} from '../src/inference/yoloDecoder';

describe('geometry', () => {
  test('clamps and pads a box without leaving the image', () => {
    const image = {width: 100, height: 80};
    expect(clampBox({x: -5, y: 10, width: 120, height: 100}, image)).toEqual({
      x: 0, y: 10, width: 100, height: 70,
    });
    expect(padBox({x: 1, y: 1, width: 20, height: 10}, image, 0.1)).toEqual({
      x: 0, y: 0, width: 23, height: 12,
    });
  });

  test('calculates IoU and contain mapping', () => {
    expect(intersectionOverUnion(
      {x: 0, y: 0, width: 10, height: 10},
      {x: 5, y: 0, width: 10, height: 10},
    )).toBeCloseTo(1 / 3);
    expect(mapBoxToView(
      {x: 0, y: 0, width: 100, height: 50},
      {width: 100, height: 50},
      {width: 200, height: 200},
      'contain',
    )).toMatchObject({x: 0, y: 50, width: 200, height: 100});
  });
});

describe('OCR normalization and history', () => {
  test.each([
    ['BB-XD 89', 'BBXD89', 'modern'],
    ['XP 2304', 'XP2304', 'legacy'],
    ['maybe: 123', 'MAYBE123', 'uncertain'],
  ] as const)('normalizes %s', (raw, text, format) => {
    expect(normalizePlateText(raw)).toMatchObject({normalizedText: text, format});
  });

  test('requires two matching OCR readings for confident consensus', () => {
    const first = normalizePlateText('BBXD89');
    expect(chooseConsensus([first])).toMatchObject({format: 'uncertain'});
    expect(chooseConsensus([first, first])).toMatchObject({format: 'modern'});
  });

  test('deduplicates recent history entries', () => {
    const plate = {box: {x: 0, y: 0, width: 1, height: 1}, confidence: 0.9, ocr: normalizePlateText('BBXD89')};
    const once = addHistoryResults([], [plate], 1_000);
    expect(addHistoryResults(once, [plate], 2_000)).toHaveLength(1);
    expect(addHistoryResults(once, [plate], 7_000)).toHaveLength(2);
  });
});

describe('YOLO decoding', () => {
  test('decodes channel-first raw YOLO11 output and reverses letterbox', () => {
    const count = 2;
    const values = new Float32Array(5 * count);
    const set = (candidate: number, channel: number, value: number) => {
      values[channel * count + candidate] = value;
    };
    set(0, 0, 320); set(0, 1, 320); set(0, 2, 100); set(0, 3, 40); set(0, 4, 0.9);
    set(1, 0, 20); set(1, 1, 20); set(1, 2, 10); set(1, 3, 10); set(1, 4, 0.1);
    const detections = decodeYolo11(
      values,
      [1, 5, count],
      {source: {width: 640, height: 480}, scale: 1, padX: 0, padY: 80, inputSize: 640},
      0.25,
      0.45,
    );
    expect(detections).toHaveLength(1);
    expect(detections[0].box).toEqual({x: 270, y: 220, width: 100, height: 40});
  });
});

describe('live tracking', () => {
  test('increments stability and exposes consensus OCR', () => {
    let id = 0;
    const detection = {box: {x: 10, y: 10, width: 50, height: 20}, confidence: 0.9};
    let tracks: DetectionTrack[] = updateTracks([], [detection], () => ++id);
    tracks = updateTracks(tracks, [detection], () => ++id);
    tracks[0].readings.push(normalizePlateText('BBXD89'), normalizePlateText('BBXD89'));
    expect(tracks[0].stableFrames).toBe(2);
    expect(trackResults(tracks)[0].ocr).toMatchObject({normalizedText: 'BBXD89', format: 'modern'});
  });
});
