import type {HistoryEntry, PlateResult} from './types';

const MAX_HISTORY = 100;
const DEDUPE_MS = 5_000;

export function addHistoryResults(
  current: HistoryEntry[],
  plates: PlateResult[],
  now = Date.now(),
): HistoryEntry[] {
  let next = [...current];
  for (const plate of plates) {
    const text = plate.ocr?.normalizedText;
    if (!text) {
      continue;
    }
    const isDuplicate = next.some(
      entry => entry.text === text && now - entry.detectedAt < DEDUPE_MS,
    );
    if (!isDuplicate) {
      next.unshift({
        id: `${now}-${text}`,
        text,
        detectedAt: now,
        confidence: plate.confidence,
      });
    }
  }
  return next.slice(0, MAX_HISTORY);
}
