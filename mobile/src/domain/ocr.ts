import type {OcrFormat, OcrResult} from './types';

const MODERN = /^[A-Z]{4}\d{2}$/;
const LEGACY = /^[A-Z]{2}\d{4}$/;

export function normalizePlateText(rawText: string): OcrResult {
  const candidates = rawText
    .toUpperCase()
    .split(/\r?\n/)
    .map(line => line.replace(/[^A-Z0-9]/g, ''))
    .filter(Boolean);
  const normalizedText =
    candidates.find(candidate => MODERN.test(candidate) || LEGACY.test(candidate)) ??
    candidates.sort((left, right) => Math.abs(left.length - 6) - Math.abs(right.length - 6))[0] ??
    '';
  let format: OcrFormat = 'uncertain';
  if (MODERN.test(normalizedText)) {
    format = 'modern';
  } else if (LEGACY.test(normalizedText)) {
    format = 'legacy';
  }
  return {rawText, normalizedText, format};
}

export function chooseConsensus(readings: OcrResult[]): OcrResult | undefined {
  const nonEmpty = readings.filter(reading => reading.normalizedText.length > 0);
  if (nonEmpty.length === 0) {
    return undefined;
  }
  const votes = new Map<string, number>();
  for (const reading of nonEmpty) {
    votes.set(reading.normalizedText, (votes.get(reading.normalizedText) ?? 0) + 1);
  }
  const winner = [...votes.entries()].sort((left, right) => right[1] - left[1])[0];
  const result = nonEmpty.find(reading => reading.normalizedText === winner[0])!;
  return winner[1] >= 2 ? result : {...result, format: 'uncertain'};
}
