import { round, trunc } from "./math";

export const random = Math.random;
export const randomInt = (random() * Number.MAX_SAFE_INTEGER) | 0;

export function randomRangeFloat(min: number, max: number): number {
  return random() * (max - min) + min;
}

export function randomRangeInt(min: number, max: number): number {
  return round(randomRangeFloat(min, max));
}

export function randomSign(): number {
  return random() > 0.5 ? 1 : -1;
}

// Mulberry32 — deterministic stream for gameplay randomness (training reproducibility).
let seededState = 0x9e3779b9;

export function setSeededRandomSeed(seed: number): void {
  seededState = seed >>> 0;
}

export function seededRandom(): number {
  seededState = (seededState + 0x6d2b79f5) >>> 0;
  let t = seededState;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

export function seededRandomRangeFloat(min: number, max: number): number {
  return seededRandom() * (max - min) + min;
}

export function randomId(): string {
  return String(trunc(Date.now() * Math.random()));
}

export function randomShortId(): string {
  return randomId().slice(-6);
}
