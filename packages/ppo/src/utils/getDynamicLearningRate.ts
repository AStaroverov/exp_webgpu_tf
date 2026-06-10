import { clamp } from "lodash-es";
import type { PpoConfig } from "../config.ts";

export function getDynamicLearningRate(
  kl: number,
  lr: number,
  lrConfig: PpoConfig["lrConfig"],
): number {
  const {
    kl: { high: klHigh, low: klLow },
    multHigh: lrMultHigh,
    multLow: lrMultLow,
    min: minLR,
    max: maxLR,
  } = lrConfig;

  if (kl > klHigh) lr *= lrMultHigh;
  else if (kl < klLow) lr *= lrMultLow;

  return clamp(lr, minLR, maxLR);
}
