/**
 * FrozenAgent — a non-learning opponent driven by a FROZEN historical snapshot of
 * the policy network. The analogue of tanks' `RandomHistoricalAgent`: same decide
 * pipeline as `UnknownAgent` (board state → mask → batchAct → applyActionToGame)
 * but greedy, with no memory and no gradient contribution — its purpose is to give
 * the learner a stable, time-shifted version of itself to beat instead of the
 * co-adapting live policy.
 *
 * A worker-shared updater loads a RANDOM historical version from IndexedDB;
 * `sync()` re-rolls it between episodes, so opponent strength varies episode to
 * episode. Early in training (before any historical snapshot exists) the random
 * pick degrades to the latest network — plain self-play, which is fine.
 */

import * as tf from "@tensorflow/tfjs";
import { batchActAsync } from "../../../ppo/src/core/train.ts";
import { Model } from "../../../ppo/src/models/def.ts";
import { getRandomHistoricalNetwork, disposeNetwork } from "../../../ppo/src/models/storage.ts";
import { getNetworkExpIteration } from "../../../ppo/src/models/networkMeta.ts";
import { patientAction } from "../../../ppo/src/utils/patientAction.ts";
import { CONFIG } from "../config.ts";
import { createInputTensors } from "../state/InputTensors.ts";
import { prepareInputArrays } from "../state/InputArrays.ts";
import { applyActionToGame } from "./applyActionToGame.ts";
import { computeActionMaskWithPrior } from "./computeActionMask.ts";

// ── Worker-shared frozen network updater ─────────────────────────────────────
let frozenNetwork: tf.LayersModel | undefined;
let frozenNetworkPromise: Promise<tf.LayersModel> | undefined;

async function refreshFrozenNetwork(): Promise<tf.LayersModel> {
  const prev = frozenNetworkPromise;
  frozenNetworkPromise = patientAction(() =>
    getRandomHistoricalNetwork(Model.Policy, CONFIG.savePath),
  );
  const next = await frozenNetworkPromise;
  if (frozenNetwork && frozenNetwork !== next) disposeNetwork(frozenNetwork);
  frozenNetwork = next;
  await prev?.catch(() => undefined);
  return next;
}

export class FrozenAgent {
  constructor(public readonly tankEid: number) {}

  /** Re-roll the frozen opponent: pull a random historical version from storage. */
  static sync(): Promise<unknown> {
    return refreshFrozenNetwork();
  }

  getVersion(): number {
    return frozenNetwork != null ? getNetworkExpIteration(frozenNetwork) : 0;
  }

  /**
   * One decision step. `snapshotUnknownBoard` must already have filled this
   * tank's board row this tick (the driver does it once for all observers).
   */
  async decide(): Promise<void> {
    const network = frozenNetwork;
    if (network == null) return;

    // Capture the observation synchronously, sample asynchronously (only the
    // GPU readback is awaited — see UnknownAgent.decide).
    const state = prepareInputArrays(this.tankEid);
    const mask = computeActionMaskWithPrior(this.tankEid);
    const input = createInputTensors([state]);
    let result;
    try {
      [result] = await batchActAsync(network, input, [mask], { greedy: true });
    } finally {
      input.forEach((t) => t.dispose());
    }

    applyActionToGame(this.tankEid, result.actions);
  }
}
