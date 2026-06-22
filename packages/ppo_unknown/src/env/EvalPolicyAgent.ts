/**
 * EvalPolicyAgent — drives one tank with a FROZEN policy loaded from static files
 * (a tfjs `model.json` + `.weights.bin`), for playing against a human in the demo
 * build. It is the inference-only twin of `FrozenAgent`: same decide pipeline
 * (board state → mask → batchAct greedy → applyActionToGame), but the network is
 * loaded from a URL pair instead of IndexedDB, so it needs no learner/storage.
 *
 * `attachPolicyOpponent` wires it end to end: tags the tank as a board observer,
 * loads the network, and installs the shared policy driver as the
 * `SystemGroup.Before` plugin (the same seam `createStandInDriverSystem` uses).
 */

import * as tf from "@tensorflow/tfjs";
// Side-effect imports: register every custom layer/optimizer the saved topology
// references, so `tf.loadLayersModel` can deserialize it (MultiHeadAttention,
// RMSNormLayer, MaskLike live in the ppo barrel; HexRingGather is ours).
import "../../../ppo/src/models/Layers/index.ts";
import "../../../ppo/src/models/Optimizer/AdamW.ts";
import "../models/Layers/HexRingGatherLayer.ts";

import { batchActAsync } from "../../../ppo/src/core/train.ts";
import { GameDI } from "../../../unknown/src/Game/DI/GameDI.ts";
import { PluginDI } from "../../../unknown/src/Game/DI/PluginDI.ts";
import { SystemGroup } from "../../../unknown/src/Game/ECS/Plugins/systems.ts";
import { ensureUnknownInputBoard } from "../state/board.ts";
import { createInputTensors } from "../state/InputTensors.ts";
import { prepareInputArrays } from "../state/InputArrays.ts";
import { applyActionToGame } from "./applyActionToGame.ts";
import { computeActionMaskWithPrior } from "./computeActionMask.ts";
import { createPolicyDriverSystem, type TankDriver } from "./createPolicyDriverSystem.ts";

/**
 * Load a tfjs LayersModel from a `model.json` URL + its `.weights.bin` URL.
 *
 * Bundlers (vite) hash and relocate both files independently, which breaks the
 * relative `weightsManifest` path the json carries — so we can't let tfjs resolve
 * the sidecar itself. Instead we fetch both URLs explicitly and hand tfjs a
 * one-shot IOHandler with the topology, weight specs, and weight bytes in place.
 */
export async function loadPolicyNetwork(
  modelJsonUrl: string,
  weightsUrl: string,
): Promise<tf.LayersModel> {
  await tf.ready(); // ensure a backend (WebGL/CPU) is registered before load
  const [modelJson, weightData] = await Promise.all([
    fetch(modelJsonUrl).then((r) => r.json()),
    fetch(weightsUrl).then((r) => r.arrayBuffer()),
  ]);
  const weightSpecs = (modelJson.weightsManifest as Array<{ weights: tf.io.WeightsManifestEntry[] }>)
    .flatMap((group) => group.weights);

  return tf.loadLayersModel({
    load: async () => ({
      modelTopology: modelJson.modelTopology,
      weightSpecs,
      weightData,
    }),
  });
}

export class EvalPolicyAgent {
  constructor(
    public readonly tankEid: number,
    private readonly network: tf.LayersModel,
  ) {}

  /**
   * One decision step. `snapshotUnknownBoard` must already have filled this
   * tank's board row this tick (the driver does it once for all observers).
   * Greedy (argmax) — a deterministic opponent, no exploration.
   */
  async decide(): Promise<void> {
    const state = prepareInputArrays(this.tankEid);
    const mask = computeActionMaskWithPrior(this.tankEid);
    const input = createInputTensors([state]);
    let result;
    try {
      [result] = await batchActAsync(this.network, input, [mask], { greedy: true });
    } finally {
      input.forEach((t) => t.dispose());
    }
    applyActionToGame(this.tankEid, result.actions);
  }
}

/**
 * Install ONE shared policy driver and load the network ONCE, then hand back an
 * `attach(eid)` to make any tank an agent-driven opponent — call it for the
 * starting enemy and for each one spawned later (the driver reads its agent map
 * live, so tanks added after install are picked up). Fire-and-forget: the
 * rendered demo never drains.
 */
export function createPolicyOpponentController(opts: {
  modelJsonUrl: string;
  weightsUrl: string;
  world?: (typeof GameDI)["world"];
}) {
  const world = opts.world ?? GameDI.world;
  const board = ensureUnknownInputBoard(world);

  const agents = new Map<number, TankDriver>();
  const driver = createPolicyDriverSystem(agents);
  PluginDI.addSystem(SystemGroup.Before, driver.system);

  const networkPromise = loadPolicyNetwork(opts.modelJsonUrl, opts.weightsUrl);

  return {
    /** Register `eid` as a board observer + policy-driven agent (network loads once). */
    async attach(eid: number): Promise<void> {
      board.addComponent(world, eid);
      const network = await networkPromise;
      agents.set(eid, new EvalPolicyAgent(eid, network));
    },
  };
}
