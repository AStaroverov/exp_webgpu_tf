/**
 * Policy / value network factories for ppo_unknown. Identical wiring to tanks'
 * `createTankNetworks` — only the trunk (Networks/v2) and head dims differ. The
 * generic learner consumes these unchanged via `createNetwork: createPolicyNetwork`.
 */

import * as tf from "@tensorflow/tfjs";
import { CONFIG } from "../config.ts";
import { createDenseLayer } from "../../../ppo/src/models/ApplyLayers.ts";
import { Model } from "../../../ppo/src/models/def.ts";
import { AdamW } from "../../../ppo/src/models/Optimizer/AdamW.ts";
import { createNetwork } from "./Networks/v3.ts";

export { ACTION_HEAD_DIMS } from "./dims.ts";
export { shouldNoiseLayer } from "../../../ppo/src/models/noiseGate.ts";

export function createPolicyNetwork(): tf.LayersModel {
  // The network builds its own final logits (per-cell action head in v1).
  const { inputs, heads } = createNetwork(Model.Policy);

  const model = tf.model({
    name: Model.Policy,
    inputs: Object.values(inputs),
    outputs: heads,
  });
  model.optimizer = new AdamW(CONFIG.lrConfig.initial);
  model.loss = "meanSquaredError"; // placeholder; real loss is applied in train.ts

  return model;
}

export function createValueNetwork(): tf.LayersModel {
  const { inputs, heads } = createNetwork(Model.Value);
  const valueOutput = createDenseLayer({
    name: Model.Value + "_output",
    units: 1,
    useBias: true,
    activation: "linear",
    biasInitializer: "zeros",
    kernelInitializer: "glorotUniform",
  }).apply(heads[0]) as tf.SymbolicTensor;

  const model = tf.model({
    name: Model.Value,
    inputs: Object.values(inputs),
    outputs: valueOutput,
  });
  model.optimizer = new AdamW(CONFIG.lrConfig.initial);
  model.loss = "meanSquaredError";

  return model;
}
