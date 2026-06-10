/**
 * v0 — deliberately dumb baseline: flatten the board → MLP → heads. No attention,
 * no posenc, no pooling. Exists to validate the TRAINING PIPELINE: an init-time
 * probe showed this trunk passes ~90% input-dependent signal to the logits (vs
 * ~17% for the v1 transformer), so if logits stay state-independent when training
 * on v0 too, the problem is in training, not the network.
 *
 * Same contract as v1 (consumed by createUnknownNetworks):
 *   createNetwork(model) → { inputs, heads }
 */

import * as tf from "@tensorflow/tfjs";
import { Model } from "../../../../ppo/src/models/def.ts";
import { createDenseLayer } from "../../../../ppo/src/models/ApplyLayers.ts";
import { ACTION_HEAD_DIMS } from "../dims.ts";
import { createInputs } from "../Inputs.ts";

export function createNetwork(modelName: Model) {
  const inputs = createInputs(modelName);

  const flatBoard = tf.layers
    .flatten({
      name: modelName + "_flatBoard",
    })
    .apply(inputs.boardInput) as tf.SymbolicTensor; // [B, ROWS*COLS*CH]

  // Concat the content mask too — keeps both model inputs connected (tf.model
  // rejects dangling inputs) and it is informative on its own.
  let feature = tf.layers
    .concatenate({
      name: modelName + "_flatInput",
    })
    .apply([flatBoard, inputs.contentMaskInput]) as tf.SymbolicTensor;

  for (const [i, units] of [256, 128].entries()) {
    feature = createDenseLayer({
      name: modelName + "_mlp" + i,
      units,
      useBias: true,
      activation: "relu",
    }).apply(feature) as tf.SymbolicTensor;
  }

  if (modelName === Model.Policy) {
    const heads = ACTION_HEAD_DIMS.map(
      (_, i) =>
        createDenseLayer({
          name: modelName + "_headBranch" + i,
          units: 64,
          useBias: true,
          activation: "relu",
        }).apply(feature) as tf.SymbolicTensor,
    );
    return { inputs, heads };
  }

  // Value: single feature; createValueNetwork appends the scalar output.
  return { inputs, heads: [feature] };
}
