/**
 * v4 — v3's perceiver, widened to the ring-target action space. The fire slice
 * is no longer 6 directions but the 36 precise hexes of rings 1..3 around the
 * tank (consts.ts / FIRE_TARGET_OFFSETS), so the latents become the 37 action
 * cells: self + the 36 ring cells (whose first 6, ring 1, are exactly the move
 * cells — POINTY_DIRECTIONS order).
 *
 *   board [ROWS, COLS, CH] → [ROWS*COLS, CH] tokens → proj dModel → +posenc (K/V)
 *   latents [37, dModel] = gather(self + rings 1..3; fixed indices — the board
 *     is egocentric, self is always the window center) + learned slot embedding
 *     → N × perceiver blocks (cross-attn "latents read the board" (content-masked)
 *       → self-attn "latents talk to each other"), pre-LN, input/output norm
 *     → Policy: hold from latent 0, move d from latent 1+d (the ring-1 cells),
 *       fire t from latent 1+t (direction-shared scorers) → logits [43]
 *     → Value: flatten latents → feature [37*dim]
 *
 * Attention cost per block is 37×121 instead of v3's 7×121 — still latent-side
 * bounded, and each latent is BY CONSTRUCTION the decision token of its action:
 * "shoot at this cell" reads the very token that perceived that cell.
 *
 * Contract (consumed by createUnknownNetworks):
 *   createNetwork(model) → { inputs, heads }
 *     - inputs: Record<string, tf.SymbolicTensor> — Object.values order matches
 *       state/InputTensors.ts.
 *     - heads: tf.SymbolicTensor[] — Policy: FINAL logits, one per
 *       ACTION_HEAD_DIMS entry; Value: a single feature (heads[0]).
 */

import * as tf from "@tensorflow/tfjs";
import { Model } from "../../../../ppo/src/models/def.ts";
import {
  createDenseLayer,
  createNormalizationLayer,
  tokenProj,
} from "../../../../ppo/src/models/ApplyLayers.ts";
import { applyPerceiverLayer } from "../../../../ppo/src/models/Layers/PerceiverLayer.ts";
import { SliceLayer } from "../../../../ppo/src/models/Layers/SliceLayer.ts";
import { HexRingGatherLayer } from "../Layers/HexRingGatherLayer.ts";
import { ACTION_CELL_INDEXES } from "../../state/hexNeighbors.ts";
import { BOARD_CELLS, BOARD_CHANNELS, FIRE_TARGET_COUNT, MOVE_DIR_COUNT } from "../dims.ts";
import { createInputs } from "../Inputs.ts";

type NetworkConfig = { dim: number; heads: number; depth: number };

const policyConfig: NetworkConfig = { dim: 128, heads: 4, depth: 4 };
const valueConfig: NetworkConfig = { dim: 64, heads: 2, depth: 2 };

export function createNetwork(
  modelName: Model,
  config: NetworkConfig = modelName === Model.Policy ? policyConfig : valueConfig,
) {
  const inputs = createInputs(modelName);

  // [B, ROWS, COLS, CH] → [B, ROWS*COLS, CH]: one token per board cell.
  const cellTokens = tf.layers
    .reshape({
      name: modelName + "_cellTokens",
      targetShape: [BOARD_CELLS, BOARD_CHANNELS],
    })
    .apply(inputs.boardInput) as tf.SymbolicTensor;

  const contentMask = inputs.contentMaskInput; // [B, BOARD_CELLS]

  const projected = tokenProj(cellTokens, config.dim, modelName + "_cell");

  const latentSeed = new HexRingGatherLayer({
    name: modelName + "_actionCellTokens",
    indexes: ACTION_CELL_INDEXES,
  }).apply(projected) as tf.SymbolicTensor; // [B, 37, dim]

  const latentNorm = createNormalizationLayer({
    name: modelName + "_latentInputNorm",
  }).apply(latentSeed) as tf.SymbolicTensor;

  const perceived = applyPerceiverLayer({
    name: modelName + "_perceiver",
    depth: config.depth,
    heads: config.heads,
    qTok: latentNorm,
    kvTok: projected,
    kvMask: contentMask,
    preNorm: true,
  });

  const encoded = createNormalizationLayer({
    name: modelName + "_outputNorm",
  }).apply(perceived) as tf.SymbolicTensor; // [B, 37, dim]

  if (modelName === Model.Policy) {
    const holdToken = tf.layers
      .reshape({
        name: modelName + "_holdToken",
        targetShape: [config.dim],
      })
      .apply(
        new SliceLayer({
          name: modelName + "_holdSlice",
          beginSlice: [0, 0, 0],
          sliceSize: [-1, 1, -1],
        }).apply(encoded) as tf.SymbolicTensor,
      ) as tf.SymbolicTensor; // [B, dim]

    const moveTokens = new SliceLayer({
      name: modelName + "_moveSlice",
      beginSlice: [0, 1, 0],
      sliceSize: [-1, MOVE_DIR_COUNT, -1],
    }).apply(encoded) as tf.SymbolicTensor; // [B, 6, dim] — ring-1 latents

    const fireTokens = new SliceLayer({
      name: modelName + "_fireSlice",
      beginSlice: [0, 1, 0],
      sliceSize: [-1, FIRE_TARGET_COUNT, -1],
    }).apply(encoded) as tf.SymbolicTensor; // [B, 36, dim] — all ring latents

    // Near-zero init on the scorers: logits start ≈ uniform (no random
    // constant prior), state signal shows as crossings immediately.
    const logitInit = () => tf.initializers.orthogonal({ gain: 0.2 });
    const holdLogit = createDenseLayer({
      name: modelName + "_holdLogit",
      units: 1,
      useBias: true,
      activation: "linear",
      biasInitializer: "zeros",
      kernelInitializer: logitInit(),
    }).apply(holdToken) as tf.SymbolicTensor; // [B, 1]

    const dirLogits = (kind: string, tokens: tf.SymbolicTensor, dims: number) =>
      tf.layers
        .reshape({
          name: modelName + `_${kind}Logits`,
          targetShape: [dims],
        })
        .apply(
          createDenseLayer({
            name: modelName + `_${kind}Scorer`,
            units: 1,
            useBias: true,
            activation: "linear",
            biasInitializer: "zeros",
            kernelInitializer: logitInit(),
          }).apply(tokens) as tf.SymbolicTensor,
        ) as tf.SymbolicTensor; // [B, dims]

    const logits = tf.layers
      .concatenate({
        name: modelName + "_logits",
      })
      .apply([
        holdLogit,
        dirLogits("move", moveTokens, MOVE_DIR_COUNT),
        dirLogits("fire", fireTokens, FIRE_TARGET_COUNT),
      ]) as tf.SymbolicTensor; // [B, ACTION_DIM_TOTAL], order = consts.ts layout

    return { inputs, heads: [logits] };
  }

  // Value: flatten the 37 latents into a single feature; createValueNetwork
  // appends the scalar output.
  const feature = tf.layers
    .flatten({
      name: modelName + "_flatReadout",
    })
    .apply(encoded) as tf.SymbolicTensor; // [B, 37 * dim]

  return { inputs, heads: [feature] };
}
