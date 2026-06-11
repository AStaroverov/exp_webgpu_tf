/**
 * v2 — perceiver over the action cells. v1 ran full 64×64 self-attention and
 * then read out only 7 tokens (self + 6 hex neighbors) — ~89% of the encoder
 * work fed tokens nobody reads. v2 inverts this: those 7 cells ARE the latents.
 *
 *   board [ROWS, COLS, CH] → [ROWS*COLS, CH] tokens → proj dModel → +posenc (K/V)
 *   latents [7, dModel] = gather(self + 6 neighbors) + learned slot embedding
 *     → N × perceiver blocks (cross-attn "latents read the board" (content-masked)
 *       → self-attn "latents talk to each other"), pre-LN, input/output norm
 *     → Policy: hold from latent 0, move/fire d from latent 1+d
 *       (direction-shared scorers, same head as v1) → logits [13]
 *     → Value: flatten latents → feature [7*dim]
 *
 * Attention cost per block is 7×64 instead of 64×64, FFN runs over 7 tokens
 * instead of 64, and each latent is BY CONSTRUCTION the decision token of its
 * action — the asymmetric-mask trick from v1 (empty cells must still query)
 * becomes structural: latents always query, the board is read content-masked.
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
import { Grid2DPositionalEncodingLayer } from "../../../../ppo/src/models/Layers/Grid2DPositionalEncodingLayer.ts";
import { SliceLayer } from "../../../../ppo/src/models/Layers/SliceLayer.ts";
import { SlotEmbeddingLayer } from "../../../../ppo/src/models/Layers/SlotEmbeddingLayer.ts";
import { HexNeighborGatherLayer } from "../Layers/HexNeighborGatherLayer.ts";
import {
  BOARD_CELLS,
  BOARD_CHANNELS,
  BOARD_COLS,
  BOARD_ROWS,
  BoardChannel,
  MOVE_DIR_COUNT,
} from "../dims.ts";

// Historical 6-direction fire slice (pre ring-target action space, see v4).
const FIRE_DIR_COUNT = MOVE_DIR_COUNT;
import { createInputs } from "../Inputs.ts";

type NetworkConfig = { dim: number; heads: number; depth: number };

const policyConfig: NetworkConfig = { dim: 64, heads: 4, depth: 4 };
const valueConfig: NetworkConfig = { dim: 32, heads: 2, depth: 2 };

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

  // Only content cells can be READ by the cross-attention (~55 of 64 are
  // empty; their pure-posenc tokens would dilute every attention average).
  const contentMask = inputs.contentMaskInput; // [B, BOARD_CELLS]

  const projected = tokenProj(cellTokens, config.dim, modelName + "_cell");

  // 2D sinusoidal position on the K/V board tokens so the latents can reason
  // about row/col geometry (attention is permutation-invariant without it).
  // Scale 0.1: projected board content is small (sparse 0/1 channels through
  // a glorot CH→dim proj) — full-amplitude sin/cos would drown it.
  const boardTokens = new Grid2DPositionalEncodingLayer({
    name: modelName + "_posenc",
    rows: BOARD_ROWS,
    cols: BOARD_COLS,
    scale: 0.1,
  }).apply(projected) as tf.SymbolicTensor; // [B, BOARD_CELLS, dim]

  // Latents = the 7 action cells: self + 6 hex neighbors (POINTY_DIRECTIONS
  // order, same as the move/fire action layout). Seeded with their own board
  // tokens (content + posenc), so each latent starts as "what is at my cell".
  const selfPlane = tf.layers
    .reshape({
      name: modelName + "_selfPlane",
      targetShape: [BOARD_CELLS],
    })
    .apply(
      new SliceLayer({
        name: modelName + "_selfChannel",
        beginSlice: [0, 0, BoardChannel.Self],
        sliceSize: [-1, -1, 1],
      }).apply(cellTokens) as tf.SymbolicTensor,
    ) as tf.SymbolicTensor; // [B, BOARD_CELLS]

  const selfLatent = tf.layers
    .reshape({
      name: modelName + "_selfLatent",
      targetShape: [1, config.dim],
    })
    .apply(
      tf.layers
        .dot({
          name: modelName + "_selfToken",
          axes: [1, 1], // one-hot picks the self cell's board token
        })
        .apply([projected, selfPlane]) as tf.SymbolicTensor,
    ) as tf.SymbolicTensor; // [B, 1, dim]

  const neighborLatents = new HexNeighborGatherLayer({
    name: modelName + "_neighborTokens",
  }).apply([projected, selfPlane]) as tf.SymbolicTensor; // [B, 6, dim]

  // Learned slot identity: "I am the self slot / the E-neighbor slot / ...".
  // Posenc above is absolute position; the ROLE relative to self is what must
  // transfer across states (and what off-board zero tokens fall back to).
  const latentSeed = new SlotEmbeddingLayer({
    name: modelName + "_slotEmb",
  }).apply(
    tf.layers
      .concatenate({
        name: modelName + "_latentSeed",
        axis: 1,
      })
      .apply([selfLatent, neighborLatents]) as tf.SymbolicTensor,
  ) as tf.SymbolicTensor; // [B, 7, dim]

  // Modern norm placement, same recipe as applySelfTransformLayers: normalize
  // the seed once, pre-LN blocks (preNorm: true → QNorm/KVNorm inside cross,
  // ln2 before FFN), final norm gathers the residual stream growth.
  const latentNorm = createNormalizationLayer({
    name: modelName + "_latentInputNorm",
  }).apply(latentSeed) as tf.SymbolicTensor;

  const perceived = applyPerceiverLayer({
    name: modelName + "_perceiver",
    depth: config.depth,
    heads: config.heads,
    qTok: latentNorm,
    kvTok: boardTokens,
    kvMask: contentMask, // qMask omitted: all 7 latents always query
    preNorm: true,
  });

  const encoded = createNormalizationLayer({
    name: modelName + "_outputNorm",
  }).apply(perceived) as tf.SymbolicTensor; // [B, 7, dim]

  if (modelName === Model.Policy) {
    // Per-cell action head, same as v1: hold from the self latent, move/fire
    // d from the d-direction latent; move/fire scorers are weight-shared
    // across the 6 directions ("is this a cell worth entering/shooting at"
    // is learned once for all directions).
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

    const dirTokens = new SliceLayer({
      name: modelName + "_dirSlice",
      beginSlice: [0, 1, 0],
      sliceSize: [-1, MOVE_DIR_COUNT, -1],
    }).apply(encoded) as tf.SymbolicTensor; // [B, 6, dim]

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

    const dirLogits = (kind: string, dirs: number) =>
      tf.layers
        .reshape({
          name: modelName + `_${kind}Logits`,
          targetShape: [dirs],
        })
        .apply(
          createDenseLayer({
            name: modelName + `_${kind}Scorer`,
            units: 1,
            useBias: true,
            activation: "linear",
            biasInitializer: "zeros",
            kernelInitializer: logitInit(),
          }).apply(dirTokens) as tf.SymbolicTensor,
        ) as tf.SymbolicTensor; // [B, dirs]

    const logits = tf.layers
      .concatenate({
        name: modelName + "_logits",
      })
      .apply([
        holdLogit,
        dirLogits("move", MOVE_DIR_COUNT),
        dirLogits("fire", FIRE_DIR_COUNT),
      ]) as tf.SymbolicTensor; // [B, ACTION_DIM_TOTAL], order = consts.ts layout

    return { inputs, heads: [logits] };
  }

  // Value: flatten the 7 latents into a single feature; createValueNetwork
  // appends the scalar output.
  const feature = tf.layers
    .flatten({
      name: modelName + "_flatReadout",
    })
    .apply(encoded) as tf.SymbolicTensor; // [B, 7 * dim]

  return { inputs, heads: [feature] };
}
