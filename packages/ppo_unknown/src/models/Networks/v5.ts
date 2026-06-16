/**
 * v5 — drops v4's perceiver for a plain SELF-ATTENTION encoder over the board and
 * a full-board aim head. The board attends to itself (every cell reads every
 * content cell), then two per-cell MLP scorers read the encoded cells:
 *
 *   board [ROWS, COLS, CH] → [ROWS*COLS, CH] tokens → proj dModel
 *     → N × self-attn blocks "the map talks to itself" (kvMask = contentMask,
 *       pre-LN, input/output norm) → encoded [CELLS, dModel]
 *     → Policy:
 *         move/hold: gather the 7 action cells (self + ring 1) → per-cell MLP
 *           scorer → logits [7] (self → Hold, ring cell d → MoveStep d)
 *         fire: per-cell MLP scorer over the WHOLE board → logits [CELLS]
 *           (the agent may fire at ANY reachable cell — see consts.ts /
 *           computeActionMask). The fire logit of cell c reads the very token
 *           that perceived cell c.
 *         → concat → logits [7 + CELLS] = ACTION_DIM_TOTAL
 *     → Value: encode → global-average-pool the content → feature [dModel]
 *
 * The fire action space is now the board window itself, so the head IS the map:
 * "shoot at this cell" scores exactly that cell's encoded token. Move stays the
 * 7 egocentric cells (self always the window center — the board is egocentric).
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
  applyGlobalAverage1d,
  applySelfTransformLayers,
  createDenseLayer,
  tokenProj,
} from "../../../../ppo/src/models/ApplyLayers.ts";
import { HexRingGatherLayer } from "../Layers/HexRingGatherLayer.ts";
import { ACTION_CELL_INDEXES } from "../../state/hexNeighbors.ts";
import { BOARD_CELLS, BOARD_CHANNELS, MOVE_DIR_COUNT } from "../dims.ts";
import { createInputs } from "../Inputs.ts";

type NetworkConfig = { dim: number; heads: number; depth: number };

const policyConfig: NetworkConfig = { dim: 128, heads: 4, depth: 3 };
const valueConfig: NetworkConfig = { dim: 128, heads: 4, depth: 2 };

/** Self (the window center) + ring-1 cells — the 7 move/hold action cells. */
const MOVE_CELL_INDEXES = ACTION_CELL_INDEXES.slice(0, 1 + MOVE_DIR_COUNT);

/**
 * Per-cell MLP scorer: a shared 2-layer head applied to every token along the
 * cell axis → one logit per cell → reshape to a flat [B, count] logit slice.
 * Near-zero init on the final scorer (orthogonal gain 0.2) so logits start ≈
 * uniform; no bias — an action-type prior must come through the tokens.
 */
function cellScorer(
  name: string,
  tokens: tf.SymbolicTensor, // [B, count, dim]
  dim: number,
  count: number,
): tf.SymbolicTensor {
  const hidden = createDenseLayer({
    name: name + "_scorerHidden",
    units: dim,
    useBias: true,
    activation: "relu",
  }).apply(tokens) as tf.SymbolicTensor; // [B, count, dim]

  const score = createDenseLayer({
    name: name + "_scorer",
    units: 1,
    useBias: false,
    activation: "linear",
    kernelInitializer: tf.initializers.orthogonal({ gain: 0.2 }),
  }).apply(hidden) as tf.SymbolicTensor; // [B, count, 1]

  return tf.layers
    .reshape({ name: name + "_logits", targetShape: [count] })
    .apply(score) as tf.SymbolicTensor; // [B, count]
}

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

  // The map attends to itself: every cell queries, only content cells are read.
  const encoded = applySelfTransformLayers(modelName + "_encoder", {
    depth: config.depth,
    heads: config.heads,
    token: projected,
    kvMask: contentMask,
  }); // [B, BOARD_CELLS, dim]

  if (modelName === Model.Policy) {
    const moveTokens = new HexRingGatherLayer({
      name: modelName + "_moveCells",
      indexes: MOVE_CELL_INDEXES,
    }).apply(encoded) as tf.SymbolicTensor; // [B, 7, dim]

    const moveLogits = cellScorer(
      modelName + "_move",
      moveTokens,
      config.dim,
      MOVE_CELL_INDEXES.length,
    ); // [B, 7] — cell 0 (self) → Hold, cell 1+d → MoveStep d

    const fireLogits = cellScorer(modelName + "_fire", encoded, config.dim, BOARD_CELLS); // [B, CELLS]

    const logits = tf.layers
      .concatenate({ name: modelName + "_logits" })
      .apply([moveLogits, fireLogits]) as tf.SymbolicTensor; // [B, ACTION_DIM_TOTAL]

    return { inputs, heads: [logits] };
  }

  // Value: pool the encoded board into a single feature; createValueNetwork
  // appends the scalar output.
  const feature = applyGlobalAverage1d({ name: modelName + "_pool" }, encoded); // [B, dim]

  return { inputs, heads: [feature] };
}
