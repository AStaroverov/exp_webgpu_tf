/**
 * v1 — a simple transformer over the chess-like board, the MVP network for
 * ppo_unknown. Each of the 64 board cells becomes a token (its CHANNELS vector),
 * so the policy can reason about spatial relations via self-attention — closer to
 * tanks' attention nets than a conv stack, but minimal: no masks, no cross-attn.
 *
 *   board [ROWS, COLS, CH] → [ROWS*COLS, CH] tokens → proj dModel → +posenc
 *     → N self-attention layers → Perceiver decode with K learned query tokens
 *       (one per action head) → per-head feature
 *
 * Contract (consumed by createUnknownNetworks, same as tanks createTankNetworks):
 *   createNetwork(model) → { inputs, heads }
 *     - inputs: Record<string, tf.SymbolicTensor> — Object.values order matches
 *       state/InputTensors.ts.
 *     - heads: tf.SymbolicTensor[] — Policy: one per ACTION_HEAD_DIMS entry;
 *       Value: a single feature (heads[0]).
 */

import * as tf from '@tensorflow/tfjs';
import { Model } from '../../../../ppo/src/models/def.ts';
import {
    applySelfTransformLayers,
    createNormalizationLayer,
    tokenProj,
} from '../../../../ppo/src/models/ApplyLayers.ts';
import { applyPerceiverLayer } from '../../../../ppo/src/models/Layers/PerceiverLayer.ts';
import { Grid2DPositionalEncodingLayer } from '../../../../ppo/src/models/Layers/Grid2DPositionalEncodingLayer.ts';
import { SliceLayer } from '../../../../ppo/src/models/Layers/SliceLayer.ts';
import { VariableLayer } from '../../../../ppo/src/models/Layers/VariableLayer.ts';
import { ACTION_HEAD_DIMS, BOARD_CELLS, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS } from '../dims.ts';
import { createInputs } from '../Inputs.ts';

type NetworkConfig = { dim: number; heads: number; depth: number };

const policyConfig: NetworkConfig = { dim: 64, heads: 4, depth: 2 };
const valueConfig: NetworkConfig = { dim: 32, heads: 2, depth: 1 };

export function createNetwork(
    modelName: Model,
    config: NetworkConfig = modelName === Model.Policy ? policyConfig : valueConfig,
) {
    const inputs = createInputs(modelName);

    // [B, ROWS, COLS, CH] → [B, ROWS*COLS, CH]: one token per board cell.
    const cellTokens = tf.layers.reshape({
        name: modelName + '_cellTokens',
        targetShape: [BOARD_CELLS, BOARD_CHANNELS],
    }).apply(inputs.boardInput) as tf.SymbolicTensor;

    const projected = tokenProj(cellTokens, config.dim, modelName + '_cell');

    // 2D sinusoidal position so self-attention can reason about row/col geometry;
    // without it the flattened cells are an unordered bag (attention is permutation-
    // invariant) and the board's position carries the only spatial signal we have.
    const tokens = new Grid2DPositionalEncodingLayer({
        name: modelName + '_posenc',
        rows: BOARD_ROWS,
        cols: BOARD_COLS,
    }).apply(projected) as tf.SymbolicTensor;

    const encoded = applySelfTransformLayers(modelName + '_encoder', {
        depth: config.depth,
        heads: config.heads,
        token: tokens,
        preNorm: true,
    });
    const numQueries = modelName === Model.Policy ? ACTION_HEAD_DIMS.length : 1;
    const queries = new VariableLayer({
        name: modelName + '_queries',
        shape: [numQueries, config.dim],
        initializer: 'truncatedNormal',
    }).apply(tokens) as tf.SymbolicTensor; // [B, numQueries, dim]

    const decoded = applyPerceiverLayer({
        name: modelName + '_decoder',
        depth: config.depth,
        heads: config.heads,
        qTok: queries,
        kvTok: encoded,
        preNorm: true,
    }); // [B, numQueries, dim]

    const normed = createNormalizationLayer({ name: modelName + '_featureNorm' })
        .apply(decoded) as tf.SymbolicTensor;

    // Slice query token `i` out of [B, numQueries, dim] → [B, dim].
    const queryFeature = (i: number): tf.SymbolicTensor => {
        const sliced = new SliceLayer({
            name: modelName + '_qSlice' + i,
            beginSlice: [0, i, 0],
            sliceSize: [-1, 1, -1],
        }).apply(normed) as tf.SymbolicTensor; // [B, 1, dim]
        return tf.layers.flatten({ name: modelName + '_qFlat' + i })
            .apply(sliced) as tf.SymbolicTensor; // [B, dim]
    };

    if (modelName === Model.Policy) {
        const heads = ACTION_HEAD_DIMS.map((_, i) =>
            tf.layers.dense({
                name: modelName + '_headBranch' + i,
                units: config.dim,
                activation: 'relu',
            }).apply(queryFeature(i)) as tf.SymbolicTensor,
        );
        return { inputs, heads };
    }

    // Value: single feature; createValueNetwork appends the scalar output.
    return { inputs, heads: [queryFeature(0)] };
}
