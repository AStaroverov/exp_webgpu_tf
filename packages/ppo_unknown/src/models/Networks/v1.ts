/**
 * v1 — a simple transformer over the chess-like board, the MVP network for
 * ppo_unknown. Each of the 64 board cells becomes a token (its CHANNELS vector),
 * so the policy can reason about spatial relations via self-attention — closer to
 * tanks' attention nets than a conv stack, but minimal.
 *
 *   board [ROWS, COLS, CH] → [ROWS*COLS, CH] tokens → proj dModel → +posenc
 *     → N self-attention layers (content-masked)
 *     → Policy: per-cell head — hold from the self token, move/fire d from the
 *       d-direction neighbor token (direction-shared scorers) → logits [13]
 *     → Value: flatten all encoded tokens → feature [ROWS*COLS*dim]
 *
 * Contract (consumed by createUnknownNetworks):
 *   createNetwork(model) → { inputs, heads }
 *     - inputs: Record<string, tf.SymbolicTensor> — Object.values order matches
 *       state/InputTensors.ts.
 *     - heads: tf.SymbolicTensor[] — Policy: FINAL logits, one per
 *       ACTION_HEAD_DIMS entry; Value: a single feature (heads[0]).
 */

import * as tf from '@tensorflow/tfjs';
import { Model } from '../../../../ppo/src/models/def.ts';
import {
    applySelfTransformLayers, createDenseLayer, tokenProj
} from '../../../../ppo/src/models/ApplyLayers.ts';
import { Grid2DPositionalEncodingLayer } from '../../../../ppo/src/models/Layers/Grid2DPositionalEncodingLayer.ts';
import { SliceLayer } from '../../../../ppo/src/models/Layers/SliceLayer.ts';
import { HexNeighborGatherLayer } from '../Layers/HexNeighborGatherLayer.ts';
import {
    BOARD_CELLS, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS, BoardChannel, FIRE_DIR_COUNT, MOVE_DIR_COUNT,
} from '../dims.ts';
import { createInputs } from '../Inputs.ts';

type NetworkConfig = { dim: number; heads: number; depth: number };

const policyConfig: NetworkConfig = { dim: 64, heads: 4, depth: 4 };
const valueConfig: NetworkConfig = { dim: 32, heads: 2, depth: 2 };

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

    // Asymmetric attention masking: only cells that carry content
    // (units/obstacles/threats) can be READ — ~55 of 64 cells are empty, and
    // their pure-posenc tokens would dilute every attention average. But ALL
    // cells query: an empty cell still looks around and gathers context, so its
    // encoded token means "what is visible from here" — which is exactly what
    // the per-cell action head reads for move/fire targets. The mask arrives as
    // a separate input (computed in createInputTensors).
    const contentMask = inputs.contentMaskInput; // [B, BOARD_CELLS]

    const projected = tokenProj(cellTokens, config.dim, modelName + '_cell');

    // 2D sinusoidal position so self-attention can reason about row/col geometry;
    // without it the flattened cells are an unordered bag (attention is permutation-
    // invariant) and the board's position carries the only spatial signal we have.
    // Scale 0.2: projected board content is small (sparse 0/1 channels through a
    // glorot CH→dim proj, ~0.1–0.3 per coord) — full-amplitude sin/cos would drown
    // it and make the encoder output nearly state-independent.
    const tokens = new Grid2DPositionalEncodingLayer({
        name: modelName + '_posenc',
        rows: BOARD_ROWS,
        cols: BOARD_COLS,
        scale: 0.1,
    }).apply(projected) as tf.SymbolicTensor;

    const encoded = applySelfTransformLayers(modelName + '_encoder', {
        depth: config.depth,
        heads: config.heads,
        token: tokens,
        kvMask: contentMask, // qMask omitted: all cells query
    });
    if (modelName === Model.Policy) {
        // Per-cell action head (AlphaZero/Chessformer style): each directional
        // action's logit is read from the token of THAT direction's neighbor
        // cell — hold from the self token, move/fire d from neighbor d. No
        // board-to-one-vector compression, and the move/fire scorers are
        // weight-shared across the 6 directions (the hex analog of AlphaZero's
        // 1x1-conv move planes): "is this a cell worth entering/shooting at"
        // is learned once for all directions.
        const selfPlane = tf.layers.reshape({
            name: modelName + '_selfPlane',
            targetShape: [BOARD_CELLS],
        }).apply(new SliceLayer({
            name: modelName + '_selfChannel',
            beginSlice: [0, 0, BoardChannel.Self],
            sliceSize: [-1, -1, 1],
        }).apply(cellTokens) as tf.SymbolicTensor) as tf.SymbolicTensor; // [B, BOARD_CELLS]

        const selfToken = tf.layers.dot({
            name: modelName + '_selfToken',
            axes: [1, 1], // one-hot picks the self cell's encoded token
        }).apply([encoded, selfPlane]) as tf.SymbolicTensor; // [B, dim]

        const neighborTokens = new HexNeighborGatherLayer({
            name: modelName + '_neighborTokens',
        }).apply([encoded, selfPlane]) as tf.SymbolicTensor; // [B, 6, dim]

        // Near-zero init on the scorers: logits start ≈ uniform (no random
        // constant prior), state signal shows as crossings immediately.
        const logitInit = () => tf.initializers.orthogonal({ gain: 0.2 });
        const holdLogit = createDenseLayer({
            name: modelName + '_holdLogit',
            units: 1,
            useBias: true,
            activation: 'linear',
            biasInitializer: 'zeros',
            kernelInitializer: logitInit(),
        }).apply(selfToken) as tf.SymbolicTensor; // [B, 1]

        const dirLogits = (kind: string, dirs: number) => tf.layers.reshape({
            name: modelName + `_${kind}Logits`,
            targetShape: [dirs],
        }).apply(createDenseLayer({
            name: modelName + `_${kind}Scorer`,
            units: 1,
            useBias: true,
            activation: 'linear',
            biasInitializer: 'zeros',
            kernelInitializer: logitInit(),
        }).apply(neighborTokens) as tf.SymbolicTensor) as tf.SymbolicTensor; // [B, dirs]

        const logits = tf.layers.concatenate({
            name: modelName + '_logits',
        }).apply([
            holdLogit,
            dirLogits('move', MOVE_DIR_COUNT),
            dirLogits('fire', FIRE_DIR_COUNT),
        ]) as tf.SymbolicTensor; // [B, ACTION_DIM_TOTAL], order = consts.ts layout

        return { inputs, heads: [logits] };
    }

    // Value: flatten all tokens into a single feature; createValueNetwork
    // appends the scalar output.
    const feature = tf.layers.flatten({
        name: modelName + '_flatReadout',
    }).apply(encoded) as tf.SymbolicTensor; // [B, BOARD_CELLS * dim]

    return { inputs, heads: [feature] };
}
