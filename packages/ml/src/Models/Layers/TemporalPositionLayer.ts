import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { getInitializer, Initializer, InitializerIdentifier, serializeInitializer } from '@tensorflow/tfjs-layers/dist/initializers';
import { HISTORY_LENGTH } from '../../../../ml-common/historyConfig.ts';

/**
 * Learned temporal position encoding layer.
 * Stores a [HISTORY_LENGTH, dModel] variable, tiles each position `slotsPerFrame` times
 * to produce [B, HISTORY_LENGTH * slotsPerFrame, dModel], then adds to input.
 */
export class TemporalPositionLayer extends tf.layers.Layer {
    static readonly className = 'TemporalPositionLayer';

    private positionVariable!: tf.LayerVariable;
    private readonly dModel: number;
    private readonly slotsPerFrame: number;
    private readonly initializer: Initializer;

    constructor(config: LayerArgs & {
        dModel: number;
        slotsPerFrame: number;
        initializer?: InitializerIdentifier | Initializer;
    }) {
        super(config);
        this.dModel = config.dModel;
        this.slotsPerFrame = config.slotsPerFrame;
        this.initializer = config.initializer instanceof Initializer
            ? config.initializer
            : getInitializer(config.initializer ?? 'zeros');
    }

    build(inputShape: tf.Shape | tf.Shape[]): void {
        this.positionVariable = this.addWeight(
            'temporal_position',
            [HISTORY_LENGTH, this.dModel],
            'float32',
            this.initializer,
        );
        super.build(inputShape);
    }

    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
        const input = Array.isArray(inputs) ? inputs[0] : inputs;

        return tf.tidy(() => {
            const positions = this.positionVariable.read(); // [T, dModel]
            // Repeat each temporal position slotsPerFrame times: [T, dModel] -> [T * slotsPerFrame, dModel]
            // Use reshape + tile: [T, 1, dModel] -> [T, slotsPerFrame, dModel] -> [T*slotsPerFrame, dModel]
            const expanded = positions.expandDims(1); // [T, 1, dModel]
            const tiled = expanded.tile([1, this.slotsPerFrame, 1]); // [T, slotsPerFrame, dModel]
            const flat = tiled.reshape([HISTORY_LENGTH * this.slotsPerFrame, this.dModel]); // [T*slotsPerFrame, dModel]
            // Broadcast over batch: [1, T*slotsPerFrame, dModel]
            const batched = flat.expandDims(0); // [1, T*slotsPerFrame, dModel]
            return input.add(batched);
        });
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]) {
        return inputShape;
    }

    getConfig(): tf.serialization.ConfigDict {
        const config = super.getConfig();
        return {
            ...config,
            dModel: this.dModel,
            slotsPerFrame: this.slotsPerFrame,
            initializer: serializeInitializer(this.initializer),
        };
    }
}

tf.serialization.registerClass(TemporalPositionLayer);
