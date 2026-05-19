import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { getInitializer, Initializer, InitializerIdentifier, serializeInitializer } from '@tensorflow/tfjs-layers/dist/initializers';

export class VariableLayer extends tf.layers.Layer {
    static readonly className = 'VariableLayer';

    private variable!: tf.LayerVariable;
    
    private shape: number[];
    private initializer: Initializer;

    constructor(config: LayerArgs & {
        shape: number[];
        initializer?: InitializerIdentifier | Initializer;
    }) {
        super(config);
        this.shape = config.shape;
        this.initializer = config.initializer instanceof Initializer ? config.initializer : getInitializer(config.initializer ?? 'glorotNormal');
    }

    build(inputShape: tf.Shape | tf.Shape[]): void {
        this.variable = this.addWeight(
            'variable',
            this.shape,
            'float32',
            this.initializer,
        );
        super.build(inputShape);
    }

    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
        const input = Array.isArray(inputs) ? inputs[0] : inputs;
        const batchSize = input.shape[0];
        
        return tf.tidy(() => {
            const variable = this.variable.read();
            const expanded = variable.expandDims(0); // [1, ...shape]
            const result = expanded.tile([batchSize, ...Array(this.shape.length).fill(1)]);
            return result;
        });
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]) {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;
        return [shape[0], ...this.shape];
    }

    getConfig(): tf.serialization.ConfigDict {
        const config = super.getConfig();
        return {
            ...config,
            shape: this.shape,
            initializer: serializeInitializer(this.initializer),
        };
    }
}

tf.serialization.registerClass(VariableLayer);