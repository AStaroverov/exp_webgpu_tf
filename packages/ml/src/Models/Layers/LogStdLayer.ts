import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

interface Config extends LayerArgs {
    units: number;
}

export class LogStdLayer extends tf.layers.Layer {
    static readonly className = 'LogStdLayer';

    private units: number;
    private logStd!: tf.LayerVariable;

    constructor(config: Config) {
        super(config);
        this.units = config.units;
    }

    build(): void {
        this.logStd = this.addWeight(
            'logStd',
            [this.units],
            'float32',
            tf.initializers.constant({ value: 1.5 }),
        );
    }

    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor | tf.Tensor[] {
        const x = Array.isArray(inputs) ? inputs[0] : inputs;
        const batchSize = x.shape[0] ?? 1;
        return this.logStd.read().reshape([1, this.units]).tile([batchSize, 1]);
    }

    getConfig() {
        const baseConfig = super.getConfig();
        return { ...baseConfig, units: this.units };
    }
}

tf.serialization.registerClass(LogStdLayer);