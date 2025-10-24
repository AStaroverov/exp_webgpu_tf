import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

interface Config extends LayerArgs {
    len: number;
}

export class TrainableLogStdLayer extends tf.layers.Layer {
    static readonly className = 'TrainableLogStdLayer';

    private len: number;
    private logStd!: tf.LayerVariable;

    constructor(config: Config) {
        super(config);
        this.len = config.len;
    }

    build(): void {
        this.logStd = this.addWeight(
            'logStd',
            [this.len],
            'float32',
            tf.initializers.zeros(),
            undefined,
            true
        );
    }

    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor | tf.Tensor[] {
        const x = Array.isArray(inputs) ? inputs[0] : inputs;
        const batchSize = x.shape[0] ?? 1;
        return this.logStd.read().reshape([1, this.len]).tile([batchSize, 1]);
    }

    getConfig() {
        const baseConfig = super.getConfig();
        return { ...baseConfig, len: this.len };
    }
}

tf.serialization.registerClass(TrainableLogStdLayer);