import * as tf from '@tensorflow/tfjs';
import { Layer, LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

export class CloneLayer extends Layer {
    static readonly className = 'CloneLayer';

    constructor(config: LayerArgs) {
        super(config);
    }

    call(inputs: tf.Tensor ): tf.Tensor  {
        const input = (Array.isArray(inputs) ? inputs[0] : inputs) as tf.Tensor;
        return input.clone();
    }
}

tf.serialization.registerClass(CloneLayer);