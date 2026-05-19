import * as tf from '@tensorflow/tfjs';
import { Layer, LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

export class SliceLayer extends Layer {
    static readonly className = 'SliceLayer';

    private beginSlice: number[];
    private sliceSize: number[];

    constructor(config: LayerArgs & { beginSlice: number[], sliceSize: number[] } ) {
        super(config);
        this.beginSlice = config.beginSlice;
        this.sliceSize = config.sliceSize;
    }

    computeOutputShape(inputShape: number[]): number[] {
        return this.sliceSize.map((size, i) => {
            return size === -1 ? inputShape[i] : size;
        });
    }
    
    call(inputs: tf.Tensor ): tf.Tensor  {
        const input = (Array.isArray(inputs) ? inputs[0] : inputs) as tf.Tensor;
        return input.slice(this.beginSlice, this.sliceSize);
    }

    getConfig() {
        const config = super.getConfig();
        return {
            ...config,
            beginSlice: this.beginSlice,
            sliceSize: this.sliceSize,
        };
    }
}

tf.serialization.registerClass(SliceLayer);