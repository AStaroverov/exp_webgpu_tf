import * as tf from "@tensorflow/tfjs";
import { Layer } from "@tensorflow/tfjs-layers/dist/engine/topology";
import type { LayerArgs } from "@tensorflow/tfjs-layers/dist/engine/topology";

/**
 * Mask-weighted pooling over a token sequence:
 *   inputs [tokens [B, N, D], weights [B, N]] → [B, D]
 *
 * `normalize: true`  → weighted mean (sum / Σweights, eps-guarded) — e.g. average
 *                      over content tokens for a global context feature.
 * `normalize: false` → plain weighted sum — e.g. a one-hot plane (the board's
 *                      `Self` cell) reads out exactly that token.
 */
export class MaskPoolLayer extends Layer {
  static readonly className = "MaskPoolLayer";

  private normalize: boolean;

  constructor(config: LayerArgs & { normalize?: boolean }) {
    super(config);
    this.normalize = config.normalize ?? false;
  }

  computeOutputShape(inputShape: tf.Shape[]): tf.Shape {
    const [tokensShape] = inputShape;
    return [tokensShape[0], tokensShape[2]];
  }

  call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    return tf.tidy(() => {
      const [tokens, weights] = inputs as tf.Tensor[];
      const w = weights.expandDims(-1); // [B, N, 1]
      const sum = tokens.mul(w).sum(1); // [B, D]
      return this.normalize ? sum.div(w.sum(1).add(1e-6)) : sum;
    });
  }

  getConfig() {
    const config = super.getConfig();
    return {
      ...config,
      normalize: this.normalize,
    };
  }
}

tf.serialization.registerClass(MaskPoolLayer);
