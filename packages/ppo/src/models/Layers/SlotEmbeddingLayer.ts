import * as tf from "@tensorflow/tfjs";
import { Layer } from "@tensorflow/tfjs-layers/dist/engine/topology";

/**
 * SlotEmbeddingLayer — adds a learned per-slot vector to a fixed-length token
 * sequence [B, N, D] (+= emb[1, N, D]). Gives each slot a trainable identity:
 * unlike RoleEmbeddingLayer (one shared vector for the whole sequence), every
 * position gets its own. Use when slots have fixed roles (e.g. perceiver
 * decoder queries: self + 6 hex directions).
 */
export class SlotEmbeddingLayer extends Layer {
  static readonly className = "SlotEmbeddingLayer";

  private emb!: tf.LayerVariable;

  build(inputShape: tf.Shape | tf.Shape[]) {
    const shape = (
      inputShape[0] === null || typeof inputShape[0] === "number" ? inputShape : inputShape[0]
    ) as tf.Shape;

    const [, slots, dModel] = shape;
    this.emb = this.addWeight(
      "slot_emb",
      [1, slots!, dModel!],
      "float32",
      tf.initializers.randomNormal({ stddev: 0.02 }),
    );
    this.built = true;
  }

  call(inputs: tf.Tensor | tf.Tensor[]) {
    const x = Array.isArray(inputs) ? inputs[0] : inputs; // [B, N, D]
    return tf.add(x, this.emb.read());
  }
}

tf.serialization.registerClass(SlotEmbeddingLayer);
