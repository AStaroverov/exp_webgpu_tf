import * as tf from "@tensorflow/tfjs";
import { LayerArgs } from "@tensorflow/tfjs-layers/dist/engine/topology";
import {
  getInitializer,
  Initializer,
  InitializerIdentifier,
  serializeInitializer,
} from "@tensorflow/tfjs-layers/dist/initializers";
import { gatherND } from "./gatherND";
import { topK } from "./topK";

/**
 * Mixture of Experts Layer configuration
 */
export interface MoELayerConfig extends LayerArgs {
  /** Number of top experts to route to (default: 2, as in Mixtral/GLaM) */
  topK: number;

  /**
   * Hidden dimension for each expert FFN.
   * Often 4 * inputDim for dense FFN; in MoE you may scale by 1/numExperts
   * to keep total params comparable.
   */
  expertDim: number;

  /** Number of expert networks */
  numExperts: number;

  /**
   * Temperature for router softmax (default: 1.0)
   * Higher values (e.g., 1.5-2.0) produce more uniform expert distribution,
   * preventing collapse. Lower values make routing more deterministic.
   */
  routerTemperature?: number;

  /**
   * Dropout rate for router logits during training (default: 0.1)
   * Adds noise to routing decisions, preventing overconfident routing.
   */
  routerDropout?: number;

  /**
   * Dropout rate for experts during training (default: 0.1)
   * Randomly masks entire experts, forcing the model to not rely on a single expert.
   */
  expertDropout?: number;

  /**
   * Jitter noise multiplier for router logits during training (default: 0.0)
   * From Switch Transformer: adds multiplicative noise to encourage exploration.
   * Set to 0 to disable.
   */
  jitterNoise?: number;

  /** Kernel initializer for expert weights */
  kernelInitializer?: InitializerIdentifier | Initializer;
}

/**
 * Mixture of Experts (MoE) Layer
 *
 * Implements sparse MoE where each token is routed to top-K experts.
 * This allows for increased model capacity without proportionally
 * increasing computation.
 *
 * Architecture:
 * 1. Router: Linear layer that produces logits for each expert
 * 2. Experts: Multiple FFN networks (up projection -> activation -> down projection)
 * 3. Top-K Gating: Select top-K experts and normalize their weights
 * 4. Weighted combination of expert outputs
 *
 * References:
 * - "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
 *   (Shazeer et al., 2017) https://arxiv.org/abs/1701.06538
 * - "Switch Transformers" (Fedus et al., 2021) https://arxiv.org/abs/2101.03961
 * - "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts"
 *   (Du et al., 2022) https://arxiv.org/abs/2112.06905
 */
export class MoELayer extends tf.layers.Layer {
  static readonly className = "MoELayer";

  private topK: number;
  private expertDim: number;
  private numExperts: number;
  private routerDropout: number;
  private expertDropout: number;
  private jitterNoise: number;
  private kernelInitializer: Initializer;

  private inputDim!: number;

  // Router weights
  private routerKernel!: tf.LayerVariable;

  // Expert weights (stored as 3D tensors for efficient computation)
  // Shape: [numExperts, inputDim, expertDim] for up projection
  // Shape: [numExperts, expertDim, inputDim] for down projection
  private expertUpKernel!: tf.LayerVariable;
  private expertDownKernel!: tf.LayerVariable;

  constructor(config: MoELayerConfig) {
    super(config);
    this.expertDim = config.expertDim;
    this.numExperts = config.numExperts;
    this.topK = config.topK ?? 2;
    this.routerDropout = config.routerDropout ?? 0.1;
    this.expertDropout = config.expertDropout ?? 0.1;
    this.jitterNoise = config.jitterNoise ?? 0.0;
    this.kernelInitializer =
      config.kernelInitializer instanceof Initializer
        ? config.kernelInitializer
        : getInitializer(config.kernelInitializer ?? "glorotNormal");
  }

  build(inputShape: tf.Shape | tf.Shape[]): void {
    const shape = (
      inputShape[0] === null || typeof inputShape[0] === "number" ? inputShape : inputShape[0]
    ) as tf.Shape;

    this.inputDim = shape[shape.length - 1]!;

    // Router: maps input to expert logits
    // Shape: [inputDim, numExperts]
    this.routerKernel = this.addWeight(
      "routerKernel",
      [this.inputDim, this.numExperts],
      "float32",
      this.kernelInitializer,
      undefined,
      true,
    );

    // Expert up projections (all experts in one tensor for efficiency)
    // Shape: [numExperts, inputDim, expertDim]
    this.expertUpKernel = this.addWeight(
      "expertUpKernel",
      [this.numExperts, this.inputDim, this.expertDim],
      "float32",
      this.kernelInitializer,
      undefined,
      true,
    );

    // Expert down projections (all experts in one tensor)
    // Shape: [numExperts, expertDim, inputDim]
    this.expertDownKernel = this.addWeight(
      "expertDownKernel",
      [this.numExperts, this.expertDim, this.inputDim],
      "float32",
      this.kernelInitializer,
      undefined,
      true,
    );

    super.build(inputShape);
  }

  call(inputs: tf.Tensor | tf.Tensor[], kwargs?: { training?: boolean }): tf.Tensor {
    const training = kwargs?.training ?? false;

    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;
      const inputShape = input.shape;

      // Flatten to [batch * seq, inputDim] for easier processing
      const batchSeq = inputShape.slice(0, -1).reduce((a, b) => (a ?? 1) * (b ?? 1), 1)!;
      const flatInput = input.reshape([batchSeq, this.inputDim]);

      // Compute router logits: [batchSeq, numExperts]
      let routerLogits = flatInput.matMul(this.routerKernel.read());

      // === Architectural collapse prevention (no auxiliary loss needed) ===

      // 1. Add jitter noise during training (multiplicative noise for exploration)
      if (training && this.jitterNoise > 0) {
        const noise = tf.randomUniform(routerLogits.shape, -this.jitterNoise, this.jitterNoise);
        routerLogits = routerLogits.mul(tf.scalar(1).add(noise));
      }

      // 2. Apply router dropout during training (adds noise to routing decisions)
      if (training && this.routerDropout > 0) {
        routerLogits = tf.dropout(routerLogits, this.routerDropout);
      }

      // Compute softmax probabilities for all experts
      const routerProbs = tf.softmax(routerLogits); // [batchSeq, numExperts]

      // Get top-K expert indices and their weights (using custom topK with gradient support)
      const { values: topKValues, indices: topKIndices } = topK(routerProbs, this.topK);

      // Normalize top-K weights to sum to 1
      const topKWeightsSum = topKValues.sum(-1, true);
      const topKWeights = topKValues.div(topKWeightsSum.add(1e-9)); // [batchSeq, topK]

      // Process through experts
      // For efficiency with top-K, we compute all experts and then mask
      // In production with many experts, you'd want sparse computation

      // Compute all expert outputs at once
      // input: [batchSeq, inputDim]
      // expertUpKernel: [numExperts, inputDim, expertDim]

      // Compute up projection for all experts: [batchSeq, numExperts, expertDim]
      // Using transpose + reshape instead of einsum (not supported on WASM backend)
      // Transpose expertUpKernel from [numExperts, inputDim, expertDim] to [inputDim, numExperts, expertDim]
      // Then reshape to [inputDim, numExperts * expertDim] for efficient matMul
      const upKernelReshaped = this.expertUpKernel
        .read()
        .transpose([1, 0, 2])
        .reshape([this.inputDim, this.numExperts * this.expertDim]);
      // matMul: [batchSeq, inputDim] @ [inputDim, numExperts * expertDim] = [batchSeq, numExperts * expertDim]
      const upProj = flatInput
        .matMul(upKernelReshaped)
        .reshape([batchSeq, this.numExperts, this.expertDim]);

      // Apply activation (using SiLU/Swish which is common in modern MoE)
      const activated = tf.mul(upProj, tf.sigmoid(upProj)); // SiLU activation

      // Compute down projection: [batchSeq, numExperts, inputDim]
      // Using batched matMul instead of einsum (not supported on WASM backend)
      // Transpose activated from [batchSeq, numExperts, expertDim] to [numExperts, batchSeq, expertDim]
      const activatedT = activated.transpose([1, 0, 2]);
      // Batched matMul: [numExperts, batchSeq, expertDim] @ [numExperts, expertDim, inputDim] = [numExperts, batchSeq, inputDim]
      const expertOutputsT = tf.matMul(activatedT, this.expertDownKernel.read());
      // Transpose back to [batchSeq, numExperts, inputDim]
      let expertOutputs = expertOutputsT.transpose([1, 0, 2]);

      // 4. Expert dropout: randomly zero out entire experts during training
      // Forces model to not rely on a single expert
      if (training && this.expertDropout > 0) {
        // Create dropout mask per expert: [1, numExperts, 1]
        const expertMask = tf
          .randomUniform([1, this.numExperts, 1])
          .greaterEqual(this.expertDropout)
          .cast("float32")
          .div(1 - this.expertDropout); // Scale to maintain expected value
        expertOutputs = expertOutputs.mul(expertMask);
      }

      // Gather outputs from top-K experts and combine with weights
      // topKIndices: [batchSeq, topK]
      // topKWeights: [batchSeq, topK]
      // expertOutputs: [batchSeq, numExperts, inputDim]

      // Gather top-K expert outputs
      const batchIndices = tf.range(0, batchSeq, 1, "int32").expandDims(1).tile([1, this.topK]);
      const gatherIndices = tf.stack(
        [batchIndices.flatten(), topKIndices.flatten().cast("int32")],
        1,
      );
      const gatheredOutputs = gatherND(expertOutputs, gatherIndices).reshape([
        batchSeq,
        this.topK,
        this.inputDim,
      ]);

      // Weighted sum of top-K expert outputs
      // topKWeights: [batchSeq, topK] -> [batchSeq, topK, 1]
      const weightedOutputs = gatheredOutputs.mul(topKWeights.expandDims(-1));
      const output = weightedOutputs.sum(1); // [batchSeq, inputDim]

      // Reshape back to original shape
      return output.reshape([...inputShape.slice(0, -1), this.inputDim]);
    });
  }

  computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape {
    const shape = (
      inputShape[0] === null || typeof inputShape[0] === "number" ? inputShape : inputShape[0]
    ) as tf.Shape;

    // Output shape is same as input shape (MoE replaces FFN)
    return shape;
  }

  getConfig(): tf.serialization.ConfigDict {
    const config = super.getConfig();
    return {
      ...config,
      numExperts: this.numExperts,
      expertDim: this.expertDim,
      topK: this.topK,
      routerDropout: this.routerDropout,
      expertDropout: this.expertDropout,
      jitterNoise: this.jitterNoise,
      kernelInitializer: serializeInitializer(this.kernelInitializer),
    };
  }
}

tf.serialization.registerClass(MoELayer);
