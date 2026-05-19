import * as tf from '@tensorflow/tfjs';
import { applySelfTransformLayers as applySelfTransformerLayers, applySwinTransformerLayer, createDenseLayer } from '../ApplyLayers';
import { SliceLayer } from './SliceLayer';
import { VariableLayer } from './VariableLayer';
import { applyPerceiverLayer } from './PerceiverLayer';

export function createRaysEncoder3({
  name,
  input,
  dModel,
}: {
  name: string;
  input: tf.SymbolicTensor;
  dModel: number;
}) {
  const inputDim = input.shape[input.shape.length - 1]!;
  const ratio = dModel / inputDim;
  if (ratio !== Math.floor(ratio)) {
    throw new Error('dModel must be divisible by inputDim');
  }
    
  input = tf.layers.conv1d({ filters: inputDim * 2, kernelSize: 5, strides: 2, padding: 'same', activation: 'relu', name: name+ '_c1' }).apply(input) as tf.SymbolicTensor;
  input = tf.layers.conv1d({ filters: inputDim * 4, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu', name: name+ '_c2' }).apply(input) as tf.SymbolicTensor;
  input = tf.layers.conv1d({ filters: inputDim * 8, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu', name: name+ '_c3' }).apply(input) as tf.SymbolicTensor;

  const attn = applySelfTransformerLayers(name, {
    depth: 4,
    heads: 4,
    token: input,
  });
  // const slice = new SliceLayer({
  //   name: name + '_slice',
  //   beginSlice: [0, 0, 0],
  //   sliceSize: [-1, 1, -1],
  // }).apply(attn) as tf.SymbolicTensor;
  const final = createDenseLayer({
    name: name + '_final',
    units: dModel,
    useBias: false,
    activation: 'linear',
  }).apply(attn) as tf.SymbolicTensor;
  return final;
}

export function createRaysEncoder4({
  name,
  input,
  dModel,
}: {
  name: string;
  input: tf.SymbolicTensor;
  dModel: number;
}) {
  const window = 4;
  
  // Helper function: SwinAttention block + Conv downsampling
  const applySwinBlock = (
    token: tf.SymbolicTensor,
    heads: number,
    blockIdx: number,
  ): tf.SymbolicTensor => {
    const dim = token.shape[token.shape.length - 1]!;
    const attn = applySwinTransformerLayer({
      name: `${name}_swin_${blockIdx}`,
      heads: heads,
      token: token,
      window: window,
      preNorm: true,
    });
    
    const output = tf.layers.conv1d({
      name: `${name}_conv_${blockIdx}`,
      filters: dim * 2,
      kernelSize: 2,
      strides: 2,
      useBias: false,
      padding: 'same',
    }).apply(attn) as tf.SymbolicTensor;
    
    return output;
  };
  
  let x = input;
  
  // Block 1: SwinAttention + Conv
  x = applySwinBlock(x, 1, 1);
  
  // Block 2: SwinAttention + Conv
  x = applySwinBlock(x, 2, 2);
  
  // Block 3: SwinAttention + Conv
  x = applySwinBlock(x, 4, 3);
  
  // Final projection to dModel
  x = createDenseLayer({
    name: `${name}_final`,
    units: dModel,
    useBias: false,
    activation: 'linear',
  }).apply(x) as tf.SymbolicTensor;

  return x;
}

export function createRaysEncoder5({
  name,
  input,
}: {
  name: string;
  input: tf.SymbolicTensor;
}) {
  const window = 8;
  const heads = 4;
  
  // Helper function: SwinAttention block + Conv downsampling
  const applySwinBlock = (
    token: tf.SymbolicTensor,
    blockIdx: number,
  ): tf.SymbolicTensor => {
    const attn = applySwinTransformerLayer({
      name: `${name}_swin_${blockIdx}`,
      heads: heads,
      token: token,
      window: window,
      preNorm: true,
    });

    return attn;
  };
  const extractSlice = (token: tf.SymbolicTensor, begin: number, size: number) => {
    const slice = new SliceLayer({
      name: name + '_slice' + begin,
      beginSlice: [0, begin, 0],
      sliceSize: [-1, size, -1],
    }).apply(token) as tf.SymbolicTensor;
    const flattened = tf.layers.flatten({ name: name + '_flattened' + begin }).apply(slice) as tf.SymbolicTensor;
    const extended = tf.layers.reshape({ name: name + '_3d' + begin, targetShape: [1, flattened.shape[1]!] }).apply(flattened) as tf.SymbolicTensor;
    return extended;
  };
  
  let x = input;
  
  x = applySwinBlock(x, 1);
  x = applySwinBlock(x, 2);
  x = applySwinBlock(x, 3);

  const s1 = extractSlice(x, window/2, 2);
  const s2 = extractSlice(x, window/2+window, 2);
  const s3 = extractSlice(x, window/2+window*2, 2);
  const s4 = extractSlice(x, window/2+window*3, 2);

  x = tf.layers.concatenate({ name: name + '_concatenate', axis: 1 }).apply([s1, s2, s3, s4]) as tf.SymbolicTensor;

  return x;
}

export function createRaysEncoder6({
  name,
  heads,
  depth,
  input,
}: {
  name: string;
  heads: number;
  depth: number;
  input: tf.SymbolicTensor;
}) {
  const qTok = new VariableLayer({
    name: name + '_qTok',
    shape: [4, input.shape[2]!],
    initializer: 'truncatedNormal',
  }).apply(input) as tf.SymbolicTensor;
  const x = applyPerceiverLayer({
    name: name,
    depth,
    heads,
    qTok: qTok,
    kvTok: input,
    preNorm: true,
  });
  return x;
}