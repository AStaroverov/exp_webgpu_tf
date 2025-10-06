/**
 * This module exports TensorFlow.js based on the environment.
 * - In Node.js: exports @tensorflow/tfjs-node
 * - In Browser: exports @tensorflow/tfjs
 */

// Detect if we're running in Node.js environment
// const isNode =
//     typeof process !== 'undefined' &&
//     process.versions != null &&
//     process.versions.node != null;

// Dynamically import the appropriate TensorFlow package
// const tf = await (isNode
//     ? import('@tensorflow/tfjs-node')
//     : import('@tensorflow/tfjs'));

// export default tf.default;
export * from '@tensorflow/tfjs-node';
