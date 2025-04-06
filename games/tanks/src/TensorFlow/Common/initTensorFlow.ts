let tf!: typeof import('@tensorflow/tfjs') | typeof import('@tensorflow/tfjs-node');

if (typeof globalThis.document === 'undefined') {
    tf = await import('@tensorflow/tfjs-node');
}

if (typeof globalThis.document !== 'undefined') {
    tf = await import('@tensorflow/tfjs');
}

async function setBackend(type: 'node' | 'wasm' | 'webgpu') {
    if (type === 'node') {

    }
    if (type === 'wasm') {
        const { setWasmPath } = await import('@tensorflow/tfjs-backend-wasm');
        setWasmPath('/node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm');
        await tf.setBackend('wasm');
    }
    if (type === 'webgpu') {
        await import('@tensorflow/tfjs-backend-webgpu');
        await tf.setBackend('webgpu');
    }
    await tf.ready();
    console.log(`Backend: ${ tf.getBackend() }`);
}

console.log(`TensorFlow.js version: ${ tf.version.tfjs }`);

export { tf, setBackend };