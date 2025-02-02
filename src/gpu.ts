export const canvas = document.querySelector('canvas')!;
export const adapter = await navigator.gpu.requestAdapter();
if (adapter === null) throw new Error('No adapter found');

export const device = await adapter.requestDevice();
export const context = canvas.getContext('webgpu') as GPUCanvasContext;

canvas.width = canvas.clientWidth //* window.devicePixelRatio;
canvas.height = canvas.clientHeight //* window.devicePixelRatio;
export const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'premultiplied',
});