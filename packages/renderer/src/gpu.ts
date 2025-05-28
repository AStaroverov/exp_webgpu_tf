export async function initWebGPU(canvas: HTMLCanvasElement): Promise<{ device: GPUDevice, context: GPUCanvasContext }> {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter === null) throw new Error('No adapter found');

    const device = await adapter.requestDevice();
    const context = canvas.getContext('webgpu') as GPUCanvasContext;

    canvas.width = canvas.clientWidth * window.devicePixelRatio;
    canvas.height = canvas.clientHeight * window.devicePixelRatio;
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    context.configure({
        device,
        format: presentationFormat,
        alphaMode: 'premultiplied',
    });

    return {
        device,
        context,
    };
}