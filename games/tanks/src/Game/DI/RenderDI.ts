export const RenderDI: {
    canvas: HTMLCanvasElement;
    device: GPUDevice;
    context: GPUCanvasContext;
    renderFrame: (delta: number) => void;
} = {
    canvas: null as any,
    device: null as any,
    context: null as any,
    renderFrame: null as any,
};
