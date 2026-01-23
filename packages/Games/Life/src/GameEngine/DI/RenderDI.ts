export const RenderDI: {
    enabled: boolean;
    canvas: HTMLCanvasElement;
    device: GPUDevice;
    context: GPUCanvasContext;
    destroy?: () => void;
    renderFrame?: (delta: number) => void;
} = {
    enabled: false,
    canvas: null as any,
    device: null as any,
    context: null as any,
    destroy: null as any,
    renderFrame: null as any,
};
