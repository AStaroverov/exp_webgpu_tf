import { createResizeSystem } from '../ECS/Systems/ResizeSystem.ts';

export function createFrameTick(
    { canvas, device, context, background, getPixelRatio }: {
        canvas: HTMLCanvasElement,
        device: GPUDevice,
        context: GPUCanvasContext,
        background: GPUColor,
        getPixelRatio: () => number,
    }, callback: (options: {
        delta: number,
        context: GPUCanvasContext,
        device: GPUDevice,
        passEncoder: GPURenderPassEncoder
    }) => void) {
    const resizeSystem = createResizeSystem(canvas, getPixelRatio);

    const arg = {
        delta: 0,
        context,
        device,
        passEncoder: null as unknown as GPURenderPassEncoder,
    };

    const depthTexture = device.createTexture({
        size: [canvas.width, canvas.height, 1],
        format: 'depth32float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const renderTexture = device.createTexture({
        size: [canvas.width, canvas.height, 1],
        format: 'bgra8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const renderFrame = (commandEncoder: GPUCommandEncoder, delta: number) => {
        const depthView = depthTexture.createView();
        const textureView = renderTexture.createView();

        const passEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [
                {
                    view: textureView,
                    clearValue: background,
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
            depthStencilAttachment: {
                view: depthView,
                depthClearValue: 0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        });

        arg.delta = delta;
        arg.passEncoder = passEncoder;

        resizeSystem();
        callback(arg);

        passEncoder.end();

        return { renderTexture, depthTexture };
    };

    return { renderFrame, renderTexture };
}