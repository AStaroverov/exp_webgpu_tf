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

    return (delta: number) => {
        const depthTexture = device.createTexture({
            size: [canvas.width, canvas.height, 1],
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        const depthView = depthTexture.createView();
        const textureView = context.getCurrentTexture().createView();
        const commandEncoder = device.createCommandEncoder();
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

        device.queue.submit([
            commandEncoder.finish(),
        ]);
    };
}