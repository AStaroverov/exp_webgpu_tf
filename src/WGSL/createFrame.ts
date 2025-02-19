import { createResizeSystem } from '../ECS/Systems/ResizeSystem.ts';

export function createFrameTick(
    { canvas, device, context, background, getPixelRatio }: {
        canvas: HTMLCanvasElement,
        device: GPUDevice,
        context: GPUCanvasContext,
        background: GPUColor,
        getPixelRatio: () => number,
    }, callback: (options: {
        context: GPUCanvasContext,
        device: GPUDevice,
        passEncoder: GPURenderPassEncoder
    }) => void) {
    const resizeSystem = createResizeSystem(canvas, getPixelRatio);

    const arg = {
        context,
        device,
        passEncoder: null as unknown as GPURenderPassEncoder,
    };

    return () => {
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
        });

        arg.passEncoder = passEncoder;

        resizeSystem();
        callback(arg);

        passEncoder.end();

        device.queue.submit([
            commandEncoder.finish(),
        ]);
    };
}