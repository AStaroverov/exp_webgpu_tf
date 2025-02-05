import { IWorld } from 'bitecs';
import { renderWorld } from '../ECS/renderWorld.ts';
import { createResizeSystem } from '../ECS/Systems/resizeSystem.ts';

export function createFrameTick(
    canvas: HTMLCanvasElement,
    device: GPUDevice,
    context: GPUCanvasContext,
    callback: (options: {
        world: IWorld,
        context: GPUCanvasContext,
        device: GPUDevice,
        passEncoder: GPURenderPassEncoder
    }) => void,
) {
    const resizeSystem = createResizeSystem(canvas);

    const arg = {
        world: renderWorld,
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
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
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