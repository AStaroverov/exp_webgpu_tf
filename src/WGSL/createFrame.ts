import { IWorld } from 'bitecs';
import { world } from '../ECS/world.ts';
import { createResizeSystem } from '../ECS/Systems/resizeSystem.ts';

export function createFrameTick(
    { canvas, device, context, background }: {
        canvas: HTMLCanvasElement,
        device: GPUDevice,
        context: GPUCanvasContext,
        background: GPUColor,

    }, callback: (options: {
        world: IWorld,
        context: GPUCanvasContext,
        device: GPUDevice,
        passEncoder: GPURenderPassEncoder
    }) => void) {
    const resizeSystem = createResizeSystem(canvas);

    const arg = {
        world: world,
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