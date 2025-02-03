import { createResizeSystem } from '../ECS/System/resizeSystem.ts';
import { world } from '../ECS/world.ts';
import { canvas, context, device } from '../gpu.ts';
import { frameTasks } from '../../lib/TasksScheduler/frameTasks.ts';
import { createRopes, createShapes } from './helpers.ts';
import { createDrawRopeSystem } from '../ECS/System/RopeSystem/createDrawRopeSystem.ts';
import { createDrawShapeSystem } from '../ECS/System/SDFSystem/createDrawShapeSystem.ts';

const resizeSystem = createResizeSystem(canvas);
const drawRopeSystem = createDrawRopeSystem(world, device);
const drawShapeSystem = createDrawShapeSystem(world, device);

function frame() {
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

    resizeSystem();
    drawRopeSystem(passEncoder);
    drawShapeSystem(passEncoder);

    passEncoder.end();

    device.queue.submit([
        commandEncoder.finish(),
    ]);
}

createRopes(world);
createShapes(world);

frameTasks.addInterval(frame, 1);
