import { world } from '../src/ECS/world.ts';
import { initWebGPU } from '../src/gpu.ts';
import { frameTasks } from '../lib/TasksScheduler/frameTasks.ts';
import { createRopes, createShapes } from './helpers.ts';
import { createDrawRopeSystem } from '../src/ECS/System/RopeSystem/createDrawRopeSystem.ts';
import { createDrawShapeSystem } from '../src/ECS/System/SDFSystem/createDrawShapeSystem.ts';
import { createFrameTick } from '../src/WGSL/createFrame.ts';

const canvas = document.querySelector('canvas')!;
const { device, context } = await initWebGPU(canvas);
const drawRopeSystem = createDrawRopeSystem(world, device);
const drawShapeSystem = createDrawShapeSystem(world, device);

createRopes(world, canvas);
createShapes(world);

const frame = createFrameTick(canvas, device, context, ({ passEncoder }) => {
    drawRopeSystem(passEncoder);
    drawShapeSystem(passEncoder);
});

frameTasks.addInterval(frame, 1);
