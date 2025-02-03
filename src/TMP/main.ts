import { shaderMeta } from './ECS/System/RopeSystem/sdf.shader.ts';
import { Variable } from './Variable.ts';

const canvas = document.createElement('canvas');

document.body.appendChild(canvas);

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
console.log('>>', shaderMeta.shader);

const shaderModule = device.createShaderModule({
    code: shaderMeta.shader,
});

const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
        module: shaderModule,
        entryPoint: 'vertex',
    },
    fragment: {
        module: shaderModule,
        entryPoint: 'fragment',
        targets: [
            {
                format: presentationFormat,
            },
        ],
    },
    primitive: {
        topology: 'triangle-list',
    },
});

const UNIFORM_USAGE = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
const STORAGE_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

const uKind = new Variable(shaderMeta.uniforms.uKind, UNIFORM_USAGE);
const uWidth = new Variable(shaderMeta.uniforms.uWidth, UNIFORM_USAGE);
const uColor = new Variable(shaderMeta.uniforms.uColor, UNIFORM_USAGE);
const uPoints = new Variable(shaderMeta.uniforms.uPoints, STORAGE_USAGE);


const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
        uKind.getBindGroupEntry(device),
        uWidth.getBindGroupEntry(device),
        uColor.getBindGroupEntry(device),
        uPoints.getBindGroupEntry(device),
    ],
});

function frame() {
    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    const renderPassDescriptor: GPURenderPassDescriptor = {
        colorAttachments: [
            {
                view: textureView,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            },
        ],
    };

    uColor.set([
        Math.random(),
        Math.random(),
        Math.random(),
        1,
    ]);

    device.queue.writeBuffer(uColor.getGPUBuffer(device), 0, uColor.getBuffer());

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.draw(3);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
