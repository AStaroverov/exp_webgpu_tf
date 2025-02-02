import { ShaderMeta } from './ShaderMeta.ts';
import { GPUVariable } from '../WebGPU/GPUVariable.ts';
import { presentationFormat } from '../gpu.ts';


export class GPUShader<M extends ShaderMeta<any, any>> {
    uniforms = {} as Record<keyof M['uniforms'], GPUVariable>;
    attributes = {} as Record<keyof M['attributes'], GPUVariable>;

    private shaderModule?: GPUShaderModule;
    private renderPipeline?: GPURenderPipeline;
    private bindGroup?: GPUBindGroup;

    constructor(public shaderMeta: M) {
        for (const key in shaderMeta.uniforms) {
            this.uniforms[key as keyof M['uniforms']] = new GPUVariable(shaderMeta.uniforms[key]);
        }

        for (const key in shaderMeta.attributes) {
            this.attributes[key as keyof M['attributes']] = new GPUVariable(shaderMeta.attributes[key]);
        }
    }

    getShaderModule(device: GPUDevice) {
        return this.shaderModule ?? (this.shaderModule = device.createShaderModule({
            code: this.shaderMeta.shader,
        }));
    }

    getRenderPipeline(device: GPUDevice, shaderModule = this.getShaderModule(device)): GPURenderPipeline {
        return this.renderPipeline ?? (this.renderPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: this.shaderMeta.vertexName,
            },
            fragment: {
                module: shaderModule,
                entryPoint: this.shaderMeta.fragmentName,
                targets: [
                    {
                        format: presentationFormat,
                    },
                ],
            },
            primitive: {
                topology: 'triangle-list',
            },
        }));
    }

    getBindGroup(device: GPUDevice, pipeline: GPURenderPipeline = this.getRenderPipeline(device)): GPUBindGroup {
        return this.bindGroup ?? (this.bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: Object.values(this.uniforms).map(u => u.getBindGroupEntry(device)),
        }));
    }

    destroy() {
        for (const key in this.uniforms) {
            this.uniforms[key].destroy();
        }

        for (const key in this.attributes) {
            this.attributes[key].destroy();
        }

        this.attributes = null!;
        this.uniforms = null!;
        this.bindGroup = null!;
    }
}