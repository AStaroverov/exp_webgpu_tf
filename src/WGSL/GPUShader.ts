import { ShaderMeta } from './ShaderMeta.ts';
import { GPUVariable } from '../WebGPU/GPUVariable.ts';

export class GPUShader<M extends ShaderMeta<any, any>> {
    uniforms = {} as Record<keyof M['uniforms'], GPUVariable>;
    attributes = {} as Record<keyof M['attributes'], GPUVariable>;

    private shaderModule?: GPUShaderModule;
    private pipelineLayout?: GPUPipelineLayout;
    private mapRenderPipeline: Map<string, GPURenderPipeline> = new Map();
    private mapBindGroup: Map<number, GPUBindGroup> = new Map();
    private mapGPUBindGroupLayout: Map<number, GPUBindGroupLayout> = new Map();

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

    getRenderPipeline(
        device: GPUDevice,
        vertexName: string,
        fragmentName: string,
        shaderModule = this.getShaderModule(device),
    ): GPURenderPipeline {
        const key = `${ vertexName }-${ fragmentName }`;

        if (!this.mapRenderPipeline.has(key)) {
            const value = device.createRenderPipeline({
                layout: this.getGPUPipelineLayout(device),
                primitive: {
                    topology: 'triangle-list',
                },
                vertex: {
                    module: shaderModule,
                    entryPoint: vertexName,
                },
                fragment: {
                    module: shaderModule,
                    entryPoint: fragmentName,
                    targets: [
                        {
                            format: navigator.gpu.getPreferredCanvasFormat(),
                            blend: {
                                color: {
                                    srcFactor: 'src-alpha',
                                    dstFactor: 'one-minus-src-alpha',
                                    operation: 'add',
                                },
                                alpha: {
                                    srcFactor: 'one',
                                    dstFactor: 'one-minus-src-alpha',
                                    operation: 'add',
                                },
                            },
                        },
                    ],
                },
                depthStencil: {
                    format: 'depth32float',
                    depthCompare: 'greater-equal',
                    depthWriteEnabled: true,
                },
            });
            this.mapRenderPipeline.set(key, value);
        }

        return this.mapRenderPipeline.get(key)!;
    }

    getBindGroup(device: GPUDevice, group: number): GPUBindGroup {
        if (!this.mapBindGroup.has(group)) {
            const bindGroupLayout = device.createBindGroup({
                layout: this.createBindGroupLayout(device, group),
                entries: Object.entries(this.uniforms)
                    .filter(([key]) => this.shaderMeta.uniforms[key].group === group)
                    .map(([_, value]) => value.getBindGroupEntry(device)),
            });

            this.mapBindGroup.set(group, bindGroupLayout);
        }

        return this.mapBindGroup.get(group)!;
    }

    createBindGroupLayout(device: GPUDevice, group: number): GPUBindGroupLayout {
        if (!this.mapGPUBindGroupLayout.has(group)) {
            const bindGroupLayout = device.createBindGroupLayout({
                entries: Object.entries(this.uniforms)
                    .filter(([key]) => this.shaderMeta.uniforms[key].group === group)
                    .map(([_, value]) => value.getBindGroupLayoutEntry()),
            });

            this.mapGPUBindGroupLayout.set(group, bindGroupLayout);
        }

        return this.mapGPUBindGroupLayout.get(group)!;
    }

    getGPUPipelineLayout(device: GPUDevice, groups?: number[]): GPUPipelineLayout {
        return this.pipelineLayout ?? (this.pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: Array.from(groups ?? Object.values(this.shaderMeta.uniforms)
                .reduce((acc, u) => acc.add(u.group), new Set<number>())
                .values())
                .map((group) => this.createBindGroupLayout(device, group)),
        }));
    }

    destroy() {
        for (const key in this.uniforms) {
            this.uniforms[key].destroy();
        }

        for (const key in this.attributes) {
            this.attributes[key].destroy();
        }

        this.mapBindGroup.clear();
        this.mapRenderPipeline.clear();
        this.mapGPUBindGroupLayout.clear();

        this.uniforms = null!;
        this.attributes = null!;
    }
}