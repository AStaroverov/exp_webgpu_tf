import { ShaderMeta } from './ShaderMeta.ts';
import { GPUVariable } from '../WebGPU/GPUVariable.ts';

export class GPUShader<M extends ShaderMeta<any, any>> {
    uniforms = {} as Record<keyof M['uniforms'], GPUVariable>;
    attributes = {} as Record<keyof M['attributes'], GPUVariable>;

    private shaderModule?: GPUShaderModule;
    private pipelineLayout?: GPUPipelineLayout;
    private mapRenderPipeline: Map<string, GPURenderPipeline> = new Map();
    private mapBindGroup: Map<string, GPUBindGroup> = new Map();
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
        options?: {
            withDepth?: boolean,
            shaderModule?: GPUShaderModule,
            targetFormat?: GPUTextureFormat,
            withBlending?: boolean,
            autoLayout?: boolean,
            /** For autoLayout pipelines: specify which uniforms to include in each bind group */
            bindGroups?: Record<number, (keyof M['uniforms'])[]>,
        },
    ): GPURenderPipeline {
        const withDepth = options?.withDepth ?? false;
        const targetFormat = options?.targetFormat ?? navigator.gpu.getPreferredCanvasFormat();
        const withBlending = options?.withBlending ?? true;
        const autoLayout = options?.autoLayout ?? false;
        const bindGroups = options?.bindGroups;
        const pipelineKey = `${ vertexName }-${ fragmentName }`;
        const key = `${ pipelineKey }-${ withDepth }-${ targetFormat }-${ withBlending }-${ autoLayout }`;
        const shaderModule = options?.shaderModule ?? this.getShaderModule(device);

        if (!this.mapRenderPipeline.has(key)) {
            const pipeline = device.createRenderPipeline({
                layout: autoLayout ? 'auto' : this.getGPUPipelineLayout(device),
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
                            format: targetFormat,
                            blend: withBlending ? {
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
                            } : undefined,
                        },
                    ],
                },
                depthStencil: withDepth ? {
                    format: 'depth32float',
                    depthCompare: 'greater-equal',
                    depthWriteEnabled: true,
                } : undefined,
            });
            this.mapRenderPipeline.set(key, pipeline);

            // Create and cache bind groups for autoLayout pipelines
            if (autoLayout && bindGroups) {
                for (const [groupStr, uniformKeys] of Object.entries(bindGroups)) {
                    const group = Number(groupStr);
                    const bindGroupKey = `${ pipelineKey }-${ group }`;
                    const bindGroup = device.createBindGroup({
                        layout: pipeline.getBindGroupLayout(group),
                        entries: uniformKeys.map((uniformKey) => 
                            this.uniforms[uniformKey].getBindGroupEntry(device)
                        ),
                    });
                    this.mapBindGroup.set(bindGroupKey, bindGroup);
                }
            }
        }

        return this.mapRenderPipeline.get(key)!;
    }

    /**
     * Gets a cached bind group.
     * @param group - bind group index
     * @param vertexName - for autoLayout pipelines, specify entry point names to get the correct bind group
     * @param fragmentName - for autoLayout pipelines, specify entry point names to get the correct bind group
     */
    getBindGroup(device: GPUDevice, group: number, vertexName?: string, fragmentName?: string): GPUBindGroup {
        const key = vertexName && fragmentName 
            ? `${ vertexName }-${ fragmentName }-${ group }`
            : `default-${ group }`;

        if (!this.mapBindGroup.has(key)) {
            // Only create default bind groups (non-autoLayout)
            if (vertexName && fragmentName) {
                throw new Error(`Bind group for ${ key } not found. Make sure to pass bindGroups option to getRenderPipeline.`);
            }
            
            const bindGroup = device.createBindGroup({
                layout: this.createBindGroupLayout(device, group),
                entries: Object.entries(this.uniforms)
                    .filter(([uniformKey]) => this.shaderMeta.uniforms[uniformKey].group === group)
                    .map(([_, value]) => value.getBindGroupEntry(device)),
            });

            this.mapBindGroup.set(key, bindGroup);
        }

        return this.mapBindGroup.get(key)!;
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