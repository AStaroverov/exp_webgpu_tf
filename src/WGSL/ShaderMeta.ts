import { VariableMeta } from '../Struct/VariableMeta.ts';
import { buildShader, setupVariable } from './utils.ts';
import { TWGSLModule } from './def.ts';

export class ShaderMeta<
    U extends Record<string, VariableMeta>,
    A extends Record<string, VariableMeta>
> {
    shader: string = '';
    uniforms: Record<keyof U, VariableMeta>;
    attributes: Record<keyof A, VariableMeta>;

    constructor(
        uniforms: U,
        attributes: A,
        module: TWGSLModule,
    ) {
        this.uniforms = setupVariable(uniforms);
        this.attributes = setupVariable(attributes);
        this.shader = buildShader(this.uniforms, this.attributes, module);
    }
}

