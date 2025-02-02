import {isNil, uniqBy} from "../utils.ts";
import {VariableMeta} from "../Struct/VariableMeta.ts";
import {TWGSLFunc, TWGSLModule, TWGSLPart} from "./def.ts";


function isWGSLFunc(module: TWGSLModule): module is TWGSLFunc {
    return 'name' in module && module.name !== undefined;
}

function isWGSLPart(module: TWGSLModule): module is TWGSLPart {
    return !('name' in module) || module.name === undefined;
}

function getSortedDependencies(dependencies: TWGSLModule[]): TWGSLModule[] {
    const partDeps = uniqBy(dependencies.filter(isWGSLPart), ({body}) => body);
    const unsortedDeps = uniqBy(dependencies.filter(isWGSLFunc), ({name}) => name);
    const sortedDeps: TWGSLFunc[] = [];

    if (unsortedDeps.length > 0) {
        const processedDeps = new Set<string>();

        while (unsortedDeps.length > processedDeps.size) {
            const currentLength = processedDeps.size;

            for (const dependency of unsortedDeps) {
                if (processedDeps.has(dependency.name)) {
                    continue;
                }

                const namedDeps = dependency.deps.filter(isWGSLFunc).map(({name}) => name);

                if (namedDeps.length === 0 || namedDeps.every((name) => processedDeps.has(name))) {
                    processedDeps.add(dependency.name);
                    sortedDeps.push(dependency);
                }
            }

            if (currentLength === processedDeps.size) {
                throw new Error(`Can't resolve dependencies`);
            }
        }
    }

    return [...partDeps, ...sortedDeps];
}

export function setupVariable<M extends Record<string, VariableMeta>>(map: M, group = 0, binding = 0): Record<keyof M, VariableMeta> {
    for (const k in map) {
        map[k].group = group;
        map[k].binding = binding++;
    }

    return map as Record<keyof M, VariableMeta>;
}

export function buildShader<U extends Record<string, VariableMeta>, A extends Record<string, VariableMeta>>(
    uniforms: U,
    attributes: A,
    module: TWGSLModule,
) {
    const uniformKeys = uniforms ? Object.keys(uniforms) : [];
    const uniformsPart = uniformKeys.reduce((acc, key) => {
        const u = uniforms![key];
        return `
            ${acc}

            @group(${u.group})
            @binding(${u.binding})
            var<${u.kind}> ${u.name}: ${u.type};
        `;
    }, '');
    const attributesKeys = attributes ? Object.keys(attributes) : [];
    const attributesPart = attributesKeys.reduce((acc, key) => {
        return `
            ${acc}

            ${attributes![key].name}: ${attributes![key].type};
        `;
    }, '');
    const body =
        getSortedDependencies(module.deps)
            .map((item) => item.body)
            .join('') + module.body;

    return `
        ${uniformsPart}
        ${attributesPart}
        ${body}
    `;
}

const vertexRegExp = /@vertex\s*(?!\n)\s+fn\s+([a-z0-9_]+)/mg;

export function extractVertexName(shader: string): string {
    const match = shader.matchAll(vertexRegExp).next().value;

    if (isNil(match) || match[1] === undefined) {
        throw new Error('Vertex function not found');
    }

    return match[1];
}

const fragmentRegExp = /@fragment\s*(?!\n)\s+fn\s+([a-z0-9_]+)/mg;

export function extractFragmentName(shader: string): string {
    const match = shader.matchAll(fragmentRegExp).next().value;

    if (isNil(match) || match[1] === undefined) {
        throw new Error('Fragment function not found');
    }

    return match[1];
}