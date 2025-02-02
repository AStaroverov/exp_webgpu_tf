import {StructMeta} from "./StructMeta.ts";
import {getTypeBufferSize} from "./index.ts";

const fieldRegex = /(\w+):\s*(array<.*>|\w+)/g;

export function extractStructMeta(shader: string, name: string): StructMeta {
    const prefix = `struct ${name} {`;
    const startIndex = shader.indexOf(prefix);

    if (startIndex === -1) {
        throw new Error(`Struct ${name} not found in shader`);
    }

    const endIndex = shader.indexOf('}', startIndex);
    const structBody = shader.slice(startIndex + prefix.length, endIndex);

    const fields = [...structBody.matchAll(fieldRegex)].slice(1).reduce((acc, [, name, type]) => {
        acc[name] = {
            type,
            size: getTypeBufferSize(type)
        };
        return acc;
    }, {} as StructMeta['fields']);

    return new StructMeta(name, fields);
}
