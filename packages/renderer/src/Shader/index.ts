import { ValueOf } from '../type.ts';

type GPU_Type = string | GPU_Primitive;

enum GPU_Primitive {
    f32 = 'f32',
    u32 = 'u32',
    i32 = 'i32',
}

const GPU_PRIMITIVE_SET = new Set([GPU_Primitive.f32, GPU_Primitive.i32, GPU_Primitive.u32]);
const GPU_PRIMITIVE_SIZE = 4;

export const mapPrimitiveToConstructor = <const>{
    f32: Float32Array,
    u32: Uint32Array,
    i32: Int32Array,
};

export function getTypeConstructor(type: GPU_Type): ValueOf<typeof mapPrimitiveToConstructor> {
    if (GPU_PRIMITIVE_SET.has(type as GPU_Primitive)) {
        return mapPrimitiveToConstructor[type as GPU_Primitive];
    }

    if (type.startsWith('vec')) {
        const [, primitive] = type.match(/vec\d+<(\w+)>/)!;
        return getTypeConstructor(primitive);
    }

    if (type.startsWith('mat')) {
        const [, primitive] = type.match(/mat\dx\d<(\w+)>/)!;
        return getTypeConstructor(primitive);
    }

    if (type.startsWith('array<vec')) {
        const [, primitive] = type.match(/array<vec\d<(\w+)>,\s*\d+>/)!;
        return getTypeConstructor(primitive);
    }

    if (type.startsWith('array<mat')) {
        const [, primitive] = type.match(/array<mat\dx\d<(\w+)>,\s*\d+>/)!;
        return getTypeConstructor(primitive);
    }

    if (type.startsWith('array')) {
        const [, primitive] = type.match(/array<([<>\w]+),\s*\d+>/)!;
        return getTypeConstructor(primitive);
    }

    throw new Error(`Unknown type ${ type }`);
}

export function getTypeSize(type: GPU_Type): number {
    if (GPU_PRIMITIVE_SET.has(type as GPU_Primitive)) {
        return 1;
    }

    if (type.startsWith('vec')) {
        const [, size] = type.match(/vec(\d+)/)!;
        return Number(size);
    }

    if (type.startsWith('mat')) {
        const [, x, y] = type.match(/mat(\d)x(\d)<\w+>/)!;
        return Number(x) * Number(y);
    }

    if (type.startsWith('array<mat')) {
        const [, x, y, size] = type.match(/array<mat(\d)x(\d)<\w+>,\s*(\d+)>/)!;
        return Number(x) * Number(y) * Number(size);
    }

    if (type.startsWith('array<vec')) {
        const [, vec, size] = type.match(/array<vec(\d)<\w+>,\s*(\d+)>/)!;
        return Number(vec) * Number(size);
    }

    if (type.startsWith('array')) {
        const [, size] = type.match(/array<\w+,\s*(\d+)>/)!;
        return Number(size);
    }

    throw new Error(`Unknown type ${ type }`);
}

export function getTypeBufferSize(type: GPU_Type): number {
    if (GPU_PRIMITIVE_SET.has(type as GPU_Primitive)) {
        return GPU_PRIMITIVE_SIZE;
    }

    if (type.startsWith('vec')) {
        const [, size, innerType] = type.match(/vec(\d+)<(\w+)>/)!;
        return Number(size) * getTypeBufferSize(innerType);
    }

    if (type.startsWith('mat')) {
        const [, x, y, innerType] = type.match(/mat(\d)x(\d)<(\w+)>/)!;
        return Number(x) * Number(y) * getTypeBufferSize(innerType);
    }

    if (type.startsWith('array<vec')) {
        const [, vec, innerType, size] = type.match(/array<vec(\d)<(\w+)>,\s*(\d+)>/)!;
        return Number(vec) * Number(size) * getTypeBufferSize(innerType);
    }

    if (type.startsWith('array<mat')) {
        const [, x, y, innerType, size] = type.match(/array<mat(\d)x(\d)<(\w+)>,\s*(\d+)>/)!;
        return Number(x) * Number(y) * Number(size) * getTypeBufferSize(innerType);
    }

    if (type.startsWith('array')) {
        const [, innerType, size] = type.match(/array<(\w+),\s*(\d+)>/)!;
        return Number(size) * getTypeBufferSize(innerType);
    }

    throw new Error(`Unknown type ${ type }`);
}

export function getTypeTypedArray(type: GPU_Type): InstanceType<ValueOf<typeof mapPrimitiveToConstructor>> {
    return new (getTypeConstructor(type))(getTypeSize(type));
}
