import {StructMeta} from "./StructMeta.ts";

export function getStructSize(meta: StructMeta): number {
    return Object.values(meta.fields).reduce((acc, {size}) => acc + size, 0);
}