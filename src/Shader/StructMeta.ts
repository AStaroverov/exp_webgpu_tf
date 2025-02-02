export class StructMeta {
    constructor(public name: string, public fields: Record<string, {
        size: number;
        type: string;
    }>) {
    }
}