import { RingBuffer as Base } from 'ring-buffer-ts';

type Point = { x: number, y: number };

export class RingBuffer {
    private buffer: Base<Point>;

    constructor(
        private size: number,
    ) {
        this.buffer = new Base<Point>(this.size);
    }

    add(...data: number[]) {
        this.addList(data);
    }

    addList(data: number[]) {
        for (let i = 0; i < data.length; i++) {
            const item = data[i];
            const last = this.buffer.getLast();
            const lastX = last !== undefined ? last.x : 0;
            this.buffer.add({ x: lastX + 1, y: item });
        }
    }

    toArray(): Point[] {
        return this.buffer.toArray().map((p, i) => ({ x: i, y: p.y }))
    }

    toJson() {
        return {
            buffer: this.buffer.toArray()
        };
    }

    fromJson(data: unknown) {
        if (typeof data === 'object' && data !== null) {
            if ('buffer' in data && data.buffer instanceof Array) {
                this.buffer = Base.fromArray<Point>(data.buffer, this.size);
            }
        }
    }
}