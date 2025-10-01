type Point = { x: number, y: number };

function getLastX(...points: Point[]): number {
    return Math.max(...points.map((p) => p.x));
}

export class CompressedBuffer {
    private buffer: Point[] = [];
    private avgBuffer: Point[] = [];
    private minBuffer: Point[] = [];
    private maxBuffer: Point[] = [];

    constructor(
        private size: number,
        private compressBatch: number,
    ) {
    }

    add(...data: number[]) {
        for (let i = 0; i < data.length; i++) {
            const item = data[i];
            const last = this.buffer[this.buffer.length - 1] ?? this.avgBuffer[this.avgBuffer.length - 1];
            const lastX = last !== undefined ? getLastX(last) : 0;
            this.buffer.push({ x: lastX + 1, y: item });
        }

        if (this.buffer.length >= this.size / 2) {
            this.compress();
        }
    }

    toArrayMin(): Point[] {
        return this.minBuffer.map((p, i) => ({ x: i, y: p.y }));
    }

    toArrayMax(): Point[] {
        return this.maxBuffer.map((p, i) => ({ x: i, y: p.y }));
    }

    toArray(): Point[] {
        return this.avgBuffer.concat(this.buffer).map((p, i) => ({ x: i, y: p.y }));
    }

    toJson() {
        return {
            buffer: this.buffer,
            avgBuffer: this.avgBuffer,
            minBuffer: this.minBuffer,
            maxBuffer: this.maxBuffer,
        };
    }

    fromJson(data: unknown) {
        if (typeof data === 'object' && data !== null) {
            if ('buffer' in data && data.buffer instanceof Array) {
                this.buffer = data.buffer;
            }
            if ('avgBuffer' in data && data.avgBuffer instanceof Array) {
                this.avgBuffer = data.avgBuffer;
            }
            if ('minBuffer' in data && data.minBuffer instanceof Array) {
                this.minBuffer = data.minBuffer;
            }
            if ('maxBuffer' in data && data.maxBuffer instanceof Array) {
                this.maxBuffer = data.maxBuffer;
            }
        }
    }

    private compress() {
        if (this.avgBuffer.length > this.size / 2) {
            this.avgBuffer = this.compressAvg(this.avgBuffer, 2);
            this.minBuffer = this.compressMin(this.minBuffer, 2);
            this.maxBuffer = this.compressMax(this.maxBuffer, 2);
        }
        this.compressRawBuffer(this.compressBatch);
    }

    private compressRawBuffer(batch: number) {
        const buffer = this.buffer;
        const length = buffer.length;

        for (let i = 0; i < length; i += batch) {
            const firstItem = buffer[i];
            let lastItem = firstItem;
            let min = firstItem;
            let max = firstItem;
            let sum = firstItem.y;
            let count = 1;
            for (let j = i + 1; j < Math.min(i + batch, length); j++) {
                lastItem = buffer[j];
                if (min.y > lastItem.y) {
                    min = lastItem;
                }
                if (max.y < lastItem.y) {
                    max = lastItem;
                }
                sum += lastItem.y;
                count++;
            }
            this.avgBuffer.push({ x: (lastItem.x + firstItem.x) / 2, y: sum / count });
            this.minBuffer.push(min);
            this.maxBuffer.push(max);
        }
        this.buffer = [];
    }

    private compressAvg(buffer: Point[], batch: number) {
        const compressedAvg: Point[] = [];
        const length = buffer.length;

        for (let i = 0; i < length; i += batch) {
            const firstItem = buffer[i];
            let lastItem = firstItem;
            let sum = firstItem.y;
            let count = 1;
            for (let j = i + 1; j < Math.min(i + batch, length); j++) {
                lastItem = buffer[j];
                sum += lastItem.y;
                count++;
            }
            compressedAvg.push({ x: (lastItem.x + firstItem.x) / 2, y: sum / count });
        }

        return compressedAvg;
    }

    private compressMin(buffer: Point[], batch: number) {
        const compressedMin: Point[] = [];
        const length = buffer.length;

        for (let i = 0; i < length; i += batch) {
            let min = buffer[i];
            for (let j = i + 1; j < Math.min(i + batch, length); j++) {
                if (min.y > buffer[j].y) {
                    min = buffer[j];
                }
            }
            compressedMin.push(min);
        }

        return compressedMin;
    }

    private compressMax(buffer: Point[], batch: number) {
        const compressedMax: Point[] = [];
        const length = buffer.length;

        for (let i = 0; i < length; i += batch) {
            let max = buffer[i];
            for (let j = i + 1; j < Math.min(i + batch, length); j++) {
                if (max.y < buffer[j].y) {
                    max = buffer[j];
                }
            }
            compressedMax.push(max);
        }

        return compressedMax;
    }
}