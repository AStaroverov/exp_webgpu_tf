import * as tfvis from '@tensorflow/tfjs-vis';
import { getAgentLog, setAgentLog } from '../PPO/Database.ts';
import { isObject } from 'lodash-es';

type Point = { x: number, y: number };

function getLastX(...points: Point[]): number {
    return Math.max(...points.map((p) => p.x));
}

class CompressedBuffer {
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

    toArrayMin(): RenderPoint[] {
        return this.minBuffer.map((p, i) => ({ x: i, y: p.y }));
    }

    toArrayMax(): RenderPoint[] {
        return this.maxBuffer.map((p, i) => ({ x: i, y: p.y }));
    }

    toArray(): RenderPoint[] {
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
            this.minBuffer = this.compressAvg(this.minBuffer, 2);
            this.maxBuffer = this.compressAvg(this.maxBuffer, 2);
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
        const compressed: Point[] = [];
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
            compressed.push({ x: (lastItem.x + firstItem.x) / 2, y: sum / count });
        }

        return compressed;
    }

}

type RenderPoint = { x: number, y: number };

const store = {
    rewards: new CompressedBuffer(10_000, 5),
    kl: new CompressedBuffer(10_000, 5),
    lr: new CompressedBuffer(1_000, 5),
    clip: new CompressedBuffer(1_000, 5),
    valueLoss: new CompressedBuffer(1_000, 5),
    policyLoss: new CompressedBuffer(1_000, 5),
    trainTime: new CompressedBuffer(1_000, 5),
    waitTime: new CompressedBuffer(1_000, 5),
    versionDelta: new CompressedBuffer(1_000, 5),
    batchSize: new CompressedBuffer(1_000, 5),
    memory: new CompressedBuffer(100, 5),
};

export function loadMetrics() {
    return getAgentLog().then((data) => {
        // @ts-ignore
        if (isObject(data) && data.version === 1) {
            // @ts-ignore
            store.rewards.fromJson(data.rewards);
            // @ts-ignore
            store.kl.fromJson(data.kl);
            // @ts-ignore
            store.lr.fromJson(data.lr);
            // @ts-ignore
            store.clip.fromJson(data.clip);
            // @ts-ignore
            store.valueLoss.fromJson(data.valueLoss);
            // @ts-ignore
            store.policyLoss.fromJson(data.policyLoss);
            // @ts-ignore
            store.trainTime.fromJson(data.trainTime);
            // @ts-ignore
            store.waitTime.fromJson(data.waitTime);
            // @ts-ignore
            store.versionDelta.fromJson(data.versionDelta);
            // @ts-ignore
            store.batchSize.fromJson(data.batchSize);
            // @ts-ignore
            store.memory.fromJson(data.memory);
        }
    });
}

export function drawMetrics() {
    drawTab1();
    drawTab2();
    drawTab3();
}

export function saveMetrics() {
    setAgentLog({
        version: 1,
        rewards: store.rewards.toJson(),
        kl: store.kl.toJson(),
        lr: store.lr.toJson(),
        clip: store.clip.toJson(),
        valueLoss: store.valueLoss.toJson(),
        policyLoss: store.policyLoss.toJson(),
        trainTime: store.trainTime.toJson(),
        waitTime: store.waitTime.toJson(),
        versionDelta: store.versionDelta.toJson(),
        batchSize: store.batchSize.toJson(),
    });
}

export function logEpoch(data: {
    kl: number;
    valueLoss: number;
    policyLoss: number;
}) {
    store.kl.add(data.kl);
    store.valueLoss.add(data.valueLoss);
    store.policyLoss.add(data.policyLoss);
}

export function logLR(lr: number) {
    store.lr.add(lr);
}

export function logClip(clip: number) {
    store.clip.add(clip);
}

export function logRewards(rewards: number[]) {
    store.rewards.add(...rewards);
}

export function logBatch(data: { versionDelta: number, batchSize: number }) {
    store.versionDelta.add(data.versionDelta);
    store.batchSize.add(data.batchSize);
}

export function logTrain(data: { trainTime: number, waitTime: number }) {
    store.trainTime.add(data.trainTime);
    store.waitTime.add(data.waitTime);
}

function drawTab1() {
    const avgRewards = store.rewards.toArray();
    const avgRewardsMA = calculateMovingAverage(avgRewards, 1000);
    const minRewards = store.rewards.toArrayMin();
    const minRewardsMA = calculateMovingAverage(minRewards, 1000);
    const maxRewards = store.rewards.toArrayMax();
    const maxRewardsMA = calculateMovingAverage(maxRewards, 1000);
    tfvis.render.linechart({ name: 'Reward', tab: 'Tab 1' }, {
        values: [minRewards, maxRewards, avgRewards, minRewardsMA, maxRewardsMA, avgRewardsMA],
        series: ['Min', 'Max', 'Avg', 'Min MA', 'Max MA', 'Avg MA'],
    }, {
        xLabel: 'Version',
        yLabel: 'Reward',
        width: 500,
        height: 300,
    });

    tfvis.render.linechart({ name: 'KL', tab: 'Tab 1' }, {
        values: [store.kl.toArray(), calculateMovingAverage(store.kl.toArray(), 10)],
        series: ['Avg', 'MA'],
    }, {
        xLabel: 'Version',
        yLabel: 'KL',
        width: 500,
        height: 300,
    });

    tfvis.render.linechart({ name: 'LR', tab: 'Tab 1' }, {
        values: [store.lr.toArray()],
    }, {
        xLabel: 'Version',
        yLabel: 'LR',
        width: 500,
        height: 300,
    });

    tfvis.render.linechart({ name: 'Clip', tab: 'Tab 1' }, {
        values: [store.clip.toArray()],
    }, {
        xLabel: 'Version',
        yLabel: 'Clip',
        width: 500,
        height: 300,
    });
}

function drawTab2() {
    tfvis.render.linechart({ name: 'Policy Loss', tab: 'Tab 2' }, {
        values: [store.policyLoss.toArray()],
    }, {
        xLabel: 'Version',
        yLabel: 'Loss',
        width: 500,
        height: 300,
    });

    tfvis.render.linechart({ name: 'Value Loss', tab: 'Tab 2' }, {
        values: [store.valueLoss.toArray()],
    }, {
        xLabel: 'Version',
        yLabel: 'Loss',
        width: 500,
        height: 300,
    });
}


function drawTab3() {
    const avgTrainTime = store.trainTime.toArray();
    const avgTrainTimeMA = calculateMovingAverage(avgTrainTime, 100);
    tfvis.render.linechart({ name: 'Train Time', tab: 'Tab 3' }, {
        values: [avgTrainTime, avgTrainTimeMA],
        series: ['Avg', 'MA'],
    }, {
        xLabel: 'Version',
        yLabel: 'Train Time',
        width: 500,
        height: 300,
    });

    const avgWaitTime = store.waitTime.toArray();
    const avgWaitTimeMA = calculateMovingAverage(avgWaitTime, 100);
    tfvis.render.linechart({ name: 'Waiting Time', tab: 'Tab 3' }, {
        values: [avgWaitTime, avgWaitTimeMA],
        series: ['Avg', 'MA'],
    }, {
        xLabel: 'Version',
        yLabel: 'Waiting Time',
        width: 500,
        height: 300,
    });
    const avgVersionDelta = store.versionDelta.toArray();
    const avgVersionDeltaMA = calculateMovingAverage(avgVersionDelta, 100);
    tfvis.render.scatterplot({ name: 'Batch Version Delta', tab: 'Tab 3' }, {
        values: [avgVersionDelta, avgVersionDeltaMA],
        series: ['Avg', 'MA'],
    }, {
        xLabel: 'Batch',
        yLabel: 'Version Delta',
        width: 500,
        height: 300,
    });

    tfvis.render.linechart({ name: 'Batch Size', tab: 'Tab 3' }, {
        values: store.batchSize.toArray(),
    }, {
        xLabel: 'Version',
        yLabel: 'Batch Size',
        width: 500,
        height: 300,
    });
}

function calculateMovingAverage(data: RenderPoint[], windowSize: number): RenderPoint[] {
    const averaged: RenderPoint[] = [];
    for (let i = 0; i < data.length; i++) {
        const win = data.slice(Math.max(i - windowSize + 1, 0), i + 1);
        const avg = win.reduce((sum, v) => sum + v.y, 0) / win.length;
        averaged.push({ x: data[i].x, y: avg });
    }
    return averaged;
}