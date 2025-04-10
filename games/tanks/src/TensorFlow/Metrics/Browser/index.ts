import * as tfvis from '@tensorflow/tfjs-vis';
import { isObject, throttle } from 'lodash-es';
import { metricsChannels } from '../../Common/channels.ts';
import { getAgentLog, setAgentLog } from './store.ts';

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

type RenderPoint = { x: number, y: number };

const store = {
    rewards: new CompressedBuffer(10_000, 5),
    values: new CompressedBuffer(10_000, 5),
    returns: new CompressedBuffer(10_000, 5),
    advantages: new CompressedBuffer(10_000, 5),
    kl: new CompressedBuffer(10_000, 5),
    lr: new CompressedBuffer(1_000, 5),
    valueLoss: new CompressedBuffer(1_000, 5),
    policyLoss: new CompressedBuffer(1_000, 5),
    trainTime: new CompressedBuffer(1_000, 5),
    waitTime: new CompressedBuffer(1_000, 5),
    versionDelta: new CompressedBuffer(1_000, 5),
    batchSize: new CompressedBuffer(1_000, 5),
};

getAgentLog().then((data) => {
    // @ts-ignore
    if (isObject(data) && data.version === 1) {
        // @ts-ignore
        store.rewards.fromJson(data.rewards);
        // @ts-ignore
        store.values.fromJson(data.values);
        // @ts-ignore
        store.returns.fromJson(data.returns);
        // @ts-ignore
        store.advantages.fromJson(data.advantages);
        // @ts-ignore
        store.kl.fromJson(data.kl);
        // @ts-ignore
        store.lr.fromJson(data.lr);
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
    }

    subscribeOnMetrics();
});

export function drawMetrics() {
    drawTab1();
    drawTab2();
    drawTab3();
}

export const saveMetrics = throttle(() => {
    setAgentLog({
        version: 1,
        kl: store.kl.toJson(),
        lr: store.lr.toJson(),
        rewards: store.rewards.toJson(),
        values: store.values.toJson(),
        returns: store.returns.toJson(),
        advantages: store.advantages.toJson(),
        valueLoss: store.valueLoss.toJson(),
        policyLoss: store.policyLoss.toJson(),
        trainTime: store.trainTime.toJson(),
        waitTime: store.waitTime.toJson(),
        versionDelta: store.versionDelta.toJson(),
        batchSize: store.batchSize.toJson(),
    });
}, 10_000);

export function subscribeOnMetrics() {
    const w = (callback: (event: MessageEvent) => void) => {
        return (event: MessageEvent) => {
            callback(event);
            saveMetrics();
        };
    };
    metricsChannels.rewards.onmessage = w((event) => store.rewards.add(...event.data));
    metricsChannels.values.onmessage = w((event) => store.values.add(...event.data));
    metricsChannels.returns.onmessage = w((event) => store.returns.add(...event.data));
    metricsChannels.advantages.onmessage = w((event) => store.advantages.add(...event.data));
    metricsChannels.kl.onmessage = w((event) => store.kl.add(event.data));
    metricsChannels.valueLoss.onmessage = w((event) => store.valueLoss.add(event.data));
    metricsChannels.policyLoss.onmessage = w((event) => store.policyLoss.add(event.data));
    metricsChannels.lr.onmessage = w((event) => store.lr.add(event.data));
    metricsChannels.versionDelta.onmessage = w((event) => store.versionDelta.add(event.data));
    metricsChannels.batchSize.onmessage = w((event) => store.batchSize.add(event.data));
    metricsChannels.trainTime.onmessage = w((event) => store.trainTime.add(event.data));
    metricsChannels.waitTime.onmessage = w((event) => store.waitTime.add(event.data));
}

function drawTab1() {
    const tab = 'Tab 1';
    const avgRewards = store.rewards.toArray();
    const avgRewardsMA = calculateMovingAverage(avgRewards, 1000);
    const minRewards = store.rewards.toArrayMin();
    const maxRewards = store.rewards.toArrayMax();
    tfvis.render.linechart({ name: 'Reward', tab }, {
        values: [minRewards, maxRewards, avgRewards, avgRewardsMA],
        series: ['Min', 'Max', 'Avg', 'Avg MA'],
    }, {
        xLabel: 'Version',
        yLabel: 'Reward',
        width: 500,
        height: 300,
    });

    tfvis.render.linechart({ name: 'KL', tab }, {
        values: [store.kl.toArray(), calculateMovingAverage(store.kl.toArray(), 50)],
        series: ['Avg', 'MA'],
    }, {
        xLabel: 'Version',
        yLabel: 'KL',
        width: 500,
        height: 300,
    });

    tfvis.render.linechart({ name: 'LR', tab }, {
        values: [store.lr.toArray()],
    }, {
        xLabel: 'Version',
        yLabel: 'LR',
        width: 500,
        height: 300,
    });

    tfvis.render.scatterplot({ name: 'Advantage', tab }, {
        values: [store.advantages.toArrayMin(), store.advantages.toArrayMax()],
    }, {
        xLabel: 'Version',
        yLabel: 'Advantage',
        width: 500,
        height: 300,
    });

    const avgValues = store.values.toArray();
    tfvis.render.scatterplot({ name: 'Value', tab }, {
        values: [
            store.values.toArrayMin(),
            store.values.toArrayMax(),
            avgValues,
            calculateMovingAverage(avgValues, 1000),
        ],
    }, {
        xLabel: 'Version',
        yLabel: 'Value',
        width: 500,
        height: 300,
    });
    const avgReturns = store.returns.toArray();
    tfvis.render.scatterplot({ name: 'Return', tab }, {
        values: [
            store.returns.toArrayMin(),
            store.returns.toArrayMax(),
            avgReturns,
            calculateMovingAverage(avgReturns, 1000),
        ],
    }, {
        xLabel: 'Version',
        yLabel: 'Return',
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