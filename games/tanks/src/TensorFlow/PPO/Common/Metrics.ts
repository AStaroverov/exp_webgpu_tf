import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getAgentLog, setAgentLog } from '../APPO_v1/Database.ts';

type CompressedBatch = {
    start: number;
    end: number;
    min: number;
    max: number;
    avg: number;
};

class CompressedBuffer {
    buffer: CompressedBatch[];

    constructor(
        private size: number,
        private compressBatch: number,
    ) {
        this.buffer = [];
    }

    add(...data: number[]) {
        for (let i = 0; i < data.length; i++) {
            const item = data[i];

            if (this.buffer.length === 0) {
                this.buffer.push({ start: 0, end: 0, min: item, max: item, avg: item });
            } else {
                const last = this.buffer[this.buffer.length - 1];
                this.buffer.push({ start: last.end + 1, end: last.end + 1, min: item, max: item, avg: item });
            }

            if (this.buffer.length === this.size) {
                this.compress();
            }
        }
    }

    compress() {
        const compressed: CompressedBatch[] = [];
        for (let i = 0; i < this.buffer.length; i += this.compressBatch) {
            const compressedBatch = this.buffer[i];
            let { min, max, start, end } = this.buffer[i];
            let allAvg: number[] = [];
            for (let j = i; j < i + this.compressBatch; j++) {
                const batch = this.buffer[j];
                start = Math.min(start, batch.start);
                end = Math.max(end, batch.end);
                min = Math.min(min, batch.min);
                max = Math.max(min, batch.min);
                allAvg.push(batch.avg);
            }
            compressedBatch.start = start;
            compressedBatch.end = end;
            compressedBatch.min = min;
            compressedBatch.max = max;
            compressedBatch.avg = allAvg.reduce((sum, v) => sum + v, 0) / allAvg.length;

            compressed.push(compressedBatch);
        }
        this.buffer = compressed;
    }

    toArrayMin(): RenderPoint[] {
        return this.buffer.flatMap(b => b.start === b.end ? [] : [{ x: (b.start + b.end) / 2, y: b.min }]);
    }

    toArrayMax(): RenderPoint[] {
        return this.buffer.flatMap(b => b.start === b.end ? [] : [{ x: (b.start + b.end) / 2, y: b.max }]);
    }

    toArrayAvg(): RenderPoint[] {
        return this.buffer.flatMap(b => [{ x: (b.start + b.end) / 2, y: b.avg }]);
    }

    toJson() {
        return this.buffer;
    }

    fromJson(data: CompressedBatch[]) {
        this.buffer = data;
    }
}

type RenderPoint = { x: number, y: number };

const store = {
    rewards: new CompressedBuffer(10_000, 100),
    kl: new CompressedBuffer(10_000, 10),
    valueLoss: new CompressedBuffer(1_000, 10),
    policyLoss: new CompressedBuffer(1_000, 10),
    trainTime: new CompressedBuffer(1_000, 10),
    waitTime: new CompressedBuffer(1_000, 10),
    versionDelta: new CompressedBuffer(1_000, 10),
    batchSize: new CompressedBuffer(1_000, 10),
};

getAgentLog().then((data) => {
    // @ts-ignore
    if (data.version === 1) {
        // @ts-ignore
        store.rewards.fromJson(data.rewards as CompressedBatch[]);
        // @ts-ignore
        store.kl.fromJson(data.kl as CompressedBatch[]);
        // @ts-ignore
        store.valueLoss.fromJson(data.valueLoss as CompressedBatch[]);
        // @ts-ignore
        store.policyLoss.fromJson(data.policyLoss as CompressedBatch[]);
        // @ts-ignore
        store.trainTime.fromJson(data.trainTime as CompressedBatch[]);
        // @ts-ignore
        store.waitTime.fromJson(data.waitTime as CompressedBatch[]);
        // @ts-ignore
        store.versionDelta.fromJson(data.versionDelta as CompressedBatch[]);
        // @ts-ignore
        store.batchSize.fromJson(data.batchSize as CompressedBatch[]);
    }

    drawRewards();
    drawBatch();
    drawEpoch();
    drawTrain();
});

function drawEpoch() {
    tfvis.render.linechart({ name: 'KL', tab: 'Training' }, {
        values: [store.kl.toArrayMin(), store.kl.toArrayMax(), store.kl.toArrayAvg()],
        series: ['Min', 'Max', 'Avg'],
    }, {
        xLabel: 'Version',
        yLabel: 'KL',
        width: 500,
        height: 300,
    });

    tfvis.render.linechart({ name: 'Policy Loss', tab: 'Loss' }, {
        values: [store.policyLoss.toArrayAvg()],
    }, {
        xLabel: 'Version',
        yLabel: 'Loss',
        width: 500,
        height: 300,
    });

    tfvis.render.linechart({ name: 'Value Loss', tab: 'Loss' }, {
        values: [store.valueLoss.toArrayAvg()],
    }, {
        xLabel: 'Version',
        yLabel: 'Loss',
        width: 500,
        height: 300,
    });

    tf.nextFrame();
}

function drawRewards() {
    const avgRewards = store.rewards.toArrayAvg();
    const avgRewardsMA = calculateMovingAverage(avgRewards, 10);
    tfvis.render.scatterplot({ name: 'Reward', tab: 'Training' }, {
        values: [store.rewards.toArrayMin(), store.rewards.toArrayMax(), avgRewards, avgRewardsMA],
        series: ['Min', 'Max', 'Avg', 'MA'],
    }, {
        xLabel: 'Version',
        yLabel: 'Reward',
        width: 500,
        height: 300,
    });
    tf.nextFrame();
}

function drawBatch() {
    tfvis.render.linechart({ name: 'Batch Version Delta', tab: 'Training' }, {
        values: [store.versionDelta.toArrayMin(), store.versionDelta.toArrayMax(), store.versionDelta.toArrayAvg()],
        series: ['Avg', 'MA'],
    }, {
        xLabel: 'Batch',
        yLabel: 'Version Delta',
        width: 500,
        height: 300,
    });

    tfvis.render.linechart({ name: 'Batch Size', tab: 'Training' }, {
        values: store.batchSize.toArrayAvg(),
    }, {
        xLabel: 'Version',
        yLabel: 'Batch Size',
        width: 500,
        height: 300,
    });

    tf.nextFrame();
}

function drawTrain() {
    const avgTrainTime = store.waitTime.toArrayAvg();
    const avgTrainTimeMA = calculateMovingAverage(avgTrainTime, 10);
    tfvis.render.linechart({ name: 'Train Time', tab: 'Training' }, {
        values: [avgTrainTime, avgTrainTimeMA],
        series: ['Avg', 'MA'],
    }, {
        xLabel: 'Version',
        yLabel: 'Train Time',
        width: 500,
        height: 300,
    });

    const avgWaitTime = store.waitTime.toArrayAvg();
    const avgWaitTimeMA = calculateMovingAverage(avgWaitTime, 10);
    tfvis.render.linechart({ name: 'Waiting Time', tab: 'Training' }, {
        values: [avgWaitTime, avgWaitTimeMA],
        series: ['Avg', 'MA'],
    }, {
        xLabel: 'Version',
        yLabel: 'Waiting Time',
        width: 500,
        height: 300,
    });
    tf.nextFrame();
}

export function saveMetrics() {
    setAgentLog({
        version: 1,
        rewards: store.rewards.toJson(),
        kl: store.kl.toJson(),
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
    drawEpoch();
}

export function logRewards(rewards: number[]) {
    store.rewards.add(...rewards);
    drawRewards();
}

export function logBatch(data: { versionDelta: number, batchSize: number }) {
    store.versionDelta.add(data.versionDelta);
    store.batchSize.add(data.batchSize);
    drawBatch();
}

export function logTrain(data: { trainTime: number, waitTime: number }) {
    store.trainTime.add(data.trainTime);
    store.waitTime.add(data.waitTime);
    drawTrain();
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