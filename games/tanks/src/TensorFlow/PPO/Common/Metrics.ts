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
            let min = this.buffer[i].min;
            let max = this.buffer[i].max;
            let start = Infinity, end = -Infinity;
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
    kl: new CompressedBuffer(1_000, 10),
    valueLoss: new CompressedBuffer(1_000, 10),
    policyLoss: new CompressedBuffer(1_000, 10),
    trainTime: new CompressedBuffer(100, 10),
    waitTime: new CompressedBuffer(100, 10),
    versionDelta: new CompressedBuffer(100, 10),
    batchSize: new CompressedBuffer(100, 10),
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
    const avgKL = store.kl.toArrayAvg();
    const avgKLMA = calculateMovingAverage(avgKL, 10);
    tfvis.render.scatterplot({ name: 'KL', tab: 'Training' }, {
        values: [store.kl.toArrayMin(), store.kl.toArrayMax(), avgKL, avgKLMA],
        series: ['Min', 'Max', 'Avg', 'MA'],
    }, {
        xLabel: 'Version',
        yLabel: 'KL',
        width: 500,
        height: 300,
    });

    const avgPolicyLoss = store.policyLoss.toArrayAvg();
    const avgPolicyLossMA = calculateMovingAverage(avgPolicyLoss, 10);
    tfvis.render.scatterplot({ name: 'Policy Loss', tab: 'Loss' }, {
        values: [avgPolicyLoss, avgPolicyLossMA],
        series: ['Avg', 'MA'],
    }, {
        xLabel: 'Version',
        yLabel: 'Loss',
        width: 500,
        height: 300,
    });


    const avgValueLoss = store.valueLoss.toArrayAvg();
    const avgValueLossMA = calculateMovingAverage(avgValueLoss, 10);
    tfvis.render.scatterplot({ name: 'Value Loss', tab: 'Loss' }, {
        values: [avgPolicyLoss, avgValueLossMA],
        series: ['Avg', 'MA'],
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
    const avgBatchDelta = store.versionDelta.toArrayAvg();
    const avgBatchDeltaMA = calculateMovingAverage(avgBatchDelta, 10);
    tfvis.render.scatterplot({ name: 'Batch Version Delta', tab: 'Training' }, {
        values: [avgBatchDelta, avgBatchDeltaMA],
        series: ['Avg', 'MA'],
    }, {
        xLabel: 'Batch',
        yLabel: 'Version Delta',
        width: 500,
        height: 300,
    });

    const avgBatchSize = store.batchSize.toArrayAvg();
    const avgBatchSizeMA = calculateMovingAverage(avgBatchSize, 10);
    tfvis.render.scatterplot({ name: 'Batch Size', tab: 'Training' }, {
        values: [avgBatchSize, avgBatchSizeMA],
        series: ['Avg', 'MA'],
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
    tfvis.render.scatterplot({ name: 'Train Time', tab: 'Training' }, {
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
    tfvis.render.scatterplot({ name: 'Waiting Time', tab: 'Training' }, {
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