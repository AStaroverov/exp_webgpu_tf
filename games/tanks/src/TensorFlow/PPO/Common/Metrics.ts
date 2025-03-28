import * as tfvis from '@tensorflow/tfjs-vis';
import { getAgentLog, setAgentLog } from '../APPO_v1/Database.ts';
import { downsample } from 'downsample-lttb-ts';

class CompressedBuffer {
    buffer: [x: number, y: number][];

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
                this.buffer.push([1, item]);
            } else {
                const last = this.buffer[this.buffer.length - 1];
                this.buffer.push([last[0] + 1, item]);
            }

            if (this.buffer.length >= this.size) {
                this.compress();
            }
        }
    }

    compress() {
        this.buffer = downsample({
            series: this.buffer,
            threshold: this.size / this.compressBatch,
        }) as [number, number][];
    }

    toArray(): RenderPoint[] {
        return this.buffer.flatMap((b) => [{
            x: b[0],
            y: b[1],
        }]);
    }

    toJson() {
        return this.buffer;
    }

    fromJson(data: [number, number][]) {
        this.buffer = data;
    }
}

type RenderPoint = { x: number, y: number };

const store = {
    rewards: new CompressedBuffer(10_000, 10),
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

    drawMetrics();
});

export function drawMetrics() {
    drawRewards();
    drawBatch();
    drawEpoch();
    drawTrain();
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

function calculateMovingAverage(data: RenderPoint[], windowSize: number): RenderPoint[] {
    const averaged: RenderPoint[] = [];
    for (let i = 0; i < data.length; i++) {
        const win = data.slice(Math.max(i - windowSize + 1, 0), i + 1);
        const avg = win.reduce((sum, v) => sum + v.y, 0) / win.length;
        averaged.push({ x: data[i].x, y: avg });
    }
    return averaged;
}

function drawEpoch() {
    tfvis.render.linechart({ name: 'KL', tab: 'Training' }, {
        values: [store.kl.toArray(), calculateMovingAverage(store.kl.toArray(), 100)],
        series: ['Avg', 'MA'],
    }, {
        xLabel: 'Version',
        yLabel: 'KL',
        width: 500,
        height: 300,
    });

    tfvis.render.linechart({ name: 'Policy Loss', tab: 'Loss' }, {
        values: [store.policyLoss.toArray()],
    }, {
        xLabel: 'Version',
        yLabel: 'Loss',
        width: 500,
        height: 300,
    });

    tfvis.render.linechart({ name: 'Value Loss', tab: 'Loss' }, {
        values: [store.valueLoss.toArray()],
    }, {
        xLabel: 'Version',
        yLabel: 'Loss',
        width: 500,
        height: 300,
    });
}

function drawRewards() {
    const avgRewards = store.rewards.toArray();
    const avgRewardsMA = calculateMovingAverage(avgRewards, 1000);
    tfvis.render.scatterplot({ name: 'Reward', tab: 'Training' }, {
        values: [avgRewards, avgRewardsMA],
        series: ['Avg', 'MA'],
    }, {
        xLabel: 'Version',
        yLabel: 'Reward',
        width: 500,
        height: 300,
    });
}

function drawBatch() {
    const avgVersionDelta = store.versionDelta.toArray();
    const avgVersionDeltaMA = calculateMovingAverage(avgVersionDelta, 100);
    tfvis.render.scatterplot({ name: 'Batch Version Delta', tab: 'Training' }, {
        values: [avgVersionDelta, avgVersionDeltaMA],
        series: ['Avg', 'MA'],
    }, {
        xLabel: 'Batch',
        yLabel: 'Version Delta',
        width: 500,
        height: 300,
    });

    tfvis.render.linechart({ name: 'Batch Size', tab: 'Training' }, {
        values: store.batchSize.toArray(),
    }, {
        xLabel: 'Version',
        yLabel: 'Batch Size',
        width: 500,
        height: 300,
    });
}

function drawTrain() {
    const avgTrainTime = store.trainTime.toArray();
    const avgTrainTimeMA = calculateMovingAverage(avgTrainTime, 100);
    tfvis.render.linechart({ name: 'Train Time', tab: 'Training' }, {
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
    tfvis.render.linechart({ name: 'Waiting Time', tab: 'Training' }, {
        values: [avgWaitTime, avgWaitTimeMA],
        series: ['Avg', 'MA'],
    }, {
        xLabel: 'Version',
        yLabel: 'Waiting Time',
        width: 500,
        height: 300,
    });
}