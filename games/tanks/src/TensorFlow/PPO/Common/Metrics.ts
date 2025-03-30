import * as tfvis from '@tensorflow/tfjs-vis';
import { getAgentLog, setAgentLog } from '../APPO_v1/Database.ts';

type Point = { x: number, y: number };
type CompressedBatch = {
    min: Point;
    max: Point;
    avg: Point;
    compressed: boolean;
};

function getLastX(batch: CompressedBatch): number {
    return Math.max(batch.min.x, batch.max.x, batch.avg.x);
}

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
                this.buffer.push({
                    min: { x: 0, y: item },
                    max: { x: 0, y: item },
                    avg: { x: 0, y: item },

                    compressed: false,
                });
            } else {
                const last = this.buffer[this.buffer.length - 1];
                const lastY = getLastX(last);
                this.buffer.push({
                    min: { x: lastY + 1, y: item },
                    max: { x: lastY + 1, y: item },
                    avg: { x: lastY + 1, y: item },

                    compressed: false,
                });
            }
        }

        if (this.buffer.length >= this.size) {
            this.compress();
        }
    }

    compress() {
        this.compressUncompressed();
        this.compressCompressed();
    }

    toArrayMin(): RenderPoint[] {
        return this.buffer.flatMap((b) => !b.compressed ? [] : [b.min]).map((v, i) => {
            return { x: i, y: v.y };
        });
    }

    toArrayMax(): RenderPoint[] {
        return this.buffer.flatMap((b) => !b.compressed ? [] : [b.max]).map((v, i) => {
            return { x: i, y: v.y };
        });
    }

    toArrayAvg(): RenderPoint[] {
        return this.buffer.flatMap((b) => [b.avg]).map((v, i) => {
            return { x: i, y: v.y };
        });
    }

    toJson() {
        return this.buffer;
    }

    fromJson(data: CompressedBatch[]) {
        this.buffer = data;
    }

    private compressUncompressed() {
        const startIndex = this.buffer.findIndex((b) => !b.compressed);
        this.compressing(startIndex, this.compressBatch);
    }

    private compressCompressed() {
        this.compressing(0, 2);
    }

    private compressing(startIndex: number, batch: number) {
        const compressed: CompressedBatch[] = [];
        const buffer = this.buffer;
        const length = buffer.length;

        for (let i = startIndex; i < length; i = Math.min(i + batch, length)) {
            const reusedItem = buffer[i];
            let { min, max, avg } = reusedItem;
            let allAvg: Point[] = [avg];
            for (let j = i + 1; j < Math.min(i + batch, length); j++) {
                const item = buffer[j];
                if (min.y > item.min.y) {
                    min = item.min;
                }
                if (max.y < item.max.y) {
                    max = item.max;
                }
                allAvg.push(item.avg);
            }
            let tmpAvg = allAvg.reduce((sum, v) => {
                sum.x += v.x;
                sum.y += v.y;
                return sum;
            }, { x: 0, y: 0 });
            tmpAvg.x /= allAvg.length;
            tmpAvg.y /= allAvg.length;

            reusedItem.min = min;
            reusedItem.max = max;
            reusedItem.avg = tmpAvg;
            reusedItem.compressed = true;

            compressed.push(reusedItem);
        }

        this.buffer = this.buffer.slice(0, startIndex).concat(compressed);
    }
}

type RenderPoint = { x: number, y: number };

const store = {
    rewards: new CompressedBuffer(10_000, 5),
    kl: new CompressedBuffer(10_000, 5),
    valueLoss: new CompressedBuffer(1_000, 5),
    policyLoss: new CompressedBuffer(1_000, 5),
    trainTime: new CompressedBuffer(1_000, 5),
    waitTime: new CompressedBuffer(1_000, 5),
    versionDelta: new CompressedBuffer(1_000, 5),
    batchSize: new CompressedBuffer(1_000, 5),
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
        values: [store.kl.toArrayAvg(), calculateMovingAverage(store.kl.toArrayAvg(), 10)],
        series: ['Avg', 'MA'],
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
}

function drawRewards() {
    const avgRewards = store.rewards.toArrayAvg();
    const avgRewardsMA = calculateMovingAverage(avgRewards, 1000);
    tfvis.render.scatterplot({ name: 'Reward', tab: 'Training' }, {
        values: [store.rewards.toArrayMin(), store.rewards.toArrayMax(), avgRewards, avgRewardsMA],
        series: ['Min', 'Max', 'Avg', 'MA'],
    }, {
        xLabel: 'Version',
        yLabel: 'Reward',
        width: 500,
        height: 300,
    });
}

function drawBatch() {
    const avgVersionDelta = store.versionDelta.toArrayAvg();
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
        values: store.batchSize.toArrayAvg(),
    }, {
        xLabel: 'Version',
        yLabel: 'Batch Size',
        width: 500,
        height: 300,
    });
}

function drawTrain() {
    const avgTrainTime = store.trainTime.toArrayAvg();
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

    const avgWaitTime = store.waitTime.toArrayAvg();
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