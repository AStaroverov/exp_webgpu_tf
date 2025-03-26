import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { RingBuffer } from 'ring-buffer-ts';
import { getAgentLog, setAgentLog } from '../APPO_v1/Database.ts';

const rewardBuffer = new RingBuffer<number>(100_000);
const epochBuffer = new RingBuffer<{
    version: number;
    kl: number;
    valueLoss: number;
    policyLoss: number;
    avgBatchSize: number;
}>(10_000);

getAgentLog().then((data) => {
    epochBuffer.fromArray(data?.epoch ?? [] as any);
    rewardBuffer.fromArray(data?.rewards ?? [] as any);
    drawRewards(rewardBuffer.toArray());
    drawEpoch(epochBuffer.toArray());
});

function drawEpoch(data: {
    version: number,
    kl: number
    valueLoss: number,
    policyLoss: number,
    avgBatchSize: number,
}[]) {
    const klList = data.map((d, i) => ({ x: i, y: d.kl }));
    tfvis.render.linechart({ name: 'KL', tab: 'Training' }, { values: klList }, {
        xLabel: 'Version',
        yLabel: 'KL',
        width: 500,
        height: 300,
    });

    const policyLossList = data.map((d, i) => ({ x: i, y: d.policyLoss }));
    tfvis.render.linechart({ name: 'Policy Loss', tab: 'Loss' }, { values: policyLossList }, {
        xLabel: 'Version',
        yLabel: 'Loss',
        width: 500,
        height: 300,
    });

    const valueLossList = data.map((d, i) => ({ x: i, y: d.valueLoss }));
    tfvis.render.linechart({ name: 'Value Loss', tab: 'Loss' }, { values: valueLossList }, {
        xLabel: 'Version',
        yLabel: 'Loss',
        width: 500,
        height: 300,
    });

    const avgBatchSizeList = data.map((d, i) => ({ x: i, y: d.avgBatchSize }));
    tfvis.render.linechart({ name: 'Batch Size', tab: 'Training' }, { values: avgBatchSizeList }, {
        xLabel: 'Version',
        yLabel: 'Batch Size',
        width: 500,
        height: 300,
    });
    tf.nextFrame();
}

function calculateMovingAverage(data: number[], windowSize: number): number[] {
    const averaged: number[] = [];
    for (let i = 0; i < data.length; i++) {
        const win = data.slice(Math.max(i - windowSize + 1, 0), i + 1);
        const avg = win.reduce((sum, v) => sum + v, 0) / win.length;
        averaged.push(avg);
    }
    return averaged;
}

function drawRewards(data: number[]) {
    const rewardList = data.map((v, i) => ({ x: i, y: v }));
    const rewardListMA = calculateMovingAverage(data, 100).map((v, i) => ({ x: i, y: v }));
    tfvis.render.scatterplot({ name: 'Reward', tab: 'Training' }, {
        values: [rewardList, rewardListMA],
        series: ['Reward', 'Moving Average'],
    }, {
        xLabel: 'Version',
        yLabel: 'Reward',
        width: 500,
        height: 300,
    });
    tf.nextFrame();
}

function saveAgentLog() {
    setAgentLog({ epoch: epochBuffer.getLastN(1_000), rewards: rewardBuffer.getLastN(10_000) });
}

let dirtyIndex = 0;

function trySave() {
    dirtyIndex++;
    if (dirtyIndex > 1) {
        dirtyIndex = 0;
        saveAgentLog();
    }
}

export function logEpoch(data: {
    version: number
    kl: number;
    valueLoss: number;
    policyLoss: number;
    avgBatchSize: number;
}) {
    epochBuffer.add(data);
    drawEpoch(epochBuffer.toArray());
    trySave();
}

export function logRewards(rewards: number[]) {
    rewardBuffer.add(...rewards);
    drawRewards(rewardBuffer.toArray());
    trySave();
}



