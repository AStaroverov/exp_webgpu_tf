import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { RingBuffer } from 'ring-buffer-ts';
import { getAgentLog, setAgentLog } from '../APPO_v1/Database.ts';

const rewardBuffer = new RingBuffer<number>(100_000);
const trainBuffer = new RingBuffer<{ trainTime: number, waitTime: number }>(1_000);
const batchBuffer = new RingBuffer<{ versionDelta: number, batchSize: number; }>(1_000);
const epochBuffer = new RingBuffer<{
    kl: number;
    valueLoss: number;
    policyLoss: number;
}>(10_000);

getAgentLog().then((data) => {
    epochBuffer.fromArray(data?.epoch ?? [] as any);
    rewardBuffer.fromArray(data?.rewards ?? [] as any);
    trainBuffer.fromArray(data?.train ?? [] as any);
    batchBuffer.fromArray(data?.batch ?? [] as any);
    drawRewards(rewardBuffer.toArray());
    drawEpoch(epochBuffer.toArray());
    drawBatch(batchBuffer.toArray());
    drawTrain(trainBuffer.toArray());
});

function drawEpoch(data: {
    kl: number
    valueLoss: number,
    policyLoss: number,
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

function drawBatch(data: { versionDelta: number, batchSize: number }[]) {
    const versionDeltaList = data.map((v, i) => ({ x: i, y: v.versionDelta }));
    const versionDeltaListMA = calculateMovingAverage(data.map(v => v.versionDelta), 100).map((v, i) => ({
        x: i,
        y: v,
    }));
    tfvis.render.scatterplot({ name: 'Batch Version Delta', tab: 'Training' }, {
        values: [versionDeltaList, versionDeltaListMA],
        series: ['Version Delta', 'Moving Average'],
    }, {
        xLabel: 'Batch',
        yLabel: 'Version Delta',
        width: 500,
        height: 300,
    });

    tfvis.render.linechart({ name: 'Batch Size', tab: 'Training' }, {
        values: data.map((d, i) => ({
            x: i,
            y: d.batchSize,
        })),
    }, {
        xLabel: 'Version',
        yLabel: 'Batch Size',
        width: 500,
        height: 300,
    });

    tf.nextFrame();
}

function drawTrain(data: { trainTime: number, waitTime: number }[]) {
    const trainTimeList = data.map((v, i) => ({ x: i, y: v.trainTime }));
    const waitingTimeList = data.map((v, i) => ({ x: i, y: v.waitTime }));
    tfvis.render.linechart({ name: 'Train Time', tab: 'Training' }, { values: trainTimeList }, {
        xLabel: 'Version',
        yLabel: 'Train Time',
        width: 500,
        height: 300,
    });
    tfvis.render.linechart({ name: 'Waiting Time', tab: 'Training' }, { values: waitingTimeList }, {
        xLabel: 'Version',
        yLabel: 'Waiting Time',
        width: 500,
        height: 300,
    });
    tf.nextFrame();
}

export function saveMetrics() {
    setAgentLog({
        train: trainBuffer.getLastN(trainBuffer.getSize() / 10),
        batch: batchBuffer.getLastN(batchBuffer.getSize() / 10),
        epoch: epochBuffer.getLastN(epochBuffer.getSize() / 10),
        rewards: rewardBuffer.getLastN(rewardBuffer.getSize() / 10),
    });
}

export function logEpoch(data: {
    kl: number;
    valueLoss: number;
    policyLoss: number;
}) {
    epochBuffer.add(data);
    drawEpoch(epochBuffer.toArray());
}

export function logRewards(rewards: number[]) {
    rewardBuffer.add(...rewards);
    drawRewards(rewardBuffer.toArray());
}

export function logBatch(data: { versionDelta: number, batchSize: number }) {
    batchBuffer.add(data);
    drawBatch(batchBuffer.toArray());
}

export function logTrain(data: { trainTime: number, waitTime: number }) {
    trainBuffer.add(data);
    drawTrain(trainBuffer.toArray());
}

