import * as tfvis from '@tensorflow/tfjs-vis';
import { isObject, throttle } from 'lodash-es';
import { metricsChannels } from '../../Common/channels.ts';
import { getAgentLog, setAgentLog } from './store.ts';
import { CompressedBuffer } from './CompressedBuffer.ts';
import { get } from 'lodash';

const store = {
    rewards: new CompressedBuffer(10_000, 10),
    values: new CompressedBuffer(10_000, 5),
    returns: new CompressedBuffer(10_000, 5),
    tdErrors: new CompressedBuffer(10_000, 5),
    advantages: new CompressedBuffer(10_000, 5),
    kl: new CompressedBuffer(1_000, 5),
    lr: new CompressedBuffer(1_000, 5),
    valueLoss: new CompressedBuffer(1_000, 5),
    policyLoss: new CompressedBuffer(1_000, 5),
    trainTime: new CompressedBuffer(1_000, 5),
    waitTime: new CompressedBuffer(1_000, 5),
    versionDelta: new CompressedBuffer(1_000, 5),
    batchSize: new CompressedBuffer(1_000, 5),
};

getAgentLog().then((data) => {
    if (isObject(data) && get(data, 'version') === 1) {
        store.rewards.fromJson(get(data, 'rewards'));
        store.values.fromJson(get(data, 'values'));
        store.returns.fromJson(get(data, 'returns'));
        store.tdErrors.fromJson(get(data, 'tdErrors'));
        store.advantages.fromJson(get(data, 'advantages'));
        store.kl.fromJson(get(data, 'kl'));
        store.lr.fromJson(get(data, 'lr'));
        store.valueLoss.fromJson(get(data, 'valueLoss'));
        store.policyLoss.fromJson(get(data, 'policyLoss'));
        store.trainTime.fromJson(get(data, 'trainTime'));
        store.waitTime.fromJson(get(data, 'waitTime'));
        store.versionDelta.fromJson(get(data, 'versionDelta'));
        store.batchSize.fromJson(get(data, 'batchSize'));
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
        tdErrors: store.tdErrors.toJson(),
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
    metricsChannels.tdErrors.onmessage = w((event) => store.tdErrors.add(...event.data));
    metricsChannels.advantages.onmessage = w((event) => store.advantages.add(...event.data));
    metricsChannels.kl.onmessage = w((event) => store.kl.add(...event.data));
    metricsChannels.policyLoss.onmessage = w((event) => store.policyLoss.add(...event.data));
    metricsChannels.valueLoss.onmessage = w((event) => store.valueLoss.add(...event.data));
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
    tfvis.render.scatterplot({ name: 'Reward', tab }, {
        values: [minRewards, maxRewards, avgRewards, avgRewardsMA],
        series: ['Min', 'Max', 'Avg', 'Avg MA'],
    }, {
        xLabel: 'Version',
        yLabel: 'Reward',
        width: 500,
        height: 300,
    });

    const avgKL = store.kl.toArray();
    tfvis.render.scatterplot({ name: 'KL', tab }, {
        values: [avgKL, calculateMovingAverage(avgKL, 25)],
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

    tfvis.render.scatterplot({ name: 'Value', tab }, {
        values: [
            store.values.toArrayMin(),
            store.values.toArrayMax(),
            store.values.toArray(),
        ],
    }, {
        xLabel: 'Version',
        yLabel: 'Value',
        width: 500,
        height: 300,
    });

    tfvis.render.scatterplot({ name: 'Return', tab }, {
        values: [
            store.returns.toArrayMin(),
            store.returns.toArrayMax(),
            store.returns.toArray(),
        ],
    }, {
        xLabel: 'Version',
        yLabel: 'Return',
        width: 500,
        height: 300,
    });

    tfvis.render.scatterplot({ name: 'TD Error', tab }, {
        values: [store.tdErrors.toArrayMin(), store.tdErrors.toArrayMax(), store.tdErrors.toArray()],
    }, {
        xLabel: 'Version',
        yLabel: 'TD Error',
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

type RenderPoint = { x: number, y: number };

function calculateMovingAverage(data: RenderPoint[], windowSize: number): RenderPoint[] {
    if (data.length === 0) return [];
    if (windowSize <= 0) throw new Error('Window size must be positive');

    const averaged: RenderPoint[] = [];
    let sum = 0;

    // Первое окно - рассчитываем сумму первых элементов
    const initialWindow = Math.min(windowSize, data.length);
    for (let i = 0; i < initialWindow; i++) {
        const val = data[i].y;
        if (!isFinite(val)) continue;
        sum += val;
        averaged.push({ x: data[i].x, y: sum / (i + 1) });
    }

    // Используем скользящее окно для остальных элементов
    for (let i = initialWindow; i < data.length; i++) {
        const val = data[i].y - (i >= windowSize ? data[i - windowSize].y : 0);
        if (!isFinite(val)) continue;
        sum += val;
        averaged.push({ x: data[i].x, y: sum / windowSize });
    }

    return averaged;
}