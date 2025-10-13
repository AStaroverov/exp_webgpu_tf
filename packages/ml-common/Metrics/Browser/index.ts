import * as tfvis from '@tensorflow/tfjs-vis';
import { get } from 'lodash';
import { isObject, mapValues, throttle } from 'lodash-es';
import { metricsChannels } from '../../channels.ts';
import { scenariosCount } from '../../Curriculum/createScenarioByCurriculumState.ts';
import { CompressedBuffer } from './CompressedBuffer.ts';
import { getAgentLog, setAgentLog } from './store.ts';

type SuccessRatioIndex = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;

const store = {
    rewards: new CompressedBuffer(10_000, 10),

    kl: new CompressedBuffer(1_000, 5),
    klPerturbed: new CompressedBuffer(1_000, 5),
    lr: new CompressedBuffer(1_000, 5),
    perturbScale: new CompressedBuffer(1_000, 5),

    values: new CompressedBuffer(10_000, 5),
    returns: new CompressedBuffer(10_000, 5),
    tdErrors: new CompressedBuffer(10_000, 5),
    advantages: new CompressedBuffer(10_000, 5),

    logStd: new CompressedBuffer(1_000, 5),
    valueLoss: new CompressedBuffer(1_000, 5),
    policyLoss: new CompressedBuffer(1_000, 5),

    trainTime: new CompressedBuffer(1_000, 5),
    waitTime: new CompressedBuffer(1_000, 5),
    batchSize: new CompressedBuffer(1_000, 5),
    versionDelta: new CompressedBuffer(1_000, 5),
    // successRatioN
    ...Array.from({ length: scenariosCount }, (_, i) => i).reduce((acc, i) => {
        acc[`successRatio${i as SuccessRatioIndex}`] = new CompressedBuffer(500, 5);
        return acc;
    }, {} as Record<`successRatio${SuccessRatioIndex}`, CompressedBuffer>),
};

getAgentLog().then((data) => {
    if (isObject(data) && get(data, 'version') === 1) {
        Object.keys(store).forEach((key) => {
            if (!(key in store)) return;
            store[key as keyof typeof store].fromJson(get(data, key));
        });
    }

    subscribeOnMetrics();
});

export function drawMetrics() {
    drawTab0();
    drawTab1();
    drawTab2();
    drawTab3();
}

export const saveMetrics = throttle(() => {
    setAgentLog({
        version: 1,
        ...mapValues(store, (value) => value.toJson()),
    });
}, 10_000);

export function subscribeOnMetrics() {
    const w = (callback: (event: MessageEvent) => void) => {
        return (event: MessageEvent) => {
            callback(event);
            saveMetrics();
        };
    };
    Object.keys(metricsChannels).forEach((key) => {
        const channel = metricsChannels[key as keyof typeof metricsChannels];
        if (key === 'successRatio') {
            channel.onmessage = w((event) => {
                (event.data as {
                    scenarioIndex: SuccessRatioIndex,
                    successRatio: number
                }[]).forEach(({ scenarioIndex, successRatio }) => {
                    store[`successRatio${scenarioIndex}` as keyof typeof store].add(successRatio);
                });
            });
        } else {
            channel.onmessage = w((event) => {
                store[key as keyof typeof store].add(...event.data);
            });
        }
    });
}

function drawTab0() {
    const tab = 'Tab 0';
    const renderSuccessRatio = (index: SuccessRatioIndex) => {
        const successRatio = store[`successRatio${index}`].toArray();
        const successRatioMA = calculateMovingAverage(successRatio, 100);
        tfvis.render.scatterplot({ name: 'Success Ratio ' + index, tab }, {
            values: [successRatio, successRatioMA],
            series: ['Success Ratio', 'MA'],
        }, {
            xLabel: 'Version',
            yLabel: 'Success Ratio',
            width: 500,
            height: 300,
        });
    };

    Array.from({ length: scenariosCount }, (_, i) => i).forEach((i) => {
        renderSuccessRatio(i as SuccessRatioIndex);
    });
}

function drawTab1() {
    const tab = 'Tab 1';

    const avgRewards = store.rewards.toArray();
    const minRewards = store.rewards.toArrayMin();
    const maxRewards = store.rewards.toArrayMax();
    tfvis.render.scatterplot({ name: 'Reward', tab }, {
        values: [minRewards, maxRewards, avgRewards],
        series: ['Min', 'Max', 'Avg'],
    }, {
        xLabel: 'Version',
        yLabel: 'Reward',
        width: 500,
        height: 300,
    });

    const avgKL = store.kl.toArray();
    const avgKLPerturbed = store.klPerturbed.toArray();
    tfvis.render.scatterplot({ name: 'KL', tab }, {
        values: [
            avgKL,
            calculateMovingMedian(avgKL, 25),
        ],
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

    tfvis.render.scatterplot({ name: 'KL - Perturbed', tab }, {
        values: [
            avgKLPerturbed,
            calculateMovingMedian(avgKLPerturbed, 25),
        ],
        series: ['Avg', 'MA'],
    }, {
        xLabel: 'Version',
        yLabel: 'KL',
        width: 500,
        height: 300,
    });

    tfvis.render.linechart({ name: 'Perturb Scale', tab }, {
        values: [store.perturbScale.toArray()],
    }, {
        xLabel: 'Version',
        yLabel: 'Perturb Scale',
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
    tfvis.render.scatterplot({ name: 'Log Std', tab: 'Tab 2' }, {
        values: [store.logStd.toArray()],
    }, {
        xLabel: 'Version',
        yLabel: 'Log Std',
        width: 500,
        height: 300,
    });

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

function calculateMovingMedian(data: RenderPoint[], windowSize: number): RenderPoint[] {
    if (data.length === 0) return [];
    if (windowSize <= 0) throw new Error('Window size must be positive');

    const averaged: RenderPoint[] = [];
    const window: number[] = [];

    for (let i = 0; i < data.length; i++) {
        const val = data[i].y;
        if (!isFinite(val)) continue;

        window.push(val);
        if (window.length > windowSize) {
            window.shift();
        }

        const sortedWindow = window.toSorted((a, b) => a - b);
        const mid = Math.floor(sortedWindow.length / 2);
        const median = sortedWindow.length % 2 !== 0
            ? sortedWindow[mid]
            : (sortedWindow[mid - 1] + sortedWindow[mid]) / 2;

        averaged.push({ x: data[i].x, y: median });
    }

    return averaged;
}

function calculateMovingMedianAverage(data: RenderPoint[], windowSize: number): RenderPoint[] {
    const average = calculateMovingAverage(data, windowSize);
    const median = calculateMovingMedian(data, windowSize);

    return average.map((point, index) => ({
        x: point.x,
        y: (point.y + median[index].y) / 2,
    }));
}