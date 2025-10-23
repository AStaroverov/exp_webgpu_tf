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

    mean: new CompressedBuffer(1_000, 5),
    logStd: new CompressedBuffer(1_000, 5),
    valueLoss: new CompressedBuffer(1_000, 5),
    policyLoss: new CompressedBuffer(1_000, 5),

    trainTime: new CompressedBuffer(1_000, 5),
    waitTime: new CompressedBuffer(1_000, 5),
    batchSize: new CompressedBuffer(1_000, 5),
    versionDelta: new CompressedBuffer(1_000, 5),
    vTraceExplainedVariance: new CompressedBuffer(1_000, 5),
    vTraceStdRatio: new CompressedBuffer(1_000, 5),
    // successRatioN
    ...Array.from({ length: scenariosCount }, (_, i) => i).reduce((acc, i) => {
        acc[`successRatio${i as SuccessRatioIndex}Ref`] = new CompressedBuffer(500, 5);
        acc[`successRatio${i as SuccessRatioIndex}Train`] = new CompressedBuffer(500, 5);
        return acc;
    }, {} as Record<`successRatio${SuccessRatioIndex}${'Ref' | 'Train'}`, CompressedBuffer>),
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
                    successRatio: number,
                    isReference: boolean,
                }[]).forEach(({ scenarioIndex, successRatio, isReference }) => {
                    store[`successRatio${scenarioIndex}${isReference ? 'Ref' : 'Train'}` as keyof typeof store].add(successRatio);
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
        const successRatioTrain = store[`successRatio${index}Train`].toArray();
        const successRatioTrainMA = calculateMovingAverage(successRatioTrain, 100);
        tfvis.render.scatterplot({ name: 'Train Success Ratio ' + index, tab }, {
            values: [successRatioTrain, successRatioTrainMA],
            series: ['V', 'MA'],
        }, {
            xLabel: 'Version',
            yLabel: 'Train Success Ratio',
            width: 500,
            height: 300,
        });

        const successRatioRef = store[`successRatio${index}Ref`].toArray();
        const successRatioRefMA = calculateMovingAverage(successRatioRef, 100);
        tfvis.render.scatterplot({ name: 'Ref Success Ratio ' + index, tab }, {
            values: [successRatioRef, successRatioRefMA],
            series: ['V', 'MA'],
        }, {
            xLabel: 'Version',
            yLabel: 'Ref Success Ratio',
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
    tfvis.render.scatterplot({ name: 'Reward', tab }, {
        values: [avgRewards],
    }, {
        xLabel: 'Version',
        yLabel: 'Reward',
        width: 500,
        height: 300,
    });

    const avgKL = store.kl.toArray();
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

    // const avgKLPerturbed = store.klPerturbed.toArray();
    // tfvis.render.scatterplot({ name: 'KL - Perturbed', tab }, {
    //     values: [
    //         avgKLPerturbed,
    //         calculateMovingMedian(avgKLPerturbed, 25),
    //     ],
    //     series: ['Avg', 'MA'],
    // }, {
    //     xLabel: 'Version',
    //     yLabel: 'KL',
    //     width: 500,
    //     height: 300,
    // });

    // tfvis.render.linechart({ name: 'Perturb Scale', tab }, {
    //     values: [store.perturbScale.toArray()],
    // }, {
    //     xLabel: 'Version',
    //     yLabel: 'Perturb Scale',
    //     width: 500,
    //     height: 300,
    // });

    const evSeries = store.vTraceExplainedVariance.toArray();
    tfvis.render.linechart({ name: 'V-Trace Explained Variance', tab }, {
        values: [
            evSeries,
            constantLine(evSeries, 0.65),
            constantLine(evSeries, 0.75),
        ],
        series: ['Explained Variance', 'Target Low', 'Target High'],
    }, {
        xLabel: 'Version',
        yLabel: 'Explained Variance',
        width: 500,
        height: 300,
    });

    const stdRatioSeries = store.vTraceStdRatio.toArray();
    tfvis.render.linechart({ name: 'V-Trace Std Ratio', tab }, {
        values: [
            stdRatioSeries,
            constantLine(stdRatioSeries, 0.8),
            constantLine(stdRatioSeries, 1.2),
        ],
        series: ['σ(V)/σ(R)', 'Target Low', 'Target High'],
    }, {
        xLabel: 'Version',
        yLabel: 'σ(V)/σ(R)',
        width: 500,
        height: 300,
    });

    tfvis.render.scatterplot({ name: 'Value', tab }, {
        values: [
            store.values.toArray(),
        ],
    }, {
        xLabel: 'Version',
        yLabel: 'Value',
        width: 500,
        height: 300,
    });

    tfvis.render.scatterplot({ name: 'Return', tab }, {
        values: [store.returns.toArray()],
    }, {
        xLabel: 'Version',
        yLabel: 'Return',
        width: 500,
        height: 300,
    });

    tfvis.render.scatterplot({ name: 'TD Error', tab }, {
        values: [store.tdErrors.toArray()],
    }, {
        xLabel: 'Version',
        yLabel: 'TD Error',
        width: 500,
        height: 300,
    });

    tfvis.render.scatterplot({ name: 'Advantage', tab }, {
        values: [store.advantages.toArray()],
    }, {
        xLabel: 'Version',
        yLabel: 'Advantage',
        width: 500,
        height: 300,
    });

}

function drawTab2() {
    tfvis.render.scatterplot({ name: 'mean', tab: 'Tab 2' }, {
        values: [store.mean.toArray()],
    }, {
        xLabel: 'Version',
        yLabel: 'mean',
        width: 500,
        height: 300,
    });
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

function constantLine(data: RenderPoint[], value: number): RenderPoint[] {
    return data.map((point) => ({ x: point.x, y: value }));
}

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
