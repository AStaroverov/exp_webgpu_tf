import { Chart } from 'chart.js/auto';
import { get } from 'lodash';
import { isObject, mapValues, throttle } from 'lodash-es';
import { metricsChannels } from '../../channels.ts';
import { ACTION_DIM } from '../../consts.ts';
import { scenariosCount } from '../../Curriculum/createScenarioByCurriculumState.ts';
import { CompressedBuffer } from './CompressedBuffer.ts';
import { RingBuffer } from './RingBuffer.ts';
import { getAgentLog, setAgentLog } from './store.ts';

// --- Chart.js global dark theme ---
Chart.defaults.color = '#999';
Chart.defaults.borderColor = '#2a2a3e';
Chart.defaults.backgroundColor = 'transparent';

// --- Types ---

type MetricIndex = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;
type Point = { x: number; y: number };

interface SeriesDef {
    getData: () => Point[];
    color: string;
    label?: string;
    dashed?: boolean;
    width?: number;
    dot?: boolean;
}

// --- Store ---

const store = {
    rewards: new CompressedBuffer(10_000, 10),
    kl: new CompressedBuffer(1_000, 5),
    lr: new CompressedBuffer(1_000, 5),

    values: new RingBuffer(30_000),
    returns: new RingBuffer(30_000),
    tdErrors: new RingBuffer(30_000),
    advantages: new RingBuffer(30_000),

    ...Array.from({ length: ACTION_DIM }, (_, i) => i).reduce((acc, i) => {
        acc[`logit${i as MetricIndex}`] = new RingBuffer(5_000);
        return acc;
    }, {} as Record<`logit${MetricIndex}`, RingBuffer>),

    valueLoss: new CompressedBuffer(1_000, 5),
    policyLoss: new CompressedBuffer(1_000, 5),
    entropy: new CompressedBuffer(1_000, 5),
    entropyAlpha: new CompressedBuffer(1_000, 5),

    trainTime: new CompressedBuffer(1_000, 5),
    waitTime: new CompressedBuffer(1_000, 5),
    batchSize: new CompressedBuffer(1_000, 5),
    versionDelta: new CompressedBuffer(1_000, 5),

    vTraceStdRatio: new RingBuffer(5_000),
    vTraceExplainedVariance: new RingBuffer(5_000),

    ...Array.from({ length: scenariosCount }, (_, i) => i).reduce((acc, i) => {
        acc[`successRatio${i as MetricIndex}Ref`] = new CompressedBuffer(500, 5);
        acc[`successRatio${i as MetricIndex}Train`] = new CompressedBuffer(500, 5);
        return acc;
    }, {} as Record<`successRatio${MetricIndex}${'Ref' | 'Train'}`, CompressedBuffer>),
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

// --- Save ---

export const saveMetrics = throttle(() => {
    setAgentLog({
        version: 1,
        ...mapValues(store, (value) => value.toJson()),
    });
}, 10_000);

// --- Subscriptions ---

export function subscribeOnMetrics() {
    const w = (callback: (event: MessageEvent) => void) => {
        return (event: MessageEvent) => {
            callback(event);
            saveMetrics();
        };
    };
    Object.keys(metricsChannels).forEach((key) => {
        const channel = metricsChannels[key as keyof typeof metricsChannels];
        if (key === 'logit') {
            channel.onmessage = w((event) => {
                const logits = event.data as number[][];
                store.logit0.addList(logits[0]);
                store.logit1.addList(logits[1]);
                store.logit2.addList(logits[2]);
                store.logit3.addList(logits[3]);
            });
        } else if (key === 'successRatio') {
            channel.onmessage = w((event) => {
                (event.data as {
                    scenarioIndex: MetricIndex,
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

// --- Charts Panel (popup) ---

let panelEl: HTMLElement | null = null;
let visible = false;

interface ChartEntry {
    chart: Chart;
    series: SeriesDef[];
}
const entries: ChartEntry[] = [];

export function toggleChartsPanel() {
    if (!panelEl) {
        panelEl = buildPanel();
        document.body.appendChild(panelEl);
    }
    visible = !visible;
    panelEl.style.display = visible ? 'flex' : 'none';
    if (visible) updateCharts();
}

export function isChartsPanelVisible() {
    return visible;
}

export function updateCharts() {
    if (!visible) return;
    for (const entry of entries) {
        for (let i = 0; i < entry.series.length; i++) {
            entry.chart.data.datasets[i].data = entry.series[i].getData();
        }
        entry.chart.update('none');
    }
}

// --- Build Panel ---

function buildPanel(): HTMLElement {
    const panel = document.createElement('div');
    Object.assign(panel.style, {
        position: 'fixed', left: '0', top: '0', right: '460px', bottom: '0',
        background: 'rgba(10, 10, 20, 0.95)',
        overflowY: 'auto',
        zIndex: '999',
        display: 'none',
        flexDirection: 'column',
        padding: '12px',
        fontFamily: 'monospace',
        color: '#ccc',
    });

    // header
    const header = document.createElement('div');
    Object.assign(header.style, { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' });
    const title = document.createElement('span');
    title.textContent = 'Metrics';
    title.style.fontSize = '15px';
    const closeBtn = document.createElement('button');
    closeBtn.textContent = 'Close (M)';
    Object.assign(closeBtn.style, { cursor: 'pointer', padding: '4px 12px', background: '#333', color: '#ccc', border: '1px solid #555', borderRadius: '4px', fontFamily: 'monospace' });
    closeBtn.onclick = toggleChartsPanel;
    header.append(title, closeBtn);
    panel.appendChild(header);

    // chart grid
    const grid = document.createElement('div');
    Object.assign(grid.style, { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(480px, 1fr))', gap: '8px' });
    panel.appendChild(grid);

    // ---- Success Ratios ----
    addSectionHeader(grid, 'Success Ratios');
    for (let i = 0; i < scenariosCount; i++) {
        const idx = i as MetricIndex;
        addChartToGrid(grid, `Scenario ${i}`, [
            { getData: () => store[`successRatio${idx}Train`].toArray(), color: '#4a9eff', label: 'Train', dot: true },
            { getData: () => movingAvg(store[`successRatio${idx}Train`].toArray(), 20), color: '#ff6b6b', label: 'Train MA', width: 2 },
            { getData: () => store[`successRatio${idx}Ref`].toArray(), color: '#51cf66', label: 'Ref', dot: true },
            { getData: () => movingAvg(store[`successRatio${idx}Ref`].toArray(), 20), color: '#ffd43b', label: 'Ref MA', width: 2 },
        ]);
    }

    // ---- Training ----
    addSectionHeader(grid, 'Training');

    addChartToGrid(grid, 'KL', [
        { getData: () => store.kl.toArray(), color: '#4a9eff', label: 'KL' },
        { getData: () => movingMedian(store.kl.toArray(), 25), color: '#ff6b6b', label: 'Median', width: 2 },
    ]);
    addChartToGrid(grid, 'Learning Rate', [
        { getData: () => store.lr.toArray(), color: '#51cf66', label: 'LR' },
    ]);
    addChartToGrid(grid, 'Entropy H(π)', [
        { getData: () => store.entropy.toArray(), color: '#b197fc', label: 'Entropy' },
        { getData: () => movingAvg(store.entropy.toArray(), 20), color: '#ff6b6b', label: 'MA', width: 2 },
    ]);
    addChartToGrid(grid, 'Entropy α', [
        { getData: () => store.entropyAlpha.toArray(), color: '#63e6be', label: 'α' },
    ]);
    addChartToGrid(grid, 'V-Trace Explained Variance', [
        { getData: () => store.vTraceExplainedVariance.toArray(), color: '#4a9eff', label: 'EV' },
        { getData: () => constLine(store.vTraceExplainedVariance.toArray(), 0.65), color: '#ffd43b', label: '0.65', dashed: true },
        { getData: () => constLine(store.vTraceExplainedVariance.toArray(), 0.75), color: '#ff6b6b', label: '0.75', dashed: true },
    ]);
    addChartToGrid(grid, 'V-Trace σ(V)/σ(R)', [
        { getData: () => store.vTraceStdRatio.toArray(), color: '#4a9eff', label: 'Ratio' },
        { getData: () => constLine(store.vTraceStdRatio.toArray(), 0.8), color: '#ffd43b', label: '0.8', dashed: true },
        { getData: () => constLine(store.vTraceStdRatio.toArray(), 1.2), color: '#ff6b6b', label: '1.2', dashed: true },
    ]);
    addChartToGrid(grid, 'Reward', [
        { getData: () => store.rewards.toArray(), color: '#4a9eff', label: 'Reward' },
    ]);
    addChartToGrid(grid, 'Value', [
        { getData: () => store.values.toArray(), color: '#b197fc', label: 'Value' },
    ]);
    addChartToGrid(grid, 'Return', [
        { getData: () => store.returns.toArray(), color: '#63e6be', label: 'Return' },
    ]);
    addChartToGrid(grid, 'TD Error', [
        { getData: () => store.tdErrors.toArray(), color: '#ffa94d', label: 'TD Error' },
    ]);
    addChartToGrid(grid, 'Advantage', [
        { getData: () => store.advantages.toArray(), color: '#e599f7', label: 'Adv' },
        { getData: () => movingAvg(store.advantages.toArray(), 1000), color: '#ff6b6b', label: 'MA', width: 2 },
    ]);

    // ---- Policy ----
    addSectionHeader(grid, 'Policy');

    for (let i = 0; i < ACTION_DIM; i++) {
        addScatterToGrid(grid, `Logit ${i}`, [
            { getData: () => store[`logit${i as MetricIndex}`].toArray(), color: '#4a9eff', label: `Logit ${i}` },
        ]);
    }
    addChartToGrid(grid, 'Policy Loss', [
        { getData: () => store.policyLoss.toArray(), color: '#ff6b6b', label: 'Policy Loss' },
    ]);
    addChartToGrid(grid, 'Value Loss', [
        { getData: () => store.valueLoss.toArray(), color: '#ffa94d', label: 'Value Loss' },
    ]);

    // ---- Performance ----
    addSectionHeader(grid, 'Performance');

    addChartToGrid(grid, 'Train Time', [
        { getData: () => store.trainTime.toArray(), color: '#4a9eff', label: 'Train' },
        { getData: () => movingAvg(store.trainTime.toArray(), 100), color: '#ff6b6b', label: 'MA', width: 2 },
    ]);
    addChartToGrid(grid, 'Wait Time', [
        { getData: () => store.waitTime.toArray(), color: '#ffa94d', label: 'Wait' },
        { getData: () => movingAvg(store.waitTime.toArray(), 100), color: '#ff6b6b', label: 'MA', width: 2 },
    ]);
    addChartToGrid(grid, 'Batch Version Delta', [
        { getData: () => store.versionDelta.toArray(), color: '#b197fc', label: 'Delta' },
        { getData: () => movingAvg(store.versionDelta.toArray(), 100), color: '#ff6b6b', label: 'MA', width: 2 },
    ]);
    addChartToGrid(grid, 'Batch Size', [
        { getData: () => store.batchSize.toArray(), color: '#51cf66', label: 'Size' },
    ]);

    return panel;
}

function addSectionHeader(container: HTMLElement, text: string) {
    const h = document.createElement('div');
    h.textContent = text;
    Object.assign(h.style, {
        gridColumn: '1 / -1',
        fontSize: '13px',
        fontWeight: 'bold',
        color: '#aaa',
        borderBottom: '1px solid #333',
        padding: '8px 0 4px',
        marginTop: '4px',
    });
    container.appendChild(h);
}

function addChartToGrid(container: HTMLElement, title: string, series: SeriesDef[]) {
    const wrapper = document.createElement('div');
    Object.assign(wrapper.style, { background: '#111122', borderRadius: '4px', padding: '4px' });

    const canvas = document.createElement('canvas');
    wrapper.appendChild(canvas);
    container.appendChild(wrapper);

    const chart = new Chart(canvas, {
        type: 'line',
        data: {
            datasets: series.map(s => ({
                data: [] as Point[],
                borderColor: s.color,
                borderWidth: s.dot ? 0 : (s.width ?? 1.2),
                borderDash: s.dashed ? [6, 4] : [],
                pointRadius: s.dot ? 1.5 : 0,
                backgroundColor: s.color,
                showLine: !s.dot,
                fill: false,
                tension: 0,
                label: s.label ?? '',
            })),
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2,
            animation: false,
            parsing: false,
            scales: {
                x: {
                    type: 'linear',
                    ticks: { maxTicksLimit: 6, font: { size: 10 } },
                    grid: { color: '#1a1a2e' },
                },
                y: {
                    type: 'linear',
                    ticks: { maxTicksLimit: 5, font: { size: 10 } },
                    grid: { color: '#1a1a2e' },
                },
            },
            plugins: {
                legend: {
                    display: series.length > 1,
                    labels: { boxWidth: 12, font: { size: 10 } },
                },
                title: {
                    display: true,
                    text: title,
                    font: { size: 12, weight: 'normal' },
                },
                decimation: {
                    enabled: true,
                    algorithm: 'lttb',
                    samples: 800,
                },
            },
        },
    });

    entries.push({ chart, series });
}

function addScatterToGrid(container: HTMLElement, title: string, series: SeriesDef[]) {
    const wrapper = document.createElement('div');
    Object.assign(wrapper.style, { background: '#111122', borderRadius: '4px', padding: '4px' });

    const canvas = document.createElement('canvas');
    wrapper.appendChild(canvas);
    container.appendChild(wrapper);

    const chart = new Chart(canvas, {
        type: 'scatter',
        data: {
            datasets: series.map(s => ({
                data: [] as Point[],
                backgroundColor: s.color,
                borderColor: s.color,
                pointRadius: 1.2,
                label: s.label ?? '',
            })),
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2,
            animation: false,
            parsing: false,
            scales: {
                x: {
                    type: 'linear',
                    ticks: { maxTicksLimit: 6, font: { size: 10 } },
                    grid: { color: '#1a1a2e' },
                },
                y: {
                    type: 'linear',
                    ticks: { maxTicksLimit: 5, font: { size: 10 } },
                    grid: { color: '#1a1a2e' },
                },
            },
            plugins: {
                legend: {
                    display: series.length > 1,
                    labels: { boxWidth: 12, font: { size: 10 } },
                },
                title: {
                    display: true,
                    text: title,
                    font: { size: 12, weight: 'normal' },
                },
            },
        },
    });

    entries.push({ chart, series });
}

// --- Helpers ---

function constLine(data: Point[], value: number): Point[] {
    if (data.length === 0) return [];
    return [
        { x: data[0].x, y: value },
        { x: data[data.length - 1].x, y: value },
    ];
}

function movingAvg(data: Point[], windowSize: number): Point[] {
    if (data.length === 0) return [];
    const result: Point[] = [];
    let sum = 0;
    const w = Math.min(windowSize, data.length);
    for (let i = 0; i < w; i++) {
        const val = data[i].y;
        if (!isFinite(val)) continue;
        sum += val;
        result.push({ x: data[i].x, y: sum / (i + 1) });
    }
    for (let i = w; i < data.length; i++) {
        const val = data[i].y - (i >= windowSize ? data[i - windowSize].y : 0);
        if (!isFinite(val)) continue;
        sum += val;
        result.push({ x: data[i].x, y: sum / windowSize });
    }
    return result;
}

function movingMedian(data: Point[], windowSize: number): Point[] {
    if (data.length === 0) return [];
    const result: Point[] = [];
    const win: number[] = [];
    for (let i = 0; i < data.length; i++) {
        const val = data[i].y;
        if (!isFinite(val)) continue;
        win.push(val);
        if (win.length > windowSize) win.shift();
        const sorted = win.toSorted((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        const median = sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
        result.push({ x: data[i].x, y: median });
    }
    return result;
}
