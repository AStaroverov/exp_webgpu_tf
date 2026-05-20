import Stats from 'stats-gl';

export const stats = new Stats({
    trackGPU: true,
    trackHz: false,
    trackCPT: false,
    logsPerSecond: 4,
    graphsPerSecond: 30,
    samplesLog: 40,
    samplesGraph: 10,
    precision: 2,
    horizontal: true,
    minimal: false,
    mode: 2,
});
