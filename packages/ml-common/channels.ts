export const forceExitChannel = new BroadcastChannel('exit');

if (globalThis.document != null) {
    // we cannot listen forceExitChannel, because don't send message at the same thread
    new BroadcastChannel('exit').onmessage = () => {
        globalThis.location.reload();
    };
}

export const metricsChannels = {
    rewards: new BroadcastChannel('rewards'),
    values: new BroadcastChannel('value'),
    returns: new BroadcastChannel('returns'),
    tdErrors: new BroadcastChannel('tdErrors'),
    advantages: new BroadcastChannel('advantages'),
    kl: new BroadcastChannel('kl'),
    lr: new BroadcastChannel('lr'),
    mean: new BroadcastChannel('mean'),
    logStd: new BroadcastChannel('logStd'),
    vTraceStdRatio: new BroadcastChannel('vTraceStdRatio'),
    vTraceExplainedVariance: new BroadcastChannel('vTraceExplainedVariance'),
    valueLoss: new BroadcastChannel('valueLoss'),
    policyLoss: new BroadcastChannel('policyLoss'),
    trainTime: new BroadcastChannel('trainTime'),
    waitTime: new BroadcastChannel('waitTime'),
    batchSize: new BroadcastChannel('batchSize'),
    versionDelta: new BroadcastChannel('versionDelta'),
    successRatio: new BroadcastChannel('successRatio'),

};