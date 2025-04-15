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
    valueLoss: new BroadcastChannel('valueLoss'),
    policyLoss: new BroadcastChannel('policyLoss'),
    trainTime: new BroadcastChannel('trainTime'),
    waitTime: new BroadcastChannel('waitTime'),
    versionDelta: new BroadcastChannel('versionDelta'),
    batchSize: new BroadcastChannel('batchSize'),
    prbAlpha: new BroadcastChannel('prbAlpha'),
    prbBeta: new BroadcastChannel('prbBeta'),
};