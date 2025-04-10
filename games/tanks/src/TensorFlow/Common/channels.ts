export const forceExitChannel = new BroadcastChannel('exit');

if (globalThis.document != null) {
    // we cannot listen forceExitChannel, because don't send message at the same thread
    new BroadcastChannel('exit').onmessage = () => {
        globalThis.location.reload();
    };
}

export const newModelVersionChannel = new BroadcastChannel('newVersion');

export const metricsChannels = {
    rewards: new BroadcastChannel('rewards'),
    values: new BroadcastChannel('value'),
    returns: new BroadcastChannel('returns'),
    advantages: new BroadcastChannel('advantages'),
    kl: new BroadcastChannel('kl'),
    lr: new BroadcastChannel('lr'),
    valueLoss: new BroadcastChannel('valueLoss'),
    policyLoss: new BroadcastChannel('policyLoss'),
    trainTime: new BroadcastChannel('trainTime'),
    waitTime: new BroadcastChannel('waitTime'),
    versionDelta: new BroadcastChannel('versionDelta'),
    batchSize: new BroadcastChannel('batchSize'),
};