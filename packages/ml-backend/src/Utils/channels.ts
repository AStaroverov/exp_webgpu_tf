// Single-thread refactor: BroadcastChannel заменён на лёгкий RxJS Subject-шим.
// Сохраняем похожий интерфейс (postMessage, onmessage) для минимальных правок.
import { Observable, Subject } from 'rxjs';

type ShimChannel<T = any> = {
    postMessage: (data: T) => void,
    onmessage: ((e: { data: T }) => void) | null,
    obs: Observable<T>
};

function createChannel<T = any>(): ShimChannel<T> {
    const subj = new Subject<T>();
    let handler: ((e: { data: T }) => void) | null = null;
    subj.subscribe((data) => handler && handler({ data }));
    return {
        postMessage: (data: T) => subj.next(data),
        get onmessage() { return handler; },
        set onmessage(fn) { handler = fn; },
        obs: subj.asObservable()
    } as ShimChannel<T>;
}

export const forceExitChannel = createChannel<null>();

// Метрики временно отключены (см. требование) — каналы оставлены как заглушки.
export const metricsChannels = {
    rewards: createChannel<number[] | Float32Array>(),
    values: createChannel<number[] | Float32Array>(),
    returns: createChannel<number[] | Float32Array>(),
    tdErrors: createChannel<number[] | Float32Array>(),
    advantages: createChannel<number[] | Float32Array>(),
    kl: createChannel<number[] | Float32Array>(),
    lr: createChannel<number[] | Float32Array>(),
    valueLoss: createChannel<number[] | Float32Array>(),
    policyLoss: createChannel<number[] | Float32Array>(),
    trainTime: createChannel<number[]>(),
    waitTime: createChannel<number[]>(),
    batchSize: createChannel<number[]>(),
    versionDelta: createChannel<number[]>(),
    successRatio: createChannel<any>(),
};