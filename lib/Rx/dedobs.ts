import { identity, Observable } from 'rxjs';
import { tap } from 'rxjs/operators';

import { shareReplayWithDelayedReset, shareReplayWithImmediateReset } from './share.ts';
import { EMPTY_OBJECT } from '../const.ts';
import { debounceBy } from '../debounceBy.ts';
import { shallowHash } from '../shallowHash.ts';
import { macroTasks } from '../TasksScheduler/macroTasks.ts';

type TKey = boolean | number | string;

const DEFAULT_NORMALIZER = <T extends any[]>(args: T): TKey => shallowHash(...args);
const DEFAULT_REMOVE_UNSUBSCRIBED_DELAY = 5_000;

export const DEDOBS_SKIP_KEY = Symbol('SKIP_KEY');
export const DEDOBS_REMOVE_DELAY = 60 * 1000; // 1 minute
export const DEDOBS_RESET_DELAY = 30 * 1000; // 30 seconds

export type TDedobsOptions<Args extends unknown[]> = {
    normalize?: (args: Args) => typeof DEDOBS_SKIP_KEY | TKey;
    /**
     * Delay on reset internal cache in shareReplay
     */
    resetDelay?: number;
    replayCount?: number;
    /**
     * Delay on remove from bank after ref count equal zero
     */
    removeDelay?: number;
    removeUnsubscribedDelay?: number;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function dedobs<Fn extends (...args: any[]) => Observable<any>>(
    fn: Fn,
    options: TDedobsOptions<Parameters<Fn>> = EMPTY_OBJECT,
) {
    const normalize = options.normalize ?? DEFAULT_NORMALIZER;
    const getObsCache = getObsCacheFactory<ReturnType<Fn>>(
        options.replayCount ?? 1,
        options.resetDelay,
        options.removeDelay,
        options.removeUnsubscribedDelay ?? DEFAULT_REMOVE_UNSUBSCRIBED_DELAY,
    );

    return (...args: Parameters<Fn>): ReturnType<Fn> => {
        const key = normalize(args);
        return key === DEDOBS_SKIP_KEY
            ? (fn(...args) as ReturnType<Fn>)
            : getObsCache(key, () => fn(...args) as ReturnType<Fn>);
    };
}

export function constantNormalizer() {
    return 0;
}

function getObsCacheFactory<Obs extends Observable<any>>(
    replayCount: number,
    resetDelay: undefined | number,
    removeDelay: undefined | number,
    removeUnsubscribedDelay: number,
) {
    const mapKeyToCache = new Map<TKey, { refCount: number; obs: Obs }>();

    const removeIfDerelict = (key: TKey) => {
        const cache = mapKeyToCache.get(key);
        if (cache !== undefined && cache.refCount === 0) {
            mapKeyToCache.delete(key);
        }
    };
    const removeCache =
        removeDelay === undefined || removeDelay === 0 || !isFinite(removeDelay)
            ? removeIfDerelict
            : debounceBy(removeIfDerelict, ([key]) => ({
                group: key,
                delay: removeDelay,
            }));
    const onSubscribe = (key: TKey, cb: VoidFunction) => {
        const cache = mapKeyToCache.get(key);
        if (cache !== undefined) {
            cache.refCount++;
            cb();
        }
    };
    const onFinalize = (key: TKey) => {
        const cache = mapKeyToCache.get(key);
        if (cache !== undefined) {
            cache.refCount--;
            cache.refCount === 0 && removeCache(key);
        }
    };
    const createCache = (key: TKey, obs: Obs) => {
        // If anyone doesn't sub during some time we remove cache
        const stop = macroTasks.addTimeout(() => removeIfDerelict(key), removeUnsubscribedDelay);

        return {
            refCount: 0,
            obs: obs.pipe(
                resetDelay === undefined
                    ? identity
                    : resetDelay === 0
                        ? shareReplayWithImmediateReset(replayCount)
                        : shareReplayWithDelayedReset(resetDelay, replayCount),
                tap({
                    subscribe: onSubscribe.bind(null, key, stop),
                    finalize: onFinalize.bind(null, key),
                }),
            ) as Obs,
        };
    };

    return (key: TKey, getObs: () => Obs) => {
        if (!mapKeyToCache.has(key)) {
            mapKeyToCache.set(key, createCache(key, getObs()));
        }

        const value = mapKeyToCache.get(key)!;

        return value.obs;
    };
}
