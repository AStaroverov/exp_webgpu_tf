import { isUndefined, noop } from 'lodash-es';
import { useMemo, useSyncExternalStore } from 'react';
import { animationFrameScheduler, Observable, Subscription } from 'rxjs';
import { debounceTime } from 'rxjs/operators';
import { tapOnce } from '../Rx/tap.ts';
import { throwingError } from '../throwingError.ts';
import { useUnmount } from 'react-use';

export function useObservable<T>(obs$: Observable<T>): undefined | T;
export function useObservable<T>(obs$: Observable<T>, seed: T): T;
export function useObservable<T>(obs$: Observable<T>, seed?: T) {
    const store = useMemo(
        () => createStore(obs$, seed),
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [obs$],
    );
    // protect case, when store is not call subscribe + unsubscribe
    useUnmount(store.unsubscribe);
    // useEffect(() => store.unsubscribe);
    return useSyncExternalStore(store.subscribe, store.getSnapshot);
}

export function createStore<T>(obs$: Observable<T>, seed?: T) {
    let value = seed;
    let onChange: VoidFunction = noop;
    const update = (v: T) => {
        value = v;
        onChange();
    };
    const subscribe = () => {
        return obs$.pipe(tapOnce(update), debounceTime(0, animationFrameScheduler)).subscribe({
            next: update,
            error: throwingError,
        });
    };
    const unsubscribe = () => {
        subscription?.unsubscribe();
        onChange = noop;
        value = seed;
    };
    const getSnapshot = () => value;

    let subscription: undefined | Subscription = undefined;

    return {
        subscribe(onStoreChange: () => void) {
            onChange = onStoreChange;
            subscription =
                isUndefined(subscription) || subscription.closed ? subscribe() : subscription;

            if (value !== seed) onChange();

            return unsubscribe;
        },
        unsubscribe,
        getSnapshot,
    };
}
