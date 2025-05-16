import { once } from 'lodash-es';
import { defer, Observable } from 'rxjs';
import { tap } from 'rxjs/operators';

export function tapError<E extends Error>(fn: (err: E) => unknown) {
    return <T>(source: Observable<T>): Observable<T> =>
        defer(() => {
            return source.pipe(tap({ error: fn }));
        });
}

export function tapOnce<T>(fn: (value: T) => unknown) {
    return (source: Observable<T>): Observable<T> =>
        defer(() => {
            const onceFn = once((payload: T) => fn(payload));
            return source.pipe(tap(onceFn));
        });
}

export function tapOnceAny<T>(fn: () => unknown) {
    return (source: Observable<T>): Observable<T> =>
        defer(() => {
            const onceFn = once(fn);
            return source.pipe(
                tap({
                    next: onceFn,
                    error: onceFn,
                    complete: onceFn,
                }),
            );
        });
}
