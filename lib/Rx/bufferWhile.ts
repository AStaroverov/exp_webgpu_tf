import { Observable } from 'rxjs';

export function bufferWhile<T>(predicate: (buffer: T[], value: T) => boolean) {
    return (source: Observable<T>) =>
        new Observable<T[]>(observer => {
            let buffer: T[] = [];

            const subscription = source.subscribe({
                next(value) {
                    buffer.push(value);

                    if (!predicate(buffer, value) && buffer.length > 0) {
                        observer.next(buffer);
                        buffer = [];
                    }
                },
                error(err) {
                    observer.error(err);
                },
                complete() {
                    if (buffer.length > 0) {
                        observer.next(buffer);
                    }
                    observer.complete();
                },
            });

            return () => subscription.unsubscribe();
        });
}
