import { Observable } from 'rxjs';

export function getValue<T>(obs$: Observable<T>): T {
    let value;
    let received = false;
    const sub = obs$.subscribe((v) => {
        value = v;
        received = true;
    });
    sub.unsubscribe();

    if (!received) {
        throw new Error(`Value cannot be received synchronously`);
    }

    return value!;
}