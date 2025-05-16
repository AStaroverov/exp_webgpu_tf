import { hashString } from './hashString';
import { randomId } from './random.ts';

export function shallowHash(...values: unknown[]): number {
    let str = '';

    for (let i = 0; i < values.length; i += 1) {
        const value = values[i];

        if (value === undefined) {
            str += '/undefined';
        } else if (value === null) {
            str += '/null';
        } else if (typeof value === 'object' || typeof value === 'function') {
            str += '/' + getIdForPointer(value as object);
        } else {
            // don't forget about value == empty string
            str += '/' + String(value);
        }
    }

    return hashString(str);
}

const mapPointerToMap = new WeakMap<object, string>();

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getIdForPointer(obj: object | ((...args: any[]) => any)): string {
    if (!mapPointerToMap.has(obj)) {
        mapPointerToMap.set(obj, randomId());
    }

    return mapPointerToMap.get(obj)!;
}
