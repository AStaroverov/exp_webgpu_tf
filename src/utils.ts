export function uniq<T>(arr: T[]): T[] {
    return Array.from(new Set(arr));
}

export function uniqBy<T>(arr: T[], predicate: (v: T) => unknown): T[] {
    return Array.from(arr.reduce((map, item) => {
        const key = (item === null || item === undefined)
            ? item
            : predicate(item);

        map.has(key) || map.set(key, item);

        return map;
    }, new Map()), ([, v]) => v);
}


export function isNil<T>(v: T | null | undefined): v is null | undefined {
    return v === null || v === undefined;
}

export function notNil<T>(v: T | null | undefined): v is T {
    return v !== null && v !== undefined;
}
