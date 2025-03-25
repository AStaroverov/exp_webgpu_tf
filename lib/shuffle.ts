interface ArrayLike<T> {
    length: number;

    [n: number]: T;
}

export function shuffle<T extends ArrayLike<any>>(array: T): T {
    let m = array.length - 1,
        t,
        i;

    while (m > 0) {
        i = Math.round(Math.random() * m--);
        t = array[m];
        array[m] = array[i];
        array[i] = t;
    }

    return array;
}

export function batchShuffle<T extends (unknown[] | Float32Array | Float64Array)[]>(...arrays: T) {
    const l = arrays[0].length;

    if (arrays.some(a => a.length !== l)) {
        throw new Error('All arrays must have the same length');
    }

    let m = l - 1,
        t,
        i;

    while (m > 0) {
        i = Math.round(Math.random() * m--);
        for (let j = 0; j < arrays.length; j++) {
            t = arrays[j][m];
            arrays[j][m] = arrays[j][i];
            arrays[j][i] = t;
        }
    }
}
