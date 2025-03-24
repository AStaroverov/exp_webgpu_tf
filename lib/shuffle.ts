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
