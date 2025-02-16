interface ArrayLike<T> {
    length: number;
    [n: number]: T;
}

export function shuffle<T extends ArrayLike<any>>(array: T): T {
    let m = array.length,
        t,
        i;

    // While there remain elements to shuffle…
    while (m > 0) {
        // Pick a remaining element…
        i = (Math.random() | 0) * m--;

        // And swap it with the current element.
        t = array[m];
        array[m] = array[i];
        array[i] = t;
    }

    return array;
}
