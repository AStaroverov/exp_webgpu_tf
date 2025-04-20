export function binarySearch(low: number, high: number, comparator: (index: number) => number) {
    const min = low;
    const max = high;

    if (max < min) throw new RangeError('invalid bounds');

    let mid;
    let cmp;
    let absCmp;
    let diff = Infinity;
    let best = -1;

    while (low <= high) {
        // The naive `low + high >>> 1` could fail for array lengths > 2**31
        // because `>>>` converts its operands to int32. `low + (high - low >>> 1)`
        // works for array lengths <= 2**32-1 which is also Javascript's max array
        // length.
        mid = low + ((high - low) >>> 1);
        cmp = comparator(mid);
        absCmp = Math.abs(cmp);

        if (absCmp < diff) {
            diff = absCmp;
            best = mid;
        }

        // Too low.
        if (cmp < 0) low = mid + 1;
        // Too high.
        else if (cmp > 0) high = mid - 1;
        // Key found.
        else break;
    }

    return best;
}

