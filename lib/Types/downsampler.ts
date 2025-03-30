type Series<T> = T[];
type Threshold = number;

const floor = Math.floor;
const abs = Math.abs;

export function downsample<T>(series: Series<T>, threshold: Threshold, getX: (item: T) => number, getY: (item: T) => number): Series<T> {
    const seriesLength: number = series.length;

    if (threshold >= seriesLength || threshold <= 0) {
        return series;
    }

    const sampled: Series<T> = [];
    let sampledIndex = 0;

    const bucketSize = (seriesLength - 2) / (threshold - 2);

    let initialPointInTriangle = 0;
    let maxAreaPoint: T;
    let maxArea: number;
    let area: number;
    let nextPointInTriangle: number;

    sampled[sampledIndex++] = series[initialPointInTriangle]; // Always add the first point

    for (let i = 0; i < threshold - 2; i++) {
        // Calculate point average for next bucket (containing c)
        let averageX = 0;
        let averageY = 0;
        let avgRangeStart = floor((i + 1) * bucketSize) + 1;
        let avgRangeEnd = floor((i + 2) * bucketSize) + 1;
        avgRangeEnd = avgRangeEnd < seriesLength ? avgRangeEnd : seriesLength;

        const avgRangeLength = avgRangeEnd - avgRangeStart;

        for (; avgRangeStart < avgRangeEnd; avgRangeStart++) {
            averageX += getX(series[avgRangeStart]);
            averageY += getY(series[avgRangeStart]);
        }
        averageX /= avgRangeLength;
        averageY /= avgRangeLength;

        // Get range for bucket
        let rangeOffs = floor((i + 0) * bucketSize) + 1;
        const rangeTo = floor((i + 1) * bucketSize) + 1;

        // Point of triangle
        const pointTriangleX = getX(series[initialPointInTriangle]);
        const pointTriangleY = getY(series[initialPointInTriangle]);

        maxArea = area = -1;

        for (; rangeOffs < rangeTo; rangeOffs++) {
            // Calculate triangle area over three buckets
            area =
                abs(
                    (pointTriangleX - averageX) *
                    (getY(series[rangeOffs]) - pointTriangleY) -
                    (pointTriangleX - getX(series[rangeOffs])) *
                    (averageY - pointTriangleY),
                ) * 0.5;
            if (area > maxArea) {
                maxArea = area;
                maxAreaPoint = series[rangeOffs];
                nextPointInTriangle = rangeOffs;
            }
        }

        // @ts-ignore
        sampled[sampledIndex++] = maxAreaPoint; // Pick this point from the bucket
        // @ts-ignore
        initialPointInTriangle = nextPointInTriangle; // This a is the next a (chosen b)
    }

    sampled[sampledIndex++] = series[seriesLength - 1]; // Always add last

    return sampled;
}