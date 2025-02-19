import { ObservableHook, observe, World } from 'bitecs';

const detectorsMap = new Map<World, Set<ReturnType<typeof createChangedDetector>>>();

export function createChangedDetector(world: World, hooks: ObservableHook[]) {
    const detectors = detectorsMap.get(world);

    if (!detectors) {
        throw new Error('createChangedDetector called with a world that is not registered');
    }

    const value = new Set<number>();
    const stops = hooks.map((hook) => observe(world, hook, (eid) => value.add(eid)));
    const stop = () => stops.forEach((s) => s());
    const hasChanges = () => value.size > 0;
    const has = (eid: number) => value.has(eid);
    const clear = () => value.clear();
    const destroy = () => {
        stop();
        value.clear();
        detectors.delete(inst);
    };

    const inst = { value, has, hasChanges, clear, destroy };

    detectors.add(inst);

    return inst;
}

export function createChangedDetectorSystem(world: World) {
    const detectors = new Set<ReturnType<typeof createChangedDetector>>();
    detectorsMap.set(world, detectors);
    return () => {
        detectors.forEach((detector) => detector.clear());
    };
}

export function destroyChangedDetectorSystem(world: World) {
    const detectors = detectorsMap.get(world);

    if (!detectors) {
        throw new Error('destroyChangedDetectorSystem called with a world that is not registered');
    }

    detectors.forEach((detector) => detector.destroy());
    detectorsMap.delete(world);
}