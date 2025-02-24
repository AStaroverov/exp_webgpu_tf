import { EntityId, ObservableHook, observe, World } from 'bitecs';

const detectorsMap = new Map<World, Set<ReturnType<typeof createChangeDetector>>>();

export function createChangeDetector(world: World, hooks: ObservableHook[]) {
    const detectors = detectorsMap.get(world);

    if (!detectors) {
        throw new Error('createChangeDetector called with a world that is not registered');
    }

    const value = new Set<EntityId>();
    const stops = hooks.map((hook) => observe(world, hook, (eid: EntityId) => value.add(eid)));
    const stop = () => stops.forEach((s) => s());
    const hasChanges = () => value.size > 0;
    const has = (eid: EntityId) => value.has(eid);
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

export function createChangeDetectorSystem(world: World) {
    if (detectorsMap.has(world)) {
        throw new Error('createChangeDetectorSystem called with a world that is already registered');
    }

    const detectors = new Set<ReturnType<typeof createChangeDetector>>();
    detectorsMap.set(world, detectors);
    return () => {
        detectors.forEach((detector) => detector.clear());
    };
}

export function destroyChangeDetectorSystem(world: World) {
    const detectors = detectorsMap.get(world);

    if (!detectors) {
        throw new Error('destroyChangedDetectorSystem called with a world that is not registered');
    }

    detectors.forEach((detector) => detector.destroy());
    detectorsMap.delete(world);
}