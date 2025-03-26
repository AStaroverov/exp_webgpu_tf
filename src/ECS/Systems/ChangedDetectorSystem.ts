import { EntityId, ObservableHook, observe, World } from 'bitecs';

const detectorsMap = new Map<World, Set<ReturnType<typeof createChangeDetector>>>();

export function createChangeDetector(world: World, hooks: ObservableHook[]) {
    if (!detectorsMap.has(world)) {
        detectorsMap.set(world, new Set<ReturnType<typeof createChangeDetector>>());
    }

    const detectors = detectorsMap.get(world)!;
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

export function destroyChangeDetectorSystem(world: World) {
    const detectors = detectorsMap.get(world);

    if (!detectors) {
        throw new Error('destroyChangedDetectorSystem called with a world that is not registered');
    }

    detectors.forEach((detector) => detector.destroy());
    detectorsMap.delete(world);
}