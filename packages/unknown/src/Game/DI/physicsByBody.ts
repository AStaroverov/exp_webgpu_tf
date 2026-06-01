// Rapier body handle -> physics atom eid. A handle is not an ECS entity, so this
// stays a plain map (replaces BridgeDI.physicalIdToPhysics).
const map = new Map<number, number>();
export const physicsByBody = {
    set(handle: number, physEid: number) { map.set(handle, physEid); },
    delete(handle: number) { map.delete(handle); },
    get(handle: number): number | undefined { return map.get(handle); },
    clear() { map.clear(); },
};
