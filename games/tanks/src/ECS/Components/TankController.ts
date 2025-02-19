import { addComponent, World } from 'bitecs';
import { createMethods, NestedArray, TypedArray } from '../../../../../src/utils.ts';
import { delegate } from '../../../../../src/delegate.ts';

export const TankController = {
    move: TypedArray.f64(delegate.defaultSize),
    rotation: TypedArray.f64(delegate.defaultSize),
    shot: TypedArray.i8(delegate.defaultSize),
    turretTarget: NestedArray.f64(2, delegate.defaultSize),
};

export const TankControllerMethods = createMethods(TankController, {
    addComponent: (world: World, id: number) => addComponent(world, id, TankController),
    setShot$(eid): void {
        TankController.shot[eid] = 1;
    },
    resetShot$(eid: number): void {
        TankController.shot[eid] = 0;
    },
    shouldShot(eid: number): boolean {
        return TankController.shot[eid] > 0;
    },
    setMove$(eid: number, dir: number): void {
        TankController.move[eid] = dir;
    },
    setRotate$(eid: number, dir: number): void {
        TankController.rotation[eid] = dir;
    },
    setTurretTarget$(eid: number, x: number, y: number): void {
        TankController.turretTarget.set(eid, 0, x);
        TankController.turretTarget.set(eid, 1, y);
    },
    getTurretTarget(eid: number): Float64Array {
        return TankController.turretTarget.getBatche(eid);
    },
});
