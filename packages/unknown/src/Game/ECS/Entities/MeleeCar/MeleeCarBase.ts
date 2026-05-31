import { PhysicsWorld } from '../../createPhysicsWorld.ts';
import { PhysicalWorld } from '../../../Physical/initPhysicalWorld.ts';
import { PI } from '../../../../../../../lib/math.ts';
import { WheelPosition } from '../../Components/Wheel.ts';
import { createWheel, WheelOptions } from '../Wheel/createWheel.ts';
import { createVehicleBase } from '../Vehicle/VehicleBase.ts';
import { type MeleeCarOptions } from './Options.ts';

export type MeleeCarWheelsConfig = {
    frontLeft: { x: number; y: number };
    frontRight: { x: number; y: number };
    rearLeft: { x: number; y: number };
    rearRight: { x: number; y: number };
    wheelWidth: number;
    wheelHeight: number;
    maxSteeringAngle?: number;
    steeringSpeed?: number;
};

export function createMeleeCarBase(world: PhysicsWorld, physicalWorld: PhysicalWorld, options: MeleeCarOptions): [number, number, number] {
    // MeleeCar doesn't have caterpillars or turret - just a simple ramming vehicle
    // No Tank component needed since there's no turret
    // Wheels are added as children via createMeleeCarWheels
    return createVehicleBase(world, physicalWorld, options);
}

/**
 * Creates 4 wheel entities for a melee car.
 * Front wheels are steerable, rear wheels are drive wheels.
 */
export function createMeleeCarWheels(
    world: PhysicsWorld,
    physicalWorld: PhysicalWorld,
    options: MeleeCarOptions,
    wheelsConfig: MeleeCarWheelsConfig,
    carRenderEid: number,
    carPid: number,
): [frontLeftRenderEid: number, frontRightRenderEid: number, rearLeftRenderEid: number, rearRightRenderEid: number] {
    const wheelOptions: WheelOptions = {
        ...options,
        width: wheelsConfig.wheelWidth,
        height: wheelsConfig.wheelHeight,
        density: options.density * 0.2,
        maxSteeringAngle: wheelsConfig.maxSteeringAngle ?? PI / 6,
        steeringSpeed: wheelsConfig.steeringSpeed ?? PI * 2,

        wheelPosition: WheelPosition.FrontLeft,
        anchorX: 9999999,
        anchorY: 9999999,
        isSteerable: false,
        isDrive: false,
    };

    wheelOptions.wheelPosition = WheelPosition.FrontLeft;
    wheelOptions.anchorX = wheelsConfig.frontLeft.x;
    wheelOptions.anchorY = wheelsConfig.frontLeft.y;
    wheelOptions.isSteerable = true;
    wheelOptions.isDrive = true;
    wheelOptions.x = options.x + wheelsConfig.frontLeft.x;
    wheelOptions.y = options.y + wheelsConfig.frontLeft.y;
    const [, frontLeftRenderEid] = createWheel(world, physicalWorld, wheelOptions, carRenderEid, carPid);

    // Front Right wheel - steerable
    wheelOptions.wheelPosition = WheelPosition.FrontRight;
    wheelOptions.anchorX = wheelsConfig.frontRight.x;
    wheelOptions.anchorY = wheelsConfig.frontRight.y;
    wheelOptions.isSteerable = true;
    wheelOptions.isDrive = true;
    wheelOptions.x = options.x + wheelsConfig.frontRight.x;
    wheelOptions.y = options.y + wheelsConfig.frontRight.y;
    const [, frontRightRenderEid] = createWheel(world, physicalWorld, wheelOptions, carRenderEid, carPid);

    // Rear Left wheel - drive wheel
    wheelOptions.wheelPosition = WheelPosition.RearLeft;
    wheelOptions.anchorX = wheelsConfig.rearLeft.x;
    wheelOptions.anchorY = wheelsConfig.rearLeft.y;
    wheelOptions.isSteerable = false;
    wheelOptions.isDrive = true;
    wheelOptions.x = options.x + wheelsConfig.rearLeft.x;
    wheelOptions.y = options.y + wheelsConfig.rearLeft.y;
    const [, rearLeftRenderEid] = createWheel(world, physicalWorld, wheelOptions, carRenderEid, carPid);

    // Rear Right wheel - drive wheel
    wheelOptions.wheelPosition = WheelPosition.RearRight;
    wheelOptions.anchorX = wheelsConfig.rearRight.x;
    wheelOptions.anchorY = wheelsConfig.rearRight.y;
    wheelOptions.isSteerable = false;
    wheelOptions.isDrive = true;
    wheelOptions.x = options.x + wheelsConfig.rearRight.x;
    wheelOptions.y = options.y + wheelsConfig.rearRight.y;
    const [, rearRightRenderEid] = createWheel(world, physicalWorld, wheelOptions, carRenderEid, carPid);

    return [frontLeftRenderEid, frontRightRenderEid, rearLeftRenderEid, rearRightRenderEid];
}

