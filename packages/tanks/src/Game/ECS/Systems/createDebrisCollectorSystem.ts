import { Ball, Collider } from '@dimforge/rapier2d-simd';
import { EntityId, hasComponent } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { PlayerEnvDI } from '../../DI/PlayerEnvDI.ts';
import { Debris } from '../Components/Debris.ts';
import { getEntityIdByPhysicalId, RigidBodyState } from '../Components/Physical.ts';
import { Tank } from '../Components/Tank.ts';
import { Vehicle } from '../Components/Vehicle.ts';
import { CollisionGroup, createCollisionGroups } from '../../Physical/createRigid.ts';
import { HeuristicsData } from '../Components/HeuristicsData.ts';
import { getTankTotalSlotCount, getTankCurrentPartsCount } from '../Entities/Tank/TankUtils.ts';
import { spawnSoundAtPosition, SoundType } from './Sound/index.ts';
import { fillSlot, findFirstEmptySlot, getEmptySlotsCount } from '../Entities/Vehicle/VehicleParts.ts';
import { mutatedVehicleOptions, resetOptions } from '../Entities/Vehicle/Options.ts';
import { PlayerRef } from '../Components/PlayerRef.ts';
import { TeamRef } from '../Components/TeamRef.ts';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.ts';
import { Color } from 'renderer/src/ECS/Components/Common.ts';

const COLLECTION_RADIUS = 50;
const DEBRIS_FOR_HEAL = 5;

export function createDebrisCollectorSystem({ world, physicalWorld } = GameDI) {
    let lastCollectionTime = 0;
    return (_delta: number) => {
        const currentTime = performance.now();
        if (currentTime - lastCollectionTime < 150) return;
        lastCollectionTime = currentTime;

        const playerTankEid = PlayerEnvDI.tankEid;
        if (playerTankEid == null) return;

        const emptySlotsCount = getEmptySlotsCount(playerTankEid);
        if (emptySlotsCount === 0) return;

        // Check if player vehicle exists
        if (!hasComponent(world, playerTankEid, Vehicle)) return;

        // Check if tank has empty slots
        const totalSlots = getTankTotalSlotCount(playerTankEid);
        const filledSlots = getTankCurrentPartsCount(playerTankEid);
        if (filledSlots >= totalSlots) return; // All slots filled, no need to collect

        // Get player position
        const playerX = RigidBodyState.position.get(playerTankEid, 0);
        const playerY = RigidBodyState.position.get(playerTankEid, 1);
        const playerRotation = RigidBodyState.rotation[playerTankEid];

        // Get approximate collision radius of player tank
        const colliderRadius = HeuristicsData.approxColliderRadius[playerTankEid] || COLLECTION_RADIUS;
        const searchRadius = colliderRadius * 1.5;
        
        // Get player's team to filter debris
        const playerTeamId = TeamRef.id[playerTankEid];

        // Find nearby debris from enemy teams
        const debrisEids: EntityId[] = [];

        physicalWorld.intersectionsWithShape(
            { x: playerX, y: playerY },
            playerRotation,
            new Ball(searchRadius),
            (collider: Collider) => {
                const eid = getEntityIdByPhysicalId(collider.handle);
                if (eid === 0 || eid === playerTankEid) return true;

                // Check if this entity is debris
                if (hasComponent(world, eid, Debris)) {
                    // Only collect debris from enemy teams
                    const debrisTeamId = TeamRef.id[eid];
                    if (debrisTeamId !== playerTeamId) {
                        debrisEids.push(eid);
                    }
                    if (debrisEids.length >= emptySlotsCount * DEBRIS_FOR_HEAL) {
                        return false; // Stop searching
                    }
                }

                return true;
            },
            undefined,
            // Look for parts that are not part of tanks anymore
            createCollisionGroups(CollisionGroup.ALL, CollisionGroup.VEHICALE_HULL_PARTS | CollisionGroup.TANK_TURRET_HEAD_PARTS),
        );

        if (debrisEids.length === 0) return;
        debrisEids.length = debrisEids.length - (debrisEids.length % DEBRIS_FOR_HEAL);

        // Collect debris and fill an empty slot
        fillSlots(playerTankEid, debrisEids);
        collectDebris(debrisEids);

        // Spawn collection sound entity at player position
        spawnSoundAtPosition({
            type: SoundType.DebrisCollect,
            x: playerX,
            y: playerY,
            volume: 1,
            destroyOnFinish: true,
        });
    };
}

function fillSlots(vehicleEid: number, debrisEids: EntityId[]) {
    const turretEid = Tank.turretEId[vehicleEid];
    const options = resetOptions(mutatedVehicleOptions);

    options.playerId = PlayerRef.id[vehicleEid];
    options.teamId = TeamRef.id[vehicleEid];
    options.x = RigidBodyState.position.get(vehicleEid, 0);
    options.y = RigidBodyState.position.get(vehicleEid, 1);
    options.rotation = RigidBodyState.rotation[vehicleEid];

    for (let i = 0; i < Math.floor(debrisEids.length / DEBRIS_FOR_HEAL); i++) {
        const emptySlotEid = findFirstEmptySlot(vehicleEid) ?? findFirstEmptySlot(turretEid);        
        if (emptySlotEid === null) break;
        mixColors(emptySlotEid, debrisEids.slice(i * DEBRIS_FOR_HEAL, i * DEBRIS_FOR_HEAL + DEBRIS_FOR_HEAL));
        fillSlot(emptySlotEid, options);
    }
}

function mixColors(baseColorEid: EntityId, mixEids: EntityId[]) {
    let r = Color.getR(baseColorEid) * 3;
    let g = Color.getG(baseColorEid) * 3;
    let b = Color.getB(baseColorEid) * 3;
    for (let i = 0; i < mixEids.length; i++) {
        r += Color.getR(mixEids[i]);
        g += Color.getG(mixEids[i]);
        b += Color.getB(mixEids[i]);
    }
    Color.setR$(baseColorEid, r / (DEBRIS_FOR_HEAL + 4));
    Color.setG$(baseColorEid, g / (DEBRIS_FOR_HEAL + 4));
    Color.setB$(baseColorEid, b / (DEBRIS_FOR_HEAL + 4));
}


function collectDebris(debrisEids: EntityId[]) {
    for (const debrisEid of debrisEids) {
        scheduleRemoveEntity(debrisEid);
    }
}
