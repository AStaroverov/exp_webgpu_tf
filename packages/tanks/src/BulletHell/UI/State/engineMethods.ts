import { PI } from '../../../../../../lib/math.ts';
import { randomRangeFloat } from '../../../../../../lib/random.ts';
import { EntityId, innerQuery } from 'bitecs';
import { Tank, TankType } from '../../../Game/ECS/Components/Tank.ts';
import { engine$, getEngine } from './engine.ts';
import { createTank } from '../../../Game/ECS/Entities/Tank/createTank.ts';
import { createPlayer } from '../../../Game/ECS/Entities/Player.ts';
import { map, shareReplay } from 'rxjs';
import { getValue } from '../../../../../../lib/Rx/getValue.ts';
import { BULLET_HELL_MAP_SIZE } from './def.ts';

export const BULLET_HELL_MAX_ENEMIES = 10;

export const playerId$ = engine$.pipe(
    map((e) => createPlayer(0, e)),
    shareReplay(1),
);

// Player team = 0, Enemy team = 1
export const PLAYER_TEAM_ID = 0;
export const ENEMY_TEAM_ID = 1;

export function addPlayerTank(tankType: TankType = TankType.Light) {
    // Player spawns at bottom center
    const x = BULLET_HELL_MAP_SIZE / 2;
    const y = BULLET_HELL_MAP_SIZE - 150;

    return createTank({
        type: tankType,
        playerId: getValue(playerId$),
        teamId: PLAYER_TEAM_ID,
        x,
        y,
        rotation: -PI / 2, // Facing up
        color: [0.2, 0.8, 0.2, 1], // Green player
    });
}

export function addEnemyTank(slot: number, tankType: TankType = TankType.Light) {
    // Enemies spawn at top, spread horizontally
    const padding = 100;
    const spacing = (BULLET_HELL_MAP_SIZE - padding * 2) / (BULLET_HELL_MAX_ENEMIES - 1);
    const x = padding + slot * spacing;
    const y = 150;

    return createTank({
        type: tankType,
        playerId: getValue(playerId$),
        teamId: ENEMY_TEAM_ID,
        x,
        y,
        rotation: PI / 2 + randomRangeFloat(-PI / 6, PI / 6), // Facing down with slight variation
        color: [0.8, 0.2, 0.2, 1], // Red enemies
    });
}

export function getTankEids() {
    return innerQuery(getEngine().world, [Tank]);
}

export function getPlayerTankEids() {
    const allTanks = Array.from(getTankEids());
    // Filter by team - this would need TeamRef component check
    // For now just return first tank as player
    return allTanks.slice(0, 1);
}

export function getEnemyTankEids() {
    const allTanks = Array.from(getTankEids());
    return allTanks.slice(1);
}

export function getTankType(tankEid: EntityId) {
    return Tank.type[tankEid] as TankType;
}
