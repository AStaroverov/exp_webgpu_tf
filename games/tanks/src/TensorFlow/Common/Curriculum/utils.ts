import { getTankHealth, getTankTeamId } from '../../../Game/ECS/Entities/Tank/TankUtils.ts';

export function getTeamHealth(tanks: number[]) {
    return tanks.reduce((acc, tankEid) => {
        const team = getTankTeamId(tankEid);
        const health = getTankHealth(tankEid);
        acc[team] = (acc[team] || 0) + health;
        return acc;
    }, {} as Record<number, number>);
}

export function getSuccessRatio(activeTeam: number, initialTeamHealth: Record<number, number>, currentTeamHealth: Record<number, number>) {
    const teamCount = Object.keys(initialTeamHealth).length;
    const successRatio = Object.entries(currentTeamHealth)
        .map(([k, v]) => [Number(k), v])
        .reduce((acc, [teamId, health]) => {
            const delta = activeTeam === Number(teamId)
                ? (health / initialTeamHealth[teamId])
                : 1 - (health / initialTeamHealth[teamId]);
            return acc + delta;
        }, 0);

    return successRatio / teamCount;
}