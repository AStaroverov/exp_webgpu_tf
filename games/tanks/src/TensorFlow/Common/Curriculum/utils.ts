import { getTankHealth, getTankTeamId } from '../../../Game/ECS/Entities/Tank/TankUtils.ts';

export function getTeamHealth(tanks: number[]) {
    return tanks.reduce((acc, tankEid) => {
        const team = getTankTeamId(tankEid);
        const health = getTankHealth(tankEid);
        acc[team] = (acc[team] || 0) + health;
        return acc;
    }, {} as Record<number, number>);
}

export function getSuccessRatio(
    activeTeam: number,
    initialHealth: Record<number, number>,
    currentHealth: Record<number, number>,
): number {
    const teamIds = Object.keys(initialHealth).map(Number);
    const opponentIds = teamIds.filter(id => id !== activeTeam);
    const activeRatio = currentHealth[activeTeam] / initialHealth[activeTeam];
    const opponentAvg = opponentIds.reduce((sum, id) => sum + currentHealth[id] / initialHealth[id], 0) / opponentIds.length;

    return activeRatio - opponentAvg;
}