import { getTankHealth, getTankTeamId } from '../../tanks/src/Game/ECS/Entities/Tank/TankUtils.ts';

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
    const activeRatio = (currentHealth[activeTeam] ?? 0) / initialHealth[activeTeam];
    const opponentAvg = opponentIds.reduce((sum, id) => sum + (currentHealth[id] ?? 0) / initialHealth[id], 0) / opponentIds.length;

    if (Number.isNaN(activeRatio) || Number.isNaN(opponentAvg)) {
        console.error(`[getSuccessRatio] activeRatio: ${activeRatio}, opponentAvg: ${opponentAvg}`);
    }

    return activeRatio - opponentAvg;
}