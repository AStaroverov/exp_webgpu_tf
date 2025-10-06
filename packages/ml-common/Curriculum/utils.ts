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

    // Считаем общее здоровье для нормализации
    const totalInitialHealth = teamIds.reduce((sum, id) => sum + initialHealth[id], 0);
    const activeInitialHealthRatio = initialHealth[activeTeam] / totalInitialHealth;

    // Текущие соотношения здоровья
    const activeRatio = (currentHealth[activeTeam] ?? 0) / initialHealth[activeTeam];
    const opponentAvg = opponentIds.reduce((sum, id) => sum + (currentHealth[id] ?? 0) / initialHealth[id], 0) / opponentIds.length;

    if (Number.isNaN(activeRatio) || Number.isNaN(opponentAvg)) {
        console.error(`[getSuccessRatio] activeRatio: ${activeRatio}, opponentAvg: ${opponentAvg}`);
    }

    // Базовая разница в соотношениях здоровья (от -1 до 1)
    const healthDiff = activeRatio - opponentAvg;

    // Корректируем с учетом начального дисбаланса
    // При меньшинстве (< 0.5) усиливаем положительные результаты и смягчаем отрицательные
    // При большинстве (> 0.5) наоборот
    const balanceFactor = 2 * activeInitialHealthRatio; // 0.5 при меньшинстве, 1.0 при равенстве, 1.5+ при большинстве

    const adjusted = healthDiff / balanceFactor;

    return Math.max(-1, Math.min(1, adjusted));
}