export const GameSession = {
    initialTeamSize: new Map<number, number>(),

    reset() {
        GameSession.initialTeamSize.clear();
    },

    setInitialTeamSize(teamId: number) {
        GameSession.initialTeamSize.set(teamId, (GameSession.initialTeamSize.get(teamId) ?? 0) + 1);
    },

    getInitialTeamSize(teamId: number) {
        return GameSession.initialTeamSize.get(teamId) ?? 0;
    },
};
