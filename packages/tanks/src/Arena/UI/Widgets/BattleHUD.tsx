import { CSSProperties } from 'react';
import { useObservable } from 'react-use';
import { playerScore$, getTankInfo$, tankEids$ } from '../../State/Game/gameMethods.ts';
import { PLAYER_TEAM_ID } from '../../State/Game/def.ts';

const defaultScore = { spice: 0, debris: 0 };

function TankHealthBar({ eid }: { eid: number }) {
    const tank = useObservable(getTankInfo$(eid));
    
    if (!tank) return null;
    
    const isAlly = tank.teamId === PLAYER_TEAM_ID;
    const healthPercent = Math.round(tank.health * 100);
    
    const getHealthColor = (health: number) => {
        if (health > 0.6) return 'bg-emerald-500';
        if (health > 0.3) return 'bg-amber-500';
        return 'bg-red-500';
    };

    const tankColor = `rgb(${Math.round(tank.color[0] * 255)}, ${Math.round(tank.color[1] * 255)}, ${Math.round(tank.color[2] * 255)})`;

    return (
        <div className="flex items-center gap-2">
            <div
                className="w-3 h-3 rounded-sm shadow-inner"
                style={{ backgroundColor: tankColor }}
            />
            
            <div className="relative flex-1 h-4 bg-black/40 rounded overflow-hidden backdrop-blur-sm">
                <div
                    className={`absolute inset-y-0 left-0 transition-all duration-200 ${getHealthColor(tank.health)}`}
                    style={{ width: `${healthPercent}%` }}
                />
                <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-[10px] font-bold text-white drop-shadow-[0_1px_1px_rgba(0,0,0,0.8)]">
                        {healthPercent}%
                    </span>
                </div>
            </div>
            
            <span className={`text-[10px] font-bold ${isAlly ? 'text-cyan-400' : 'text-red-400'}`}>
                {isAlly ? 'A' : 'E'}
            </span>
        </div>
    );
}

function AllyTankHealthBar({ eid }: { eid: number }) {
    const tank = useObservable(getTankInfo$(eid));
    if (!tank || tank.teamId !== PLAYER_TEAM_ID) return null;
    return <TankHealthBar eid={eid} />;
}

function EnemyTankHealthBar({ eid }: { eid: number }) {
    const tank = useObservable(getTankInfo$(eid));
    if (!tank || tank.teamId === PLAYER_TEAM_ID) return null;
    return <TankHealthBar eid={eid} />;
}

function ResourceCounter({ icon, value, color }: { icon: string; value: number; color: string }) {
    return (
        <div className={`flex items-center gap-1.5 px-2 py-1 rounded bg-black/30 backdrop-blur-sm ${color}`}>
            <span className="text-sm">{icon}</span>
            <span className="text-sm font-bold tabular-nums">{Math.round(value)}</span>
        </div>
    );
}

export function BattleHUD({ className, style }: { className?: string; style?: CSSProperties }) {
    const tankEids = useObservable(tankEids$, []);
    const playerScore = useObservable(playerScore$, defaultScore);

    return (
        <div className={`${className} pointer-events-none select-none`} style={style}>
            <div className="absolute top-4 left-1/2 -translate-x-1/2 flex gap-3">
                <ResourceCounter
                    icon="💎"
                    value={playerScore.spice}
                    color="text-orange-300"
                />
                <ResourceCounter
                    icon="🔩"
                    value={playerScore.debris}
                    color="text-slate-300"
                />
            </div>

            <div className="absolute left-4 top-16 w-40 space-y-1">
                <div className="text-xs font-bold text-cyan-400 uppercase tracking-wider mb-2 drop-shadow-lg">
                    Allies
                </div>
                {tankEids.map(eid => (
                    <AllyTankHealthBar key={eid} eid={eid} />
                ))}
            </div>

            <div className="absolute right-4 top-16 w-40 space-y-1">
                <div className="text-xs font-bold text-red-400 uppercase tracking-wider mb-2 drop-shadow-lg text-right">
                    Enemies
                </div>
                {tankEids.map(eid => (
                    <EnemyTankHealthBar key={eid} eid={eid} />
                ))}
            </div>
        </div>
    );
}
