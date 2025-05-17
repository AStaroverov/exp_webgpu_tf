export class OrnsteinUhlenbeckNoise {
    private xPrev: number;
    private readonly theta: number;
    private readonly mu: number;
    private readonly sigma: number;
    private readonly dt: number;

    constructor(mu = 0, theta = 0.15, sigma = 0.2, dt = 1) {
        this.mu = mu;
        this.theta = theta;
        this.sigma = sigma;
        this.dt = dt;
        this.xPrev = mu;
    }

    // Box–Muller для N(0,1)
    private static gaussian(): number {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    next(): number {
        const dx = this.theta * (this.mu - this.xPrev) * this.dt
            + this.sigma * Math.sqrt(this.dt) * OrnsteinUhlenbeckNoise.gaussian();
        this.xPrev += dx;
        return this.xPrev;
    }
}
