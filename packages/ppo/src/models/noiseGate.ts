/**
 * Default predicate: whether a given layer should apply noise during a forward pass.
 * Currently always-on; specific domains can substitute their own predicate when
 * threaded through factories, but the generic ppo training step uses this default.
 */
export const shouldNoiseLayer = (_name: string, _iteration?: number): boolean => {
    return true;
};
