export type ValueOf<T> = T[keyof T];

export type Opaque<Type, BaseType> = BaseType & {
    readonly __type__: Type;
    readonly __baseType__: BaseType;
};
