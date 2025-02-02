export type TPrimitive = string | number | boolean | null | undefined;

export type Assign<A, B> = Omit<A, keyof B> & B;

export type Nil = null | undefined;

export type Opaque<Type, BaseType> = BaseType & {
    readonly __type__: Type;
    readonly __baseType__: BaseType;
};

export type ValueOf<T> = T[keyof T];
