export type TNode<ID, V> = {
    id: ID;
    value: V;
    prevId: null | ID;
    nextId: null | ID;
};

function create<ID, V>(id: ID, value: V, prev?: ID, next?: ID): TNode<ID, V> {
    return {
        id,
        value,
        prevId: prev || null,
        nextId: next || null,
    };
}

function setValue<ID, V>(node: TNode<ID, V>, value: V): void {
    node.value = value;
}

function setNextId<ID, V>(node: TNode<ID, V>, id: null | ID): void {
    node.nextId = id || null;
}

function setPrevId<ID, V>(node: TNode<ID, V>, id: null | ID): void {
    node.prevId = id || null;
}

export const Node = {
    create,
    setValue,
    setNextId,
    setPrevId,
};
