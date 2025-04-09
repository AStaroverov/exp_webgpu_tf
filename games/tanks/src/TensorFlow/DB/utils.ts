export async function upsert<T extends {}>(db: PouchDB.Database<T>, id: string, data: Partial<T>) {
    try {
        const existing = await db.get(id);
        return db.put({ ...(existing as any), ...data, _id: id, _rev: existing._rev });
    } catch (err: any) {
        if (err.status === 404) {
            return db.put({ ...(data as any), _id: id });
        } else {
            throw err;
        }
    }
}