/// <reference types="vite/client" />

interface ImportMetaEnv {
    readonly ABLY_API_KEY: string
    readonly SUPABASE_URL: string
    readonly SUPABASE_PUBLICK_KEY: string
    readonly SUPABASE_BUCKET_MODELS: string
    readonly SUPABASE_BUCKET_EXP_BATCHES: string
}

interface ImportMeta {
    readonly env: ImportMetaEnv
}
