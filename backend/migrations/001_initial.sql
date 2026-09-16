CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS scanner_profiles (
    id TEXT PRIMARY KEY,
    state JSONB NOT NULL DEFAULT '{"saved": [], "ratings": {}}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS wines (
    slug TEXT PRIMARY KEY,
    card JSONB NOT NULL,
    image_name TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS wine_embeddings (
    slug TEXT PRIMARY KEY REFERENCES wines(slug) ON DELETE CASCADE,
    model TEXT NOT NULL,
    image_hash TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS wine_embedding_cosine ON wine_embeddings USING hnsw (embedding vector_cosine_ops);
