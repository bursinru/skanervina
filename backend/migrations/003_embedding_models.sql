-- One gallery image is embedded by several encoders (224 and 384 px). Safe to re-run.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
        WHERE i.indrelid = 'wine_embeddings'::regclass
          AND i.indisprimary
          AND a.attname = 'model'
    ) THEN
        ALTER TABLE wine_embeddings DROP CONSTRAINT wine_embeddings_pkey;
        ALTER TABLE wine_embeddings ADD PRIMARY KEY (slug, model, image_hash);
    END IF;
END $$;
-- The encoders share an embedding space, so one index filtered afterwards by
-- model would lose neighbours. Queries inline the model to use these.
CREATE INDEX IF NOT EXISTS wine_embedding_cosine_224 ON wine_embeddings
    USING hnsw (embedding vector_cosine_ops) WHERE model = 'google/siglip2-base-patch16-224';
CREATE INDEX IF NOT EXISTS wine_embedding_cosine_384 ON wine_embeddings
    USING hnsw (embedding vector_cosine_ops) WHERE model = 'google/siglip2-base-patch16-384';
