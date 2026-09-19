-- Multiple gallery photos per wine. Safe to re-run.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
        WHERE i.indrelid = 'wine_embeddings'::regclass
          AND i.indisprimary
        GROUP BY i.indexrelid
        HAVING COUNT(*) = 1 AND MIN(a.attname) = 'slug'
    ) THEN
        ALTER TABLE wine_embeddings DROP CONSTRAINT wine_embeddings_pkey;
        ALTER TABLE wine_embeddings ADD PRIMARY KEY (slug, image_hash);
    END IF;
END $$;
