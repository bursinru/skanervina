-- Anonymous public feed: only the matched catalog slug and time, no photo or visitor id.
CREATE TABLE IF NOT EXISTS scan_events (
    id BIGSERIAL PRIMARY KEY,
    slug TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS scan_events_created_at ON scan_events (created_at DESC);
