-- Suno has been removed; the local HeartMuLa engine is the only generator.
-- Drop the single-row Suno auth/session table. It held only transient upstream
-- credentials (no user data), so dropping it loses nothing recoverable.
DROP TABLE IF EXISTS suno_session;
