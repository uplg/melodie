-- Sessions moved to an in-memory store (see router.rs); the SQLite-backed
-- store previously created this table itself, outside this migration set.
-- Drop it — it held only opaque session tokens, nothing recoverable.
DROP TABLE IF EXISTS tower_sessions;
