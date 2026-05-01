-- Initial schema for Melodie.

CREATE TABLE users (
  id            TEXT PRIMARY KEY,
  email         TEXT UNIQUE NOT NULL,
  display_name  TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  role          TEXT NOT NULL DEFAULT 'member',
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE relations (
  subject_type TEXT NOT NULL,
  subject_id   TEXT NOT NULL,
  relation     TEXT NOT NULL,
  object_type  TEXT NOT NULL,
  object_id    TEXT NOT NULL,
  PRIMARY KEY (subject_type, subject_id, relation, object_type, object_id)
);

CREATE TABLE invites (
  code        TEXT PRIMARY KEY,
  -- Bootstrap invites are inserted before any user exists; created_by is NULL
  -- in that case. Subsequent invites must reference the issuing admin.
  created_by  TEXT REFERENCES users(id),
  used_by     TEXT REFERENCES users(id),
  -- Role granted to the user who consumes this invite. Bootstrap invites
  -- grant 'admin'; admin-issued invites default to 'member'.
  role        TEXT NOT NULL DEFAULT 'member',
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  expires_at  TEXT
);

CREATE TABLE songs (
  id           TEXT PRIMARY KEY,
  owner_id     TEXT NOT NULL REFERENCES users(id),
  mode         TEXT NOT NULL,
  title        TEXT,
  tags         TEXT,
  exclude_tags TEXT,
  lyrics       TEXT,
  prompt       TEXT,
  vocal        TEXT,
  weirdness    INTEGER,
  style_inf    INTEGER,
  variation    TEXT,
  model        TEXT NOT NULL,
  status       TEXT NOT NULL,
  error        TEXT,
  created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  updated_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE clips (
  id            TEXT PRIMARY KEY,
  song_id       TEXT NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
  variant_index INTEGER NOT NULL,
  status        TEXT NOT NULL,
  duration_s    REAL,
  image_url     TEXT,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_songs_owner_created ON songs(owner_id, created_at DESC);
CREATE INDEX idx_clips_song ON clips(song_id);

CREATE TABLE generation_quota (
  user_id    TEXT NOT NULL REFERENCES users(id),
  day_utc    TEXT NOT NULL,
  count      INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (user_id, day_utc)
);

-- Single-row table: holds the (one) Suno session that backs every user's
-- generations. Stored in plaintext because Melodie runs locally on the
-- operator's machine; filesystem permissions on the SQLite file are the only
-- access control we rely on.
CREATE TABLE suno_session (
  id            INTEGER PRIMARY KEY CHECK (id = 1),
  jwt           TEXT,
  session_id    TEXT,
  device_id     TEXT,
  clerk_cookie  TEXT,
  last_check    TEXT,
  last_status   TEXT NOT NULL DEFAULT 'unknown'
);
INSERT INTO suno_session (id) VALUES (1);
