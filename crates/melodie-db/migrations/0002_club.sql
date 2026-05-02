-- Club proposals: friends flag a clip as worth archiving on the operator's
-- personal server. The operator reviews them in the admin UI; nothing here
-- copies bytes — that's a separate (offline) sync step.

CREATE TABLE club_proposals (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  clip_id     TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
  user_id     TEXT NOT NULL REFERENCES users(id),
  note        TEXT,
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  UNIQUE (clip_id, user_id)
);

CREATE INDEX idx_club_proposals_clip ON club_proposals(clip_id);
CREATE INDEX idx_club_proposals_created ON club_proposals(created_at DESC);
