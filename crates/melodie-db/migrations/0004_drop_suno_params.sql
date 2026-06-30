-- Align the songs table with the local HeartMuLa engine.
--
-- HeartMuLa's only generation inputs are lyrics, styles (stored as `tags`),
-- language, and numeric knobs handled inside the engine crate. The Suno-only
-- knobs below have no analogue, and every song is now "custom" (there is no
-- describe mode), so `mode` is gone too. A `language` column is added — it
-- becomes the first, lowercased tag the engine sings against.
--
-- SQLite supports ALTER TABLE ... DROP COLUMN (>= 3.35.0).
ALTER TABLE songs DROP COLUMN exclude_tags;
ALTER TABLE songs DROP COLUMN vocal;
ALTER TABLE songs DROP COLUMN weirdness;
ALTER TABLE songs DROP COLUMN style_inf;
ALTER TABLE songs DROP COLUMN variation;
ALTER TABLE songs DROP COLUMN mode;
ALTER TABLE songs ADD COLUMN language TEXT NOT NULL DEFAULT 'english';
