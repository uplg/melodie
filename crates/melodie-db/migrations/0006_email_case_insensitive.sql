-- Belt-and-suspenders: app code now lowercases emails before every
-- write/lookup, but SQLite's default TEXT collation is case-sensitive, so
-- this index is the actual backstop against ever storing two accounts that
-- differ only by case.
CREATE UNIQUE INDEX idx_users_email_lower ON users(lower(email));
