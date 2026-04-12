PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;

CREATE TABLE IF NOT EXISTS global_term_replacements (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  language TEXT NOT NULL,
  source_text TEXT NOT NULL,
  target_text TEXT NOT NULL,
  match_mode TEXT NOT NULL DEFAULT 'exact',
  priority INTEGER NOT NULL DEFAULT 100,
  is_enabled INTEGER NOT NULL DEFAULT 1,
  note TEXT DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_global_terms
ON global_term_replacements(language, source_text, target_text, match_mode);

CREATE TABLE IF NOT EXISTS homophone_index_state (
  project_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  status TEXT NOT NULL,
  last_committed_chunk INTEGER NOT NULL DEFAULT -1,
  heartbeat_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (project_id, revision)
);

CREATE TABLE IF NOT EXISTS homophone_postings (
  project_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  language TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  sentence_index INTEGER NOT NULL,
  token_index INTEGER NOT NULL,
  token_text TEXT NOT NULL,
  reading_key TEXT NOT NULL,
  reading_key_fuzzy TEXT NOT NULL,
  reading_key_no_punct TEXT NOT NULL,
  reading_key_fuzzy_no_punct TEXT NOT NULL,
  char_start INTEGER NOT NULL,
  char_end INTEGER NOT NULL,
  PRIMARY KEY (
    project_id, revision, sentence_index, token_index
  )
);

CREATE INDEX IF NOT EXISTS idx_posting_lookup_strict
ON homophone_postings(project_id, revision, language, reading_key, sentence_index, token_index);

CREATE INDEX IF NOT EXISTS idx_posting_lookup_fuzzy
ON homophone_postings(project_id, revision, language, reading_key_fuzzy, sentence_index, token_index);

CREATE INDEX IF NOT EXISTS idx_posting_lookup_strict_no_punct
ON homophone_postings(project_id, revision, language, reading_key_no_punct, sentence_index, token_index);

CREATE INDEX IF NOT EXISTS idx_posting_lookup_fuzzy_no_punct
ON homophone_postings(project_id, revision, language, reading_key_fuzzy_no_punct, sentence_index, token_index);

CREATE TABLE IF NOT EXISTS sentence_revision (
  project_id TEXT NOT NULL,
  sentence_index INTEGER NOT NULL,
  revision INTEGER NOT NULL,
  text_hash TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (project_id, sentence_index)
);

