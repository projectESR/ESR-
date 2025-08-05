-- schema.sql
-- Defines the structure for the reports database table.

CREATE TABLE IF NOT EXISTS reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,
  blood_type TEXT NOT NULL,
  image_b64 TEXT NOT NULL,
  analysis_json TEXT NOT NULL
);
