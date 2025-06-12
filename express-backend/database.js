// Initializes and manages the SQLite database connection and schema.

import sqlite3 from 'sqlite3';
import { open } from 'sqlite';
import 'dotenv/config';

let db;

/**
 * Initializes the database connection and creates tables if they don't exist.
 * @returns {Promise<import('sqlite').Database>} A promise that resolves to the database instance.
 */
export async function initializeDatabase() {
  if (db) {
    return db;
  }

  try {
    const dbPath = process.env.DATABASE_PATH || './osint_database.sqlite';
    console.log(`Initializing database at: ${dbPath}`);

    db = await open({
      filename: dbPath,
      driver: sqlite3.Database,
    });

    await db.serialize(async () => {
      console.log('Creating database schema if not exists...');
      await db.exec(`
        CREATE TABLE IF NOT EXISTS ResearchCases (
          case_id TEXT PRIMARY KEY,
          topic TEXT NOT NULL,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
      `);
      await db.exec(`
        CREATE TABLE IF NOT EXISTS AgentRuns (
          run_id TEXT PRIMARY KEY,
          case_id TEXT NOT NULL,
          synthesized_findings TEXT,
          user_approved BOOLEAN,
          user_feedback TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(case_id) REFERENCES ResearchCases(case_id)
        );
      `);
      console.log('Database schema is ready.');
    });

    return db;
  } catch (err) {
    console.error('Failed to initialize database:', err);
    process.exit(1);
  }
}

/**
 * @returns {import('sqlite').Database} The database instance.
 */
export function getDb() {
  if (!db) {
    throw new Error('Database not initialized. Call initializeDatabase first.');
  }
  return db;
}