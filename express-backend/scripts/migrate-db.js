#!/usr/bin/env node

// Database migration script for better-sqlite3 transition

import { initializeDatabase, getDb, dbUtils } from '../database.js';
import { existsSync } from 'fs';

const OLD_DB_PATH = process.env.OLD_DATABASE_PATH || './old_osint_database.sqlite';
const MIGRATION_LOG = './migration.log';

async function migrateDatabase() {
  console.log('🔄 Starting database migration...');

  try {
    // Initialize new database
    await initializeDatabase();
    const db = getDb();

    // Check if old database exists
    if (!existsSync(OLD_DB_PATH)) {
      console.log('ℹ️  No old database found, starting fresh.');
      return;
    }

    console.log(`📂 Found old database at: ${OLD_DB_PATH}`);

    // Attach old database
    db.exec(`ATTACH DATABASE '${OLD_DB_PATH}' AS old_db`);

    // Migrate data if old tables exist
    try {
      // Check if old tables exist
      const oldTables = db.prepare('SELECT name FROM old_db.sqlite_master WHERE type=\'table\'').all();
      console.log('📋 Old tables found:', oldTables.map(t => t.name));

      if (oldTables.some(t => t.name === 'research_cases' || t.name === 'ResearchCases')) {
        console.log('🔄 Migrating research cases...');

        // Migrate from old camelCase table if it exists
        const oldCasesExist = oldTables.some(t => t.name === 'ResearchCases');
        const sourceTable = oldCasesExist ? 'old_db.ResearchCases' : 'old_db.research_cases';

        const migrateStmt = db.prepare(`
          INSERT OR IGNORE INTO research_cases (case_id, topic, created_at)
          SELECT case_id, topic, created_at FROM ${sourceTable}
        `);

        const result = migrateStmt.run();
        console.log(`✅ Migrated ${result.changes} research cases`);
      }

      if (oldTables.some(t => t.name === 'agent_runs' || t.name === 'AgentRuns')) {
        console.log('🔄 Migrating agent runs...');

        const oldRunsExist = oldTables.some(t => t.name === 'AgentRuns');
        const sourceTable = oldRunsExist ? 'old_db.AgentRuns' : 'old_db.agent_runs';

        const migrateStmt = db.prepare(`
          INSERT OR IGNORE INTO agent_runs (
            run_id, case_id, model_id, temperature, status, 
            synthesized_findings, user_approved, user_feedback, created_at
          )
          SELECT 
            run_id, case_id, 
            COALESCE(model_id, 'gpt-4o-mini') as model_id,
            COALESCE(temperature, 0.3) as temperature,
            COALESCE(status, 'completed') as status,
            synthesized_findings, user_approved, user_feedback, created_at
          FROM ${sourceTable}
        `);

        const result = migrateStmt.run();
        console.log(`✅ Migrated ${result.changes} agent runs`);
      }
    } catch (error) {
      console.warn('⚠️  Error during data migration:', error.message);
    }

    // Detach old database
    db.exec('DETACH DATABASE old_db');

    // Verify migration
    const stats = dbUtils.getSystemStats();
    console.log('📊 Migration complete! Database stats:');
    console.log(`   📁 Total cases: ${stats.total_cases}`);
    console.log(`   ▶️  Total runs: ${stats.total_runs}`);
    console.log(`   ✅ Success rate: ${stats.success_rate}%`);

    // Log migration
    const logEntry = `${new Date().toISOString()} - Migration completed successfully\n`;
    require('fs').appendFileSync(MIGRATION_LOG, logEntry);
  } catch (error) {
    console.error('❌ Migration failed:', error);
    process.exit(1);
  }
}

// Run migration if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  migrateDatabase();
}

export { migrateDatabase };
