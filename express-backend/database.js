// database.js - Enhanced database utilities for OSINT app with performance optimizations
import sqlite3 from 'sqlite3';
import { open } from 'sqlite';
import { randomUUID } from 'crypto';

// Singleton database instance
let db = null;

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
    console.log(`🗄️  Initializing database at: ${dbPath}`);

    db = await open({
      filename: dbPath,
      driver: sqlite3.Database
    });

    // Enable optimizations for better performance
    await db.exec(`
      PRAGMA foreign_keys = ON;
      PRAGMA journal_mode = WAL;
      PRAGMA synchronous = NORMAL;
      PRAGMA cache_size = 1000;
      PRAGMA temp_store = MEMORY;
    `);

    console.log('📋 Creating enhanced database schema...');

    // Use transactions for schema creation
    await db.exec('BEGIN TRANSACTION');

    try {
      // Research Cases - Main investigation topics
      await db.exec(`
        CREATE TABLE IF NOT EXISTS research_cases (
          case_id TEXT PRIMARY KEY,
          topic TEXT NOT NULL,
          description TEXT,
          priority INTEGER DEFAULT 1,
          status TEXT DEFAULT 'active' CHECK(status IN ('active', 'completed', 'archived')),
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
      `);

      // Agent Runs - Individual agent executions
      await db.exec(`
        CREATE TABLE IF NOT EXISTS agent_runs (
          run_id TEXT PRIMARY KEY,
          case_id TEXT NOT NULL,
          model_id TEXT NOT NULL,
          temperature REAL NOT NULL,
          trace_id TEXT,
          status TEXT DEFAULT 'running' CHECK(status IN ('running', 'completed', 'failed', 'cancelled')),
          execution_time_ms INTEGER,
          total_steps INTEGER DEFAULT 0,
          search_queries_count INTEGER DEFAULT 0,
          mcp_verifications_count INTEGER DEFAULT 0,
          synthesized_findings TEXT,
          final_confidence_score REAL,
          user_approved BOOLEAN DEFAULT NULL,
          user_feedback TEXT,
          error_message TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          completed_at DATETIME,
          FOREIGN KEY(case_id) REFERENCES research_cases(case_id) ON DELETE CASCADE
        );
      `);

      // Search Queries - Track all search operations
      await db.exec(`
        CREATE TABLE IF NOT EXISTS search_queries (
          query_id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL,
          query_text TEXT NOT NULL,
          results_count INTEGER DEFAULT 0,
          execution_time_ms INTEGER,
          success BOOLEAN DEFAULT TRUE,
          error_message TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
        );
      `);

      // MCP Verifications - Track verification tool usage
      await db.exec(`
        CREATE TABLE IF NOT EXISTS mcp_verifications (
          verification_id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL,
          mcp_name TEXT NOT NULL,
          input_data TEXT NOT NULL,
          result_data TEXT,
          success BOOLEAN DEFAULT TRUE,
          execution_time_ms INTEGER,
          confidence_score REAL,
          error_message TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
        );
      `);

      // User Feedback - Detailed feedback tracking
      await db.exec(`
        CREATE TABLE IF NOT EXISTS user_feedback (
          feedback_id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL,
          feedback_type TEXT NOT NULL CHECK(feedback_type IN ('approve', 'reject', 'modify')),
          rating INTEGER CHECK(rating BETWEEN 1 AND 5),
          comments TEXT,
          specific_feedback TEXT, -- JSON string for detailed feedback
          trace_id TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
        );
      `);

      // System Health - Track system performance
      await db.exec(`
        CREATE TABLE IF NOT EXISTS system_health (
          health_id TEXT PRIMARY KEY,
          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
          ai_worker_status TEXT CHECK(ai_worker_status IN ('online', 'offline', 'error')),
          ai_worker_response_time_ms INTEGER,
          database_status TEXT CHECK(database_status IN ('healthy', 'slow', 'error')),
          total_active_cases INTEGER DEFAULT 0,
          total_pending_runs INTEGER DEFAULT 0,
          last_successful_run DATETIME
        );
      `);

      // Create useful indexes for performance
      await db.exec(`
        CREATE INDEX IF NOT EXISTS idx_agent_runs_case_id ON agent_runs(case_id);
        CREATE INDEX IF NOT EXISTS idx_agent_runs_status ON agent_runs(status);
        CREATE INDEX IF NOT EXISTS idx_agent_runs_created_at ON agent_runs(created_at);
        CREATE INDEX IF NOT EXISTS idx_agent_runs_user_approved ON agent_runs(user_approved);
        CREATE INDEX IF NOT EXISTS idx_search_queries_run_id ON search_queries(run_id);
        CREATE INDEX IF NOT EXISTS idx_mcp_verifications_run_id ON mcp_verifications(run_id);
        CREATE INDEX IF NOT EXISTS idx_user_feedback_run_id ON user_feedback(run_id);
        CREATE INDEX IF NOT EXISTS idx_research_cases_status ON research_cases(status);
        CREATE INDEX IF NOT EXISTS idx_research_cases_created_at ON research_cases(created_at);
      `);

      // Create triggers for updating timestamps
      await db.exec(`
        CREATE TRIGGER IF NOT EXISTS update_research_cases_timestamp 
        AFTER UPDATE ON research_cases
        BEGIN
          UPDATE research_cases SET updated_at = CURRENT_TIMESTAMP WHERE case_id = NEW.case_id;
        END;
      `);

      await db.exec('COMMIT');
      console.log('✅ Enhanced database schema created successfully');
    } catch (error) {
      await db.exec('ROLLBACK');
      throw error;
    }

    console.log('✅ Database initialized successfully');
    return db;
  } catch (error) {
    console.error('❌ Database initialization failed:', error);
    throw new Error(`Database initialization failed: ${error.message}`);
  }
}

/**
 * @returns {import('sqlite').Database} The database instance.
 */
export function getDb() {
  if (!db) {
    throw new Error('Database not initialized. Call initializeDatabase() first and await its completion.');
  }
  return db;
}

/**
 * Database utility functions for common operations with performance optimizations
 */
export const dbUtils = {
  // Create a new research case
  async createCase(topic, description = '', priority = 1) {
    const caseId = randomUUID();
    await db.run(
      'INSERT INTO research_cases (case_id, topic, description, priority) VALUES (?, ?, ?, ?)',
      [caseId, topic, description, priority]
    );
    return caseId;
  },

  // Get case by ID with detailed information
  async getCase(caseId) {
    return await db.get(
      `SELECT c.*, 
              COUNT(DISTINCT ar.run_id) as total_runs,
              COUNT(DISTINCT CASE WHEN ar.status = 'completed' THEN ar.run_id END) as completed_runs,
              COUNT(DISTINCT CASE WHEN ar.user_approved = 1 THEN ar.run_id END) as approved_runs,
              AVG(ar.execution_time_ms) as avg_execution_time,
              MAX(ar.created_at) as last_run_at
       FROM research_cases c
       LEFT JOIN agent_runs ar ON c.case_id = ar.case_id
       WHERE c.case_id = ?
       GROUP BY c.case_id`,
      [caseId]
    );
  },

  // List all cases with summary statistics
  async getAllCases() {
    return await db.all(`
      SELECT c.*, 
             COUNT(DISTINCT ar.run_id) as total_runs,
             COUNT(DISTINCT CASE WHEN ar.status = 'completed' THEN ar.run_id END) as completed_runs,
             COUNT(DISTINCT CASE WHEN ar.user_approved = 1 THEN ar.run_id END) as approved_runs,
             MAX(ar.created_at) as last_run_at
      FROM research_cases c
      LEFT JOIN agent_runs ar ON c.case_id = ar.case_id
      GROUP BY c.case_id
      ORDER BY c.updated_at DESC
    `);
  },

  // Update case details
  async updateCase(caseId, updates) {
    const allowedFields = ['topic', 'description', 'priority', 'status'];
    const setClause = [];
    const values = [];

    for (const [key, value] of Object.entries(updates)) {
      if (allowedFields.includes(key)) {
        setClause.push(`${key} = ?`);
        values.push(value);
      }
    }

    if (setClause.length === 0) {
      throw new Error('No valid fields to update');
    }

    values.push(caseId);
    await db.run(
      `UPDATE research_cases SET ${setClause.join(', ')}, updated_at = CURRENT_TIMESTAMP WHERE case_id = ?`,
      values
    );
  },

  // Create a new agent run
  async createRun(caseId, modelId, temperature, traceId = null) {
    const runId = randomUUID();
    await db.run(
      'INSERT INTO agent_runs (run_id, case_id, model_id, temperature, trace_id) VALUES (?, ?, ?, ?, ?)',
      [runId, caseId, modelId, temperature, traceId]
    );
    return runId;
  },

  // Update agent run status and results
  async updateRun(runId, updates) {
    const allowedFields = [
      'status', 'execution_time_ms', 'total_steps', 'search_queries_count',
      'mcp_verifications_count', 'synthesized_findings', 'final_confidence_score',
      'user_approved', 'user_feedback', 'error_message', 'completed_at'
    ];

    const setClause = [];
    const values = [];

    for (const [key, value] of Object.entries(updates)) {
      if (allowedFields.includes(key)) {
        setClause.push(`${key} = ?`);
        values.push(value);
      }
    }

    if (setClause.length === 0) {
      throw new Error('No valid fields to update');
    }

    values.push(runId);
    await db.run(
      `UPDATE agent_runs SET ${setClause.join(', ')} WHERE run_id = ?`,
      values
    );
  },

  // Get run details with related data
  async getRun(runId) {
    const run = await db.get('SELECT * FROM agent_runs WHERE run_id = ?', [runId]);
    if (!run) return null;

    // Get related queries and verifications
    const queries = await db.all('SELECT * FROM search_queries WHERE run_id = ? ORDER BY created_at', [runId]);
    const verifications = await db.all('SELECT * FROM mcp_verifications WHERE run_id = ? ORDER BY created_at', [runId]);
    const feedback = await db.all('SELECT * FROM user_feedback WHERE run_id = ? ORDER BY created_at', [runId]);

    return { ...run, queries, verifications, feedback };
  },

  // Record search query
  async recordQuery(runId, queryText, resultsCount = 0, executionTimeMs = null, success = true, errorMessage = null) {
    const queryId = randomUUID();
    await db.run(
      'INSERT INTO search_queries (query_id, run_id, query_text, results_count, execution_time_ms, success, error_message) VALUES (?, ?, ?, ?, ?, ?, ?)',
      [queryId, runId, queryText, resultsCount, executionTimeMs, success, errorMessage]
    );
    return queryId;
  },

  // Record MCP verification
  async recordVerification(runId, mcpName, inputData, resultData = null, success = true, executionTimeMs = null, confidenceScore = null, errorMessage = null) {
    const verificationId = randomUUID();
    await db.run(
      'INSERT INTO mcp_verifications (verification_id, run_id, mcp_name, input_data, result_data, success, execution_time_ms, confidence_score, error_message) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
      [verificationId, runId, mcpName, JSON.stringify(inputData), resultData ? JSON.stringify(resultData) : null, success, executionTimeMs, confidenceScore, errorMessage]
    );
    return verificationId;
  },

  // Submit user feedback
  async submitFeedback(runId, feedbackType, rating = null, comments = '', specificFeedback = null, traceId = null) {
    const feedbackId = randomUUID();
    await db.run(
      'INSERT INTO user_feedback (feedback_id, run_id, feedback_type, rating, comments, specific_feedback, trace_id) VALUES (?, ?, ?, ?, ?, ?, ?)',
      [feedbackId, runId, feedbackType, rating, comments, specificFeedback ? JSON.stringify(specificFeedback) : null, traceId]
    );
    return feedbackId;
  },

  // Record system health metrics
  async recordHealthMetrics(metrics) {
    const healthId = randomUUID();
    const {
      aiWorkerStatus = 'online',
      aiWorkerResponseTimeMs = null,
      databaseStatus = 'healthy',
      totalActiveCases = 0,
      totalPendingRuns = 0,
      lastSuccessfulRun = null
    } = metrics;

    await db.run(
      'INSERT INTO system_health (health_id, ai_worker_status, ai_worker_response_time_ms, database_status, total_active_cases, total_pending_runs, last_successful_run) VALUES (?, ?, ?, ?, ?, ?, ?)',
      [healthId, aiWorkerStatus, aiWorkerResponseTimeMs, databaseStatus, totalActiveCases, totalPendingRuns, lastSuccessfulRun]
    );
    return healthId;
  },

  // Get system statistics
  async getSystemStats() {
    const stats = await db.get(`
      SELECT 
        COUNT(DISTINCT rc.case_id) as total_cases,
        COUNT(DISTINCT CASE WHEN rc.status = 'active' THEN rc.case_id END) as active_cases,
        COUNT(DISTINCT ar.run_id) as total_runs,
        COUNT(DISTINCT CASE WHEN ar.status = 'running' THEN ar.run_id END) as running_runs,
        COUNT(DISTINCT CASE WHEN ar.status = 'completed' THEN ar.run_id END) as completed_runs,
        COUNT(DISTINCT CASE WHEN ar.user_approved = 1 THEN ar.run_id END) as approved_runs,
        AVG(ar.execution_time_ms) as avg_execution_time,
        COUNT(DISTINCT sq.query_id) as total_queries,
        COUNT(DISTINCT mv.verification_id) as total_verifications,
        COUNT(DISTINCT uf.feedback_id) as total_feedback
      FROM research_cases rc
      LEFT JOIN agent_runs ar ON rc.case_id = ar.case_id
      LEFT JOIN search_queries sq ON ar.run_id = sq.run_id
      LEFT JOIN mcp_verifications mv ON ar.run_id = mv.run_id
      LEFT JOIN user_feedback uf ON ar.run_id = uf.run_id
    `);

    // Get recent health metrics
    const recentHealth = await db.get(
      'SELECT * FROM system_health ORDER BY timestamp DESC LIMIT 1'
    );

    return { ...stats, recentHealth };
  },

  // Advanced search across cases and runs
  async searchCasesAndRuns(searchTerm, limit = 50) {
    const pattern = `%${searchTerm}%`;
    return await db.all(`
      SELECT 
        'case' as type,
        rc.case_id as id,
        rc.topic as title,
        rc.description as content,
        rc.created_at,
        rc.status
      FROM research_cases rc
      WHERE rc.topic LIKE ? OR rc.description LIKE ?
      UNION ALL
      SELECT 
        'run' as type,
        ar.run_id as id,
        'Run: ' || ar.model_id as title,
        ar.synthesized_findings as content,
        ar.created_at,
        ar.status
      FROM agent_runs ar
      WHERE ar.synthesized_findings LIKE ? OR ar.user_feedback LIKE ?
      ORDER BY created_at DESC
      LIMIT ?
    `, [pattern, pattern, pattern, pattern, limit]);
  },

  // Batch insert operations for performance
  async batchInsert(tableName, records, columns) {
    if (!records.length) return { inserted: 0 };

    const placeholders = columns.map(() => '?').join(', ');
    const sql = `INSERT INTO ${tableName} (${columns.join(', ')}) VALUES (${placeholders})`;

    await db.exec('BEGIN TRANSACTION');

    try {
      for (const record of records) {
        await db.run(sql, columns.map(col => record[col]));
      }
      await db.exec('COMMIT');
      return { inserted: records.length };
    } catch (error) {
      await db.exec('ROLLBACK');
      throw error;
    }
  },

  // Database maintenance operations
  async vacuum() {
    console.log('🧹 Running database vacuum...');
    await db.exec('VACUUM');
    console.log('✅ Database vacuum completed');
  },

  async analyze() {
    console.log('📊 Running database analyze...');
    await db.exec('ANALYZE');
    console.log('✅ Database analyze completed');
  },

  // Get database info and size
  async getDatabaseInfo() {
    const pragma = await db.all('PRAGMA database_list');
    const tables = await db.all('SELECT name FROM sqlite_master WHERE type=\'table\'');

    const tableStats = {};
    for (const table of tables) {
      const count = await db.get(`SELECT COUNT(*) as count FROM ${table.name}`);
      tableStats[table.name] = count.count;
    }

    return {
      pragmas: pragma,
      tables: tableStats,
      totalTables: tables.length
    };
  },

  // Additional utility functions for API compatibility
  async getLatestRunIdForCase(caseId) {
    return await db.get(
      'SELECT run_id FROM agent_runs WHERE case_id = ? ORDER BY created_at DESC LIMIT 1',
      [caseId]
    );
  },

  async getApprovedMemory(limit = 10) {
    return await db.all(`
      SELECT ar.synthesized_findings, ar.case_id, rc.topic, ar.created_at
      FROM agent_runs ar
      JOIN research_cases rc ON ar.case_id = rc.case_id
      WHERE ar.user_approved = 1 AND ar.synthesized_findings IS NOT NULL
      ORDER BY ar.created_at DESC
      LIMIT ?
    `, [limit]);
  },

  // Legacy compatibility functions (deprecated)
  async createResearchCase(caseId, topic, description = '', priority = 1) {
    console.warn('⚠️  createResearchCase is deprecated. Use createCase instead.');
    await db.run(
      'INSERT INTO research_cases (case_id, topic, description, priority) VALUES (?, ?, ?, ?)',
      [caseId, topic, description, priority]
    );
    return { lastInsertRowid: 1 }; // Legacy return format
  },

  async createAgentRun(runId, caseId, modelId, temperature, traceId = null) {
    console.warn('⚠️  createAgentRun is deprecated. Use createRun instead.');
    await db.run(
      'INSERT INTO agent_runs (run_id, case_id, model_id, temperature, trace_id) VALUES (?, ?, ?, ?, ?)',
      [runId, caseId, modelId, temperature, traceId]
    );
    return { lastInsertRowid: 1 }; // Legacy return format
  },

  async getResearchCaseWithRuns(caseId) {
    console.warn('⚠️  getResearchCaseWithRuns is deprecated. Use getCase instead.');
    const caseData = await this.getCase(caseId);
    if (!caseData) return null;

    const runs = await db.all('SELECT * FROM agent_runs WHERE case_id = ? ORDER BY created_at DESC', [caseId]);
    return { ...caseData, runs };
  },

  async updateAgentRun(runId, updates) {
    console.warn('⚠️  updateAgentRun is deprecated. Use updateRun instead.');
    return await this.updateRun(runId, updates);
  },

  async recordUserFeedback(feedbackId, runId, feedbackType, rating, comments, feedbackData, traceId) {
    console.warn('⚠️  recordUserFeedback is deprecated. Use submitFeedback instead.');
    await db.run(
      'INSERT INTO user_feedback (feedback_id, run_id, feedback_type, rating, comments, specific_feedback, trace_id) VALUES (?, ?, ?, ?, ?, ?, ?)',
      [feedbackId, runId, feedbackType, rating, comments, feedbackData ? JSON.stringify(feedbackData) : null, traceId]
    );
    return feedbackId;
  }
};
