// Unit tests for database utility functions

import { strict as assert } from 'assert';
import { test, describe, before, after } from 'node:test';
import { initializeDatabase, dbUtils } from '../../database.js';
import fs from 'fs';

const TEST_DB_PATH = './test_database.sqlite';

describe('Database Utils', () => {
  before(async() => {
    // Set test database path
    process.env.DATABASE_PATH = TEST_DB_PATH;
    await initializeDatabase();
  });

  after(() => {
    // Clean up test database
    if (fs.existsSync(TEST_DB_PATH)) {
      fs.unlinkSync(TEST_DB_PATH);
    }
  });

  test('should create research case', async() => {
    const topic = 'Test investigation';

    const caseId = await dbUtils.createCase(topic);
    assert.ok(caseId);

    const caseData = await dbUtils.getCase(caseId);
    assert.strictEqual(caseData.case_id, caseId);
    assert.strictEqual(caseData.topic, topic);
  });

  test('should create agent run', async() => {
    const topic = 'Test investigation for run';

    // Create case first
    const caseId = await dbUtils.createCase(topic);

    // Create run
    const runId = await dbUtils.createRun(caseId, 'gpt-4o-mini', 0.3);
    assert.ok(runId);

    const runData = await dbUtils.getRun(runId);
    assert.strictEqual(runData.run_id, runId);
    assert.strictEqual(runData.case_id, caseId);
    assert.strictEqual(runData.model_id, 'gpt-4o-mini');
    assert.strictEqual(runData.temperature, 0.3);
  });

  test('should get system stats', async() => {
    const stats = await dbUtils.getSystemStats();
    assert.ok(typeof stats.total_cases === 'number');
    assert.ok(typeof stats.active_cases === 'number');
    assert.ok(typeof stats.total_runs === 'number');
    assert.ok(typeof stats.completed_runs === 'number');
    assert.ok(typeof stats.approved_runs === 'number');
  });

  test('should record feedback', async() => {
    // Create a case and run first
    const caseId = await dbUtils.createCase('Test feedback case');
    const runId = await dbUtils.createRun(caseId, 'gpt-4o-mini', 0.5);

    // Submit feedback
    const feedbackId = await dbUtils.submitFeedback(runId, 'approve', 4, 'Great work!');
    assert.ok(feedbackId);

    // Verify feedback was recorded
    const runData = await dbUtils.getRun(runId);
    assert.ok(runData.feedback);
    assert.strictEqual(runData.feedback.length, 1);
    assert.strictEqual(runData.feedback[0].feedback_type, 'approve');
  });
});
