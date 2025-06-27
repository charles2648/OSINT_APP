// Integration tests for API endpoints

import { strict as assert } from 'assert';
import { test, describe, before, after } from 'node:test';
import { initializeDatabase, dbUtils } from '../../database.js';
import request from 'supertest';
import express from 'express';
import routes from '../../api/routes.js';
import fs from 'fs';

const TEST_DB_PATH = './test_api_database.sqlite';
const app = express();

describe('API Integration Tests', () => {
  before(async() => {
    // Set test database path and initialize
    process.env.DATABASE_PATH = TEST_DB_PATH;
    await initializeDatabase();

    // Setup express app
    app.use(express.json());
    app.use('/api', routes);
  });

  after(() => {
    // Clean up test database
    if (fs.existsSync(TEST_DB_PATH)) {
      fs.unlinkSync(TEST_DB_PATH);
    }
  });

  test('GET /api/ should return API documentation', async() => {
    const response = await request(app)
      .get('/api/')
      .expect(200);

    assert.strictEqual(response.body.name, 'OSINT Agent API');
    assert.strictEqual(response.body.version, '2.0.0');
    assert.ok(response.body.endpoints);
  });

  test('GET /api/models should return models with consistent response format', async() => {
    const response = await request(app)
      .get('/api/models')
      .expect(503); // AI worker unavailable in test

    assert.strictEqual(response.body.success, false);
    assert.ok(response.body.error);
    assert.ok(response.body.data.fallback);
    assert.ok(response.body.timestamp);
  });

  test('GET /api/cases should return cases with consistent response format', async() => {
    // Create a test case first
    const caseId = await dbUtils.createCase('Test API Case', 'Test description');

    const response = await request(app)
      .get('/api/cases')
      .expect(200);

    assert.strictEqual(response.body.success, true);
    assert.ok(Array.isArray(response.body.data.cases));
    assert.ok(response.body.data.pagination);
    assert.ok(response.body.timestamp);

    // Check that our test case is in the results
    const testCase = response.body.data.cases.find(c => c.case_id === caseId);
    assert.ok(testCase);
    assert.strictEqual(testCase.topic, 'Test API Case');
  });

  test('GET /api/cases/:caseId should return case details', async() => {
    // Create a test case first
    const caseId = await dbUtils.createCase('Detailed Test Case', 'Detailed description');

    const response = await request(app)
      .get(`/api/cases/${caseId}`)
      .expect(200);

    assert.strictEqual(response.body.success, true);
    assert.strictEqual(response.body.data.case_id, caseId);
    assert.strictEqual(response.body.data.topic, 'Detailed Test Case');
    assert.ok(Array.isArray(response.body.data.runs));
    assert.ok(response.body.timestamp);
  });

  test('POST /api/submit_feedback should handle feedback submission', async() => {
    // Create a test case and run first
    const caseId = await dbUtils.createCase('Feedback Test Case');
    const runId = await dbUtils.createRun(caseId, 'gpt-4o-mini', 0.7);

    const feedbackData = {
      case_id: caseId,
      run_id: runId,
      feedback_type: 'approve',
      rating: 5,
      comments: 'Excellent work!'
    };

    const response = await request(app)
      .post('/api/submit_feedback')
      .send(feedbackData)
      .expect(200);

    assert.strictEqual(response.body.success, true);
    assert.strictEqual(response.body.message, 'Feedback recorded successfully');
    assert.strictEqual(response.body.data.case_id, caseId);
    assert.strictEqual(response.body.data.run_id, runId);
    assert.strictEqual(response.body.data.approved, true);
    assert.ok(response.body.data.feedback_id);
    assert.ok(response.body.timestamp);
  });

  test('GET /api/stats should return system statistics', async() => {
    const response = await request(app)
      .get('/api/stats')
      .expect(200);

    assert.strictEqual(response.body.success, true);
    assert.ok(typeof response.body.data.total_cases === 'number');
    assert.ok(typeof response.body.data.active_cases === 'number');
    assert.ok(typeof response.body.data.total_runs === 'number');
    assert.ok(Array.isArray(response.body.data.recent_activity));
    assert.ok(Array.isArray(response.body.data.model_usage));
    assert.ok(response.body.timestamp);
  });

  test('GET /api/search should perform search with consistent response format', async() => {
    // Create test data
    await dbUtils.createCase('Search Test Case', 'This is a searchable description');

    const response = await request(app)
      .get('/api/search?query=searchable&type=cases')
      .expect(200);

    assert.strictEqual(response.body.success, true);
    assert.strictEqual(response.body.data.query, 'searchable');
    assert.strictEqual(response.body.data.type, 'cases');
    assert.ok(response.body.data.results);
    assert.ok(response.body.data.pagination);
    assert.ok(response.body.timestamp);
  });

  test('GET /api/search should handle validation errors', async() => {
    const response = await request(app)
      .get('/api/search?query=a') // Too short
      .expect(400);

    assert.strictEqual(response.body.success, false);
    assert.ok(response.body.error.includes('at least 2 characters'));
    assert.ok(response.body.timestamp);
  });

  test('POST /api/submit_feedback should handle validation errors', async() => {
    const invalidFeedbackData = {
      case_id: 'invalid-uuid',
      feedback_type: 'invalid-type'
    };

    const response = await request(app)
      .post('/api/submit_feedback')
      .send(invalidFeedbackData)
      .expect(400);

    assert.strictEqual(response.body.error, 'Validation failed');
    assert.ok(Array.isArray(response.body.details));
    assert.ok(response.body.timestamp);
  });

  test('Legacy POST /api/agent/review should work with consistent response format', async() => {
    // Create a test case and run first
    const caseId = await dbUtils.createCase('Legacy Test Case');
    const _runId = await dbUtils.createRun(caseId, 'gpt-4o-mini', 0.5);

    const reviewData = {
      caseId,
      findings: 'Test findings',
      approved: true,
      feedback: 'Great investigation!'
    };

    const response = await request(app)
      .post('/api/agent/review')
      .send(reviewData)
      .expect(200);

    assert.strictEqual(response.body.success, true);
    assert.strictEqual(response.body.status, 'approved');
    assert.strictEqual(response.body.message, 'Review has been recorded successfully.');
    assert.strictEqual(response.body.data.case_id, caseId);
    assert.ok(response.body.data.feedback_id);
    assert.ok(response.body.timestamp);
  });
});
