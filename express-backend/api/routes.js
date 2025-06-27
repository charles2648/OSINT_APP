// Enhanced API endpoints for the Express backend with comprehensive OSINT agent integration.

import { Router } from 'express';
import { getDb, dbUtils } from '../database.js';
import { randomUUID } from 'crypto';
import axios from 'axios';
import Joi from 'joi';

const router = Router();

// Helper function to get database instance
const getDatabase = () => getDb();
const fastapiWorkerUrl = process.env.FASTAPI_WORKER_URL || 'http://127.0.0.1:8001';

// Configuration from environment
const config = {
  approvedMemoryLimit: parseInt(process.env.APPROVED_MEMORY_LIMIT) || 10,
  searchResultsLimit: parseInt(process.env.SEARCH_RESULTS_LIMIT) || 50,
  requestTimeout: parseInt(process.env.REQUEST_TIMEOUT_MS) || 300000,
  enableAnalytics: process.env.ENABLE_ANALYTICS !== 'false',
  enableExport: process.env.ENABLE_EXPORT !== 'false',
  enableAdminEndpoints: process.env.ENABLE_ADMIN_ENDPOINTS !== 'false'
};

// Request tracking middleware for analytics
router.use((req, res, next) => {
  const requestId = randomUUID();
  req.requestId = requestId;

  const startTime = Date.now();
  req.startTime = startTime;

  // Log request start
  console.log(`📥 [${requestId}] ${req.method} ${req.path} - ${req.ip}`);

  // Track response completion
  const originalEnd = res.end;
  res.end = function(...args) {
    const duration = Date.now() - startTime;
    console.log(`📤 [${requestId}] ${res.statusCode} - ${duration}ms`);

    // Track slow requests
    if (duration > 5000) {
      console.warn(`⚠️  [${requestId}] Slow request detected: ${duration}ms for ${req.method} ${req.path}`);
    }

    originalEnd.apply(this, args);
  };

  next();
});

// Request validation schemas
const agentStartSchema = Joi.object({
  topic: Joi.string().min(5).max(500).required(),
  model_id: Joi.string().required(),
  temperature: Joi.number().min(0).max(2).required(),
  description: Joi.string().max(1000).optional(),
  priority: Joi.number().integer().min(1).max(5).default(1)
});

const feedbackSchema = Joi.object({
  case_id: Joi.string().uuid().required(),
  run_id: Joi.string().uuid().optional(),
  feedback_type: Joi.string().valid('approve', 'reject', 'modify').required(),
  rating: Joi.number().integer().min(1).max(5).optional(),
  comments: Joi.string().max(1000).optional(),
  feedback_data: Joi.object().optional(),
  trace_id: Joi.string().optional()
});

// Middleware for request validation
const validateRequest = (schema) => {
  return (req, res, next) => {
    const { error, value } = schema.validate(req.body);
    if (error) {
      return res.status(400).json({
        error: 'Validation failed',
        details: error.details.map(d => d.message),
        timestamp: new Date().toISOString()
      });
    }
    req.validatedBody = value;
    next();
  };
};

// Middleware to check AI worker availability
const checkAIWorker = async(req, res, next) => {
  try {
    const response = await axios.get(`${fastapiWorkerUrl}/models`, { timeout: 5000 });
    req.aiWorkerAvailable = true;
    req.availableModels = response.data;
    next();
  } catch (error) {
    console.error('⚠️  AI Worker unavailable:', error.message);
    req.aiWorkerAvailable = false;
    next();
  }
};

// API Documentation endpoint
router.get('/', (req, res) => {
  res.json({
    name: 'OSINT Agent API',
    version: '2.0.0',
    description: 'Enhanced backend API for OSINT investigations with comprehensive tracking and analytics',
    endpoints: {
      'GET /api/': 'API documentation (this endpoint)',
      'GET /api/models': 'Get available LLM models from AI worker',
      'POST /api/agent/start': 'Start a new OSINT investigation with streaming response',
      'POST /api/submit_feedback': 'Submit user feedback on investigation results',
      'GET /api/cases': 'List research cases with optional filtering',
      'GET /api/cases/:caseId': 'Get specific case details with run history',
      'PATCH /api/cases/:caseId': 'Update case status, description, or priority',
      'GET /api/cases/:caseId/export': 'Export case data in various formats',
      'GET /api/stats': 'Get comprehensive system statistics and analytics',
      'GET /api/search': 'Advanced search across cases, runs, and feedback',
      'POST /api/agent/review': 'Legacy endpoint for backward compatibility'
    },
    aiWorker: {
      url: fastapiWorkerUrl,
      status: 'Check /health endpoint for current status'
    },
    features: [
      'Real-time streaming investigation progress',
      'Comprehensive investigation tracking',
      'Long-term memory for approved findings',
      'Advanced search and analytics',
      'Multi-format data export',
      'Performance monitoring',
      'Security and rate limiting'
    ],
    authentication: 'None (add authentication middleware as needed)',
    rateLimit: 'Applied globally to all endpoints',
    timestamp: new Date().toISOString()
  });
});

// Get available models from AI worker
router.get('/models', checkAIWorker, async(req, res) => {
  try {
    if (!req.aiWorkerAvailable) {
      return res.status(503).json({
        success: false,
        error: 'AI Worker unavailable',
        message: 'The AI worker service is currently offline. Please try again later.',
        data: {
          fallback: {
            'gpt-4o-mini': { provider: 'OpenAI', model_name: 'GPT-4o Mini (fallback)' }
          }
        },
        timestamp: new Date().toISOString()
      });
    }

    res.json({
      success: true,
      data: req.availableModels,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('❌ Failed to get models from AI worker:', error.message);
    res.status(500).json({
      success: false,
      error: 'Failed to retrieve available models',
      details: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Start a new OSINT investigation
router.post('/agent/start', validateRequest(agentStartSchema), checkAIWorker, async(req, res) => {
  const { topic, model_id, temperature, description, priority } = req.validatedBody;

  if (!req.aiWorkerAvailable) {
    return res.status(503).json({
      success: false,
      error: 'AI Worker unavailable',
      message: 'Cannot start investigation while AI worker is offline',
      timestamp: new Date().toISOString()
    });
  }

  let caseId, runId;
  try {
    caseId = await dbUtils.createCase(topic, description, priority);
    runId = await dbUtils.createRun(caseId, model_id, temperature);
    const startTime = Date.now();

    // Get approved memory for context
    const memory = await dbUtils.getApprovedMemory(config.approvedMemoryLimit);

    console.log(`🚀 Starting investigation: ${caseId} with model: ${model_id}, temp: ${temperature}`);

    // Forward to AI worker with enhanced request
    const workerResponse = await axios.post(`${fastapiWorkerUrl}/run_agent_stream`, {
      topic,
      case_id: caseId,
      long_term_memory: memory,
      model_id,
      temperature
    }, {
      responseType: 'stream',
      timeout: config.requestTimeout
    });

    // Set up streaming response with CORS headers
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Headers', 'Cache-Control');
    res.setHeader('X-Case-ID', caseId);
    res.setHeader('X-Run-ID', runId);

    // Track the stream and update database
    let streamEnded = false;
    let lastEventData = null;

    workerResponse.data.on('data', (chunk) => {
      const data = chunk.toString();
      res.write(data);

      // Parse and track important events
      try {
        const lines = data.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const eventData = JSON.parse(line.substring(6));
            lastEventData = eventData;

            // Update run status based on events
            if (eventData.trace_id) {
              // Use non-blocking promise without await in event handler
              dbUtils.updateRun(runId, { trace_id: eventData.trace_id }).catch(error => {
                console.error('Failed to update trace_id:', error);
              });
            }
          }
        }
      } catch (parseError) {
        // Ignore parsing errors for non-JSON chunks
      }
    });

    workerResponse.data.on('end', async() => {
      if (!streamEnded) {
        streamEnded = true;
        const executionTime = Date.now() - startTime;

        try {
          // Update final run status
          await dbUtils.updateRun(runId, {
            status: 'completed',
            execution_time_ms: executionTime,
            completed_at: new Date().toISOString(),
            synthesized_findings: lastEventData?.synthesized_findings || 'No findings captured'
          });
        } catch (updateError) {
          console.error('❌ Failed to update run status:', updateError);
        }

        res.end();
      }
    });

    workerResponse.data.on('error', async(error) => {
      if (!streamEnded) {
        streamEnded = true;
        console.error('❌ Stream error:', error);

        try {
          await dbUtils.updateRun(runId, {
            status: 'failed',
            error_message: error.message,
            completed_at: new Date().toISOString()
          });
        } catch (updateError) {
          console.error('❌ Failed to update run status after error:', updateError);
        }

        res.write(`event: error\ndata: ${JSON.stringify({ error: error.message })}\n\n`);
        res.end();
      }
    });

    // Handle client disconnection
    req.on('close', async() => {
      if (!streamEnded) {
        streamEnded = true;
        console.log('⚠️  Client disconnected, cancelling agent run:', runId);

        try {
          await dbUtils.updateRun(runId, {
            status: 'cancelled',
            completed_at: new Date().toISOString()
          });
        } catch (updateError) {
          console.error('❌ Failed to update run status after cancellation:', updateError);
        }
      }
    });
  } catch (error) {
    console.error('❌ Failed to start agent:', error.message);

    // Update run status to failed if we created it
    if (runId) {
      try {
        await dbUtils.updateRun(runId, {
          status: 'failed',
          error_message: error.message,
          completed_at: new Date().toISOString()
        });
      } catch (updateError) {
        console.error('❌ Failed to update run status:', updateError);
      }
    }

    res.status(500).json({
      success: false,
      error: `Failed to communicate with AI worker: ${error.message}`,
      timestamp: new Date().toISOString(),
      data: {
        case_id: caseId || null,
        run_id: runId || null
      }
    });
  }
});

// Submit user feedback (matches frontend expectations)
router.post('/submit_feedback', validateRequest(feedbackSchema), async(req, res) => {
  const { case_id, run_id, feedback_type, rating, comments, feedback_data, trace_id } = req.validatedBody;

  try {
    // Find the run_id if not provided
    let actualRunId = run_id;
    if (!actualRunId) {
      const latestRun = await dbUtils.getLatestRunIdForCase(case_id);
      actualRunId = latestRun?.run_id;
    }

    if (!actualRunId) {
      return res.status(404).json({
        success: false,
        error: 'No agent run found for this case',
        data: { case_id },
        timestamp: new Date().toISOString()
      });
    }

    // Submit feedback using the utility function
    const feedbackId = await dbUtils.submitFeedback(
      actualRunId,
      feedback_type,
      rating,
      comments,
      feedback_data,
      trace_id
    );

    // Update agent run approval status
    const approved = feedback_type === 'approve';
    await dbUtils.updateRun(actualRunId, {
      user_approved: approved,
      user_feedback: comments
    });

    // Forward feedback to AI worker for Langfuse tracking
    try {
      await axios.post(`${fastapiWorkerUrl}/submit_feedback`, {
        case_id,
        feedback_type,
        feedback_data: feedback_data || { rating, comments },
        trace_id
      }, { timeout: 5000 });
    } catch (forwardError) {
      console.warn('⚠️  Failed to forward feedback to AI worker:', forwardError.message);
      // Don't fail the request if forwarding fails
    }

    console.log(`📝 Feedback recorded: ${feedback_type} for case ${case_id}`);

    res.json({
      success: true,
      message: 'Feedback recorded successfully',
      data: {
        feedback_id: feedbackId,
        case_id,
        run_id: actualRunId,
        approved
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('❌ Failed to save feedback:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to save feedback',
      details: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Get research cases
router.get('/cases', async(req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 20;
    const offset = parseInt(req.query.offset) || 0;
    const status = req.query.status;

    let cases;
    if (status) {
      cases = await getDatabase().all(
        'SELECT * FROM research_cases WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?',
        [status, limit, offset]
      );
    } else {
      cases = await getDatabase().all(
        'SELECT * FROM research_cases ORDER BY created_at DESC LIMIT ? OFFSET ?',
        [limit, offset]
      );
    }

    // Get run counts for each case
    for (const case_data of cases) {
      const runStats = await getDatabase().get(
        'SELECT COUNT(*) as total_runs, COUNT(CASE WHEN user_approved = 1 THEN 1 END) as approved_runs FROM agent_runs WHERE case_id = ?',
        [case_data.case_id]
      );
      case_data.run_stats = runStats;
    }

    res.json({
      success: true,
      data: {
        cases,
        pagination: {
          limit,
          offset,
          has_more: cases.length === limit
        }
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('❌ Failed to get cases:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to retrieve cases',
      details: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Get specific case details
router.get('/cases/:caseId', async(req, res) => {
  try {
    const { caseId } = req.params;
    const caseData = await dbUtils.getCase(caseId);

    if (!caseData) {
      return res.status(404).json({
        success: false,
        error: 'Case not found',
        timestamp: new Date().toISOString()
      });
    }

    // Get runs for this case
    const runs = await getDatabase().all('SELECT * FROM agent_runs WHERE case_id = ? ORDER BY created_at DESC', [caseId]);

    res.json({
      success: true,
      data: {
        ...caseData,
        runs
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('❌ Failed to get case:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to retrieve case details',
      details: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Export case data in various formats
router.get('/cases/:caseId/export', async(req, res) => {
  if (!config.enableExport) {
    return res.status(404).json({ error: 'Export functionality is disabled' });
  }

  try {
    const { caseId } = req.params;
    const { format = 'json' } = req.query;

    // Get comprehensive case data
    const caseData = dbUtils.getResearchCaseWithRuns(caseId);

    if (!caseData) {
      return res.status(404).json({ error: 'Case not found' });
    }

    // Get additional details
    const feedback = await getDatabase().all(`
      SELECT uf.* FROM user_feedback uf 
      JOIN agent_runs ar ON uf.run_id = ar.run_id 
      WHERE ar.case_id = ? 
      ORDER BY uf.created_at DESC
    `, [caseId]);

    const queries = await getDatabase().all(`
      SELECT sq.* FROM search_queries sq 
      JOIN agent_runs ar ON sq.run_id = ar.run_id 
      WHERE ar.case_id = ? 
      ORDER BY sq.created_at DESC
    `, [caseId]);

    const verifications = await getDatabase().all(`
      SELECT mv.* FROM mcp_verifications mv 
      JOIN agent_runs ar ON mv.run_id = ar.run_id 
      WHERE ar.case_id = ? 
      ORDER BY mv.created_at DESC
    `, [caseId]);

    const exportData = {
      ...caseData,
      feedback,
      search_queries: queries,
      mcp_verifications: verifications,
      export_metadata: {
        exported_at: new Date().toISOString(),
        format,
        total_runs: caseData.runs?.length || 0,
        total_feedback: feedback.length,
        total_queries: queries.length,
        total_verifications: verifications.length
      }
    };

    if (format === 'csv') {
      // Convert to CSV format for basic case data
      const csvData = [
        ['Field', 'Value'],
        ['Case ID', caseData.case_id],
        ['Topic', caseData.topic],
        ['Description', caseData.description || ''],
        ['Status', caseData.status],
        ['Priority', caseData.priority],
        ['Created At', caseData.created_at],
        ['Updated At', caseData.updated_at],
        ['Total Runs', caseData.runs?.length || 0],
        ['Approved Runs', caseData.runs?.filter(r => r.user_approved === 1).length || 0]
      ].map(row => row.join(',')).join('\n');

      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', `attachment; filename="case_${caseId}_export.csv"`);
      res.send(csvData);
    } else if (format === 'txt') {
      // Create a human-readable text report
      let textReport = `OSINT Investigation Report
Case ID: ${caseData.case_id}
Topic: ${caseData.topic}
Description: ${caseData.description || 'N/A'}
Status: ${caseData.status}
Priority: ${caseData.priority}
Created: ${caseData.created_at}
Updated: ${caseData.updated_at}

=== INVESTIGATION RUNS ===
`;

      for (const run of caseData.runs || []) {
        textReport += `
Run ID: ${run.run_id}
Model: ${run.model_id}
Status: ${run.status}
Execution Time: ${run.execution_time_ms ? `${run.execution_time_ms}ms` : 'N/A'}
User Approved: ${run.user_approved === 1 ? 'Yes' : run.user_approved === 0 ? 'No' : 'Pending'}
Created: ${run.created_at}

Findings:
${run.synthesized_findings || 'No findings available'}

${run.user_feedback ? `User Feedback: ${run.user_feedback}` : ''}
---
`;
      }

      res.setHeader('Content-Type', 'text/plain');
      res.setHeader('Content-Disposition', `attachment; filename="case_${caseId}_report.txt"`);
      res.send(textReport);
    } else {
      // Default JSON format
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Content-Disposition', `attachment; filename="case_${caseId}_export.json"`);
      res.json(exportData);
    }

    console.log(`📊 Exported case ${caseId} in ${format} format`);
  } catch (error) {
    console.error('❌ Failed to export case:', error);
    res.status(500).json({
      error: 'Failed to export case data',
      details: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Get system statistics
router.get('/stats', async(req, res) => {
  try {
    const stats = await dbUtils.getSystemStats();

    // Get recent activity
    const recentRuns = await getDatabase().all(
      'SELECT COUNT(*) as count, DATE(created_at) as date FROM agent_runs WHERE created_at >= datetime("now", "-7 days") GROUP BY DATE(created_at) ORDER BY date DESC'
    );

    // Get model usage statistics
    const modelStats = await getDatabase().all(
      'SELECT model_id, COUNT(*) as usage_count, AVG(execution_time_ms) as avg_execution_time FROM agent_runs WHERE status = "completed" GROUP BY model_id ORDER BY usage_count DESC'
    );

    // Get error statistics
    const errorStats = await getDatabase().all(
      'SELECT error_message, COUNT(*) as count FROM agent_runs WHERE status = "failed" AND error_message IS NOT NULL GROUP BY error_message ORDER BY count DESC LIMIT 10'
    );

    // Get performance metrics
    const performanceStats = await getDatabase().get(`
      SELECT 
        AVG(execution_time_ms) as avg_execution_time,
        MIN(execution_time_ms) as min_execution_time,
        MAX(execution_time_ms) as max_execution_time,
        COUNT(CASE WHEN execution_time_ms > 120000 THEN 1 END) as long_running_count
      FROM agent_runs 
      WHERE status = 'completed' AND execution_time_ms IS NOT NULL
    `);

    res.json({
      success: true,
      data: {
        ...stats,
        recent_activity: recentRuns,
        model_usage: modelStats,
        error_analysis: errorStats,
        performance: performanceStats
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('❌ Failed to get stats:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to retrieve system statistics',
      details: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Admin endpoint to manage cases
router.patch('/cases/:caseId', async(req, res) => {
  if (!config.enableAdminEndpoints) {
    return res.status(404).json({
      success: false,
      error: 'Admin endpoints are disabled',
      timestamp: new Date().toISOString()
    });
  }

  try {
    const { caseId } = req.params;
    const { status, description, priority } = req.body;

    // Validate status if provided
    if (status && !['active', 'completed', 'archived'].includes(status)) {
      return res.status(400).json({
        success: false,
        error: 'Invalid status. Must be active, completed, or archived.',
        timestamp: new Date().toISOString()
      });
    }

    // Build update object
    const updates = {};
    if (status) updates.status = status;
    if (description !== undefined) updates.description = description;
    if (priority !== undefined) updates.priority = priority;

    if (Object.keys(updates).length === 0) {
      return res.status(400).json({
        success: false,
        error: 'No valid fields to update',
        timestamp: new Date().toISOString()
      });
    }

    await dbUtils.updateCase(caseId, updates);

    // Get updated case
    const updatedCase = await dbUtils.getCase(caseId);

    if (!updatedCase) {
      return res.status(404).json({
        success: false,
        error: 'Case not found',
        timestamp: new Date().toISOString()
      });
    }

    res.json({
      success: true,
      message: 'Case updated successfully',
      data: updatedCase,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('❌ Failed to update case:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to update case',
      details: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Advanced search endpoint
router.get('/search', async(req, res) => {
  try {
    const { query, type = 'all', limit = config.searchResultsLimit, offset = 0 } = req.query;

    if (!query || query.trim().length < 2) {
      return res.status(400).json({
        success: false,
        error: 'Search query must be at least 2 characters long',
        timestamp: new Date().toISOString()
      });
    }

    const searchTerm = `%${query.trim()}%`;
    const results = {};

    if (type === 'all' || type === 'cases') {
      results.cases = await getDatabase().all(
        'SELECT * FROM research_cases WHERE topic LIKE ? OR description LIKE ? ORDER BY created_at DESC LIMIT ? OFFSET ?',
        [searchTerm, searchTerm, limit, offset]
      );
    }

    if (type === 'all' || type === 'runs') {
      results.runs = await getDatabase().all(
        'SELECT ar.*, rc.topic FROM agent_runs ar JOIN research_cases rc ON ar.case_id = rc.case_id WHERE ar.synthesized_findings LIKE ? OR ar.user_feedback LIKE ? ORDER BY ar.created_at DESC LIMIT ? OFFSET ?',
        [searchTerm, searchTerm, limit, offset]
      );
    }

    if (type === 'all' || type === 'feedback') {
      results.feedback = await getDatabase().all(
        'SELECT uf.*, rc.topic FROM user_feedback uf JOIN agent_runs ar ON uf.run_id = ar.run_id JOIN research_cases rc ON ar.case_id = rc.case_id WHERE uf.comments LIKE ? ORDER BY uf.created_at DESC LIMIT ? OFFSET ?',
        [searchTerm, limit, offset]
      );
    }

    res.json({
      success: true,
      data: {
        query,
        type,
        results,
        pagination: { limit, offset }
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('❌ Search failed:', error);
    res.status(500).json({
      success: false,
      error: 'Search failed',
      details: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Legacy routes for backward compatibility
router.post('/agent/review', async(req, res) => {
  console.warn('⚠️  Using deprecated /agent/review endpoint. Use /submit_feedback instead.');

  const { caseId, findings, approved, feedback } = req.body;

  try {
    // Find the run_id
    const latestRun = await dbUtils.getLatestRunIdForCase(caseId);

    if (!latestRun) {
      return res.status(404).json({
        success: false,
        error: 'No agent run found for this case',
        data: { case_id: caseId },
        timestamp: new Date().toISOString()
      });
    }

    // Submit feedback using the new utility function
    const feedbackId = await dbUtils.submitFeedback(
      latestRun.run_id,
      approved ? 'approve' : 'reject',
      null,
      feedback,
      { findings },
      null
    );

    // Update agent run approval status
    await dbUtils.updateRun(latestRun.run_id, {
      user_approved: approved,
      user_feedback: feedback
    });

    res.json({
      success: true,
      status: approved ? 'approved' : 'rejected',
      message: 'Review has been recorded successfully.',
      data: {
        feedback_id: feedbackId,
        case_id: caseId,
        run_id: latestRun.run_id,
        report_path: approved ? `/reports/report_${latestRun.run_id}.txt` : null
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('❌ Failed to save review:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to save review.',
      details: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

export default router;
