// Defines the API endpoints for the Express backend.

import { Router } from 'express';
import { getDb } from '../database.js';
import { randomUUID } from 'crypto';
import axios from 'axios';

const router = Router();
const db = getDb();
const fastapiWorkerUrl = process.env.FASTAPI_WORKER_URL || 'http://127.0.0.1:8001';

router.get('/models', async (req, res) => {
  try {
    const response = await axios.get(`${fastapiWorkerUrl}/models`);
    res.json(response.data);
  } catch (error) {
    console.error('Failed to get models from AI worker:', error.message);
    res.status(500).json({ error: 'Failed to retrieve available models.' });
  }
});

router.post('/agent/start', async (req, res) => {
  const { topic, model_id, temperature } = req.body;
  if (!topic || !model_id || temperature === undefined) {
    return res.status(400).json({ error: 'Topic, model_id, and temperature are required.' });
  }

  try {
    const caseId = randomUUID();
    await db.run('INSERT INTO ResearchCases (case_id, topic) VALUES (?, ?)', [caseId, topic]);

    const memory = await db.all(`
        SELECT rc.topic, ar.synthesized_findings as findings
        FROM AgentRuns ar
        JOIN ResearchCases rc ON ar.case_id = rc.case_id
        WHERE ar.user_approved = 1
        ORDER BY ar.created_at DESC
        LIMIT 5
    `);

    console.log(`Forwarding request to AI worker with model: ${model_id}, temp: ${temperature}`);
    
    const workerResponse = await axios.post(`${fastapiWorkerUrl}/run_agent_stream`, {
      topic,
      case_id: caseId,
      long_term_memory: memory,
      model_id: model_id,
      temperature: temperature,
    }, {
      responseType: 'stream',
    });

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    workerResponse.data.pipe(res);

  } catch (error) {
    console.error('Failed to start agent:', error.message);
    res.status(500).json({ error: `Failed to communicate with AI worker: ${error.message}` });
  }
});

router.post('/agent/review', async (req, res) => {
    const { caseId, findings, approved, feedback } = req.body;
    if (!caseId || findings === undefined || approved === undefined) {
        return res.status(400).json({ error: 'Missing required review data.' });
    }

    try {
        const runId = randomUUID();
        await db.run(
            `INSERT INTO AgentRuns (run_id, case_id, synthesized_findings, user_approved, user_feedback)
             VALUES (?, ?, ?, ?, ?)`,
            [runId, caseId, findings, approved, feedback || null]
        );
        
        const reportPath = `/reports/report_${runId}.txt`; // Dummy path
        if (approved) {
            console.log(`Simulating report generation for run ${runId}`);
        }

        res.json({
            status: approved ? 'approved' : 'rejected',
            message: 'Review has been recorded successfully.',
            report_path: approved ? reportPath : null,
        });

    } catch (error) {
        console.error('Failed to save review:', error);
        res.status(500).json({ error: 'Failed to save review.' });
    }
});


export default router;