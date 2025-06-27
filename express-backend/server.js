// The main entry point for the Express.js backend with enhanced security and monitoring.

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import path from 'path';
import { fileURLToPath } from 'url';
import 'dotenv/config';

import { initializeDatabase, dbUtils } from './database.js';
import apiRoutes from './api/routes.js';
import { randomUUID } from 'crypto';
import axios from 'axios';

const app = express();
const PORT = process.env.PORT || 3000;
const NODE_ENV = process.env.NODE_ENV || 'development';

// Environment validation and configuration
const config = {
  port: PORT,
  nodeEnv: NODE_ENV,
  dbPath: process.env.DATABASE_PATH || './osint_database.sqlite',
  aiWorkerUrl: process.env.FASTAPI_WORKER_URL || 'http://127.0.0.1:8001',
  rateLimitMax: parseInt(process.env.RATE_LIMIT_MAX) || (NODE_ENV === 'production' ? 100 : 1000),
  requestTimeout: parseInt(process.env.REQUEST_TIMEOUT_MS) || 300000,
  streamTimeout: parseInt(process.env.STREAM_TIMEOUT_MS) || 600000,
  approvedMemoryLimit: parseInt(process.env.APPROVED_MEMORY_LIMIT) || 10,
  searchResultsLimit: parseInt(process.env.SEARCH_RESULTS_LIMIT) || 50,
  logRequests: process.env.LOG_REQUESTS !== 'false',
  enableAnalytics: process.env.ENABLE_ANALYTICS !== 'false',
  enableExport: process.env.ENABLE_EXPORT !== 'false',
  enableAdminEndpoints: process.env.ENABLE_ADMIN_ENDPOINTS !== 'false'
};

// Log configuration in development
if (NODE_ENV === 'development') {
  console.log('📋 Configuration:', JSON.stringify(config, null, 2));
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ['\'self\''],
      scriptSrc: ['\'self\'', '\'unsafe-inline\''],
      styleSrc: ['\'self\'', '\'unsafe-inline\''],
      imgSrc: ['\'self\'', 'data:', 'https:']
    }
  }
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: config.rateLimitMax,
  message: {
    error: 'Too many requests from this IP, please try again later.',
    retryAfter: '15 minutes'
  },
  standardHeaders: true,
  legacyHeaders: false
});

app.use(limiter);

// CORS configuration
const corsOptions = {
  origin: NODE_ENV === 'production'
    ? process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000']
    : true,
  credentials: true,
  optionsSuccessStatus: 200
};

app.use(cors(corsOptions));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Request logging middleware
if (config.logRequests) {
  app.use((req, res, next) => {
    const timestamp = new Date().toISOString();
    console.log(`📥 ${timestamp} ${req.method} ${req.path} - ${req.ip}`);
    next();
  });
}

// Health check endpoint
app.get('/health', async(req, res) => {
  try {
    const stats = dbUtils.getSystemStats();
    const aiWorkerUrl = config.aiWorkerUrl;

    // Quick health check of AI worker
    let aiWorkerStatus = 'offline';
    let aiWorkerResponseTime = null;
    try {
      const start = Date.now();
      const response = await fetch(`${aiWorkerUrl}/models`, {
        signal: AbortSignal.timeout(5000)
      });
      aiWorkerResponseTime = Date.now() - start;
      aiWorkerStatus = response.ok ? 'online' : 'error';
    } catch (error) {
      aiWorkerStatus = 'offline';
    }

    res.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      environment: NODE_ENV,
      configuration: {
        rateLimitMax: config.rateLimitMax,
        requestTimeout: config.requestTimeout,
        streamTimeout: config.streamTimeout,
        enableAnalytics: config.enableAnalytics,
        enableExport: config.enableExport,
        enableAdminEndpoints: config.enableAdminEndpoints
      },
      database: {
        status: 'connected',
        path: config.dbPath,
        stats
      },
      aiWorker: {
        status: aiWorkerStatus,
        url: aiWorkerUrl,
        responseTime: aiWorkerResponseTime
      },
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      version: process.env.npm_package_version || '1.0.0'
    });
  } catch (error) {
    console.error('❌ Health check failed:', error);
    res.status(503).json({
      status: 'unhealthy',
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Serve static frontend files
const frontendPath = path.join(__dirname, '..', 'frontend');
app.use(express.static(frontendPath, {
  maxAge: NODE_ENV === 'production' ? '1h' : '0',
  etag: true
}));

// API routes
app.use('/api', apiRoutes);

// Frontend-compatible routes (without /api prefix)
// These routes mirror the API routes to maintain frontend compatibility
app.get('/models', async(req, res) => {
  try {
    const response = await fetch(`${config.aiWorkerUrl}/models`, {
      signal: AbortSignal.timeout(5000)
    });

    if (!response.ok) {
      throw new Error(`AI Worker returned ${response.status}`);
    }

    const models = await response.json();
    res.json(models);
  } catch (error) {
    console.error('❌ Failed to get models from AI worker:', error.message);
    res.status(503).json({
      error: 'AI Worker unavailable',
      message: 'The AI worker service is currently offline. Please try again later.',
      fallback: {
        'gpt-4o-mini': { provider: 'OpenAI', model_name: 'GPT-4o Mini (fallback)' }
      }
    });
  }
});

app.post('/run_agent_stream', express.json(), async(req, res) => {
  // Forward to the API route - manually handle since it's a streaming response
  const { topic, model_id, temperature, case_id, long_term_memory: _long_term_memory } = req.body;

  if (!topic || !model_id || temperature === undefined) {
    return res.status(400).json({ error: 'Topic, model_id, and temperature are required.' });
  }

  try {
    const _aiWorkerUrl = process.env.FASTAPI_WORKER_URL || 'http://127.0.0.1:8001';

    // Check AI worker availability
    const healthCheck = await fetch(`${config.aiWorkerUrl}/models`, {
      signal: AbortSignal.timeout(5000)
    });

    if (!healthCheck.ok) {
      return res.status(503).json({
        error: 'AI Worker unavailable',
        message: 'Cannot start investigation while AI worker is offline'
      });
    }

    // Create records using database utilities
    const caseId = case_id || randomUUID();
    const runId = randomUUID();

    dbUtils.createResearchCase(caseId, topic);
    dbUtils.createAgentRun(runId, caseId, model_id, temperature);

    // Get approved memory
    const memory = dbUtils.getApprovedMemory(5);

    console.log(`🚀 Starting investigation: ${caseId} with model: ${model_id}, temp: ${temperature}`);

    // Forward to AI worker
    const workerResponse = await axios.post(`${config.aiWorkerUrl}/run_agent_stream`, {
      topic,
      case_id: caseId,
      long_term_memory: memory,
      model_id,
      temperature
    }, {
      responseType: 'stream',
      timeout: config.streamTimeout
    });

    // Set up streaming response
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Case-ID', caseId);
    res.setHeader('X-Run-ID', runId);

    workerResponse.data.pipe(res);
  } catch (error) {
    console.error('❌ Failed to start agent:', error.message);
    res.status(500).json({
      error: `Failed to communicate with AI worker: ${error.message}`,
      timestamp: new Date().toISOString()
    });
  }
});

app.post('/submit_feedback', express.json(), async(req, res) => {
  // Forward to the API route
  req.url = '/api/submit_feedback';
  return apiRoutes(req, res, () => {
    res.status(404).json({ error: 'Route not found' });
  });
});

// Global error handler
app.use((error, req, res, next) => {
  console.error('❌ Unhandled error:', error);

  if (res.headersSent) {
    return next(error);
  }

  const statusCode = error.statusCode || 500;
  const errorResponse = {
    error: NODE_ENV === 'production' ? 'Internal server error' : error.message,
    timestamp: new Date().toISOString(),
    path: req.path,
    method: req.method
  };

  if (NODE_ENV !== 'production') {
    errorResponse.stack = error.stack;
  }

  res.status(statusCode).json(errorResponse);
});

// Catch 404 for API routes
app.use('/api/*', (req, res) => {
  res.status(404).json({
    error: 'API endpoint not found',
    path: req.path,
    availableEndpoints: [
      '/api/models',
      '/api/agent/start',
      '/api/submit_feedback',
      '/api/cases',
      '/api/stats',
      '/health'
    ]
  });
});

// Serve frontend for all other routes (SPA support)
app.get('*', (req, res) => {
  res.sendFile(path.join(frontendPath, 'index.html'));
});

async function startServer() {
  try {
    console.log('🚀 Starting Express server...');
    console.log(`📊 Environment: ${NODE_ENV}`);

    // Initialize database (still async since it's the initial setup)
    await initializeDatabase();

    const server = app.listen(PORT, () => {
      console.log(`✅ Express server running at http://localhost:${PORT}`);
      console.log(`🤖 AI Worker expected at ${config.aiWorkerUrl}`);
      console.log(`🏥 Health check available at http://localhost:${PORT}/health`);
      console.log(`📖 API documentation available at http://localhost:${PORT}/api`);

      if (NODE_ENV === 'development') {
        console.log('🔧 Development mode - Enhanced logging enabled');
        console.log(`📊 Admin endpoints: ${config.enableAdminEndpoints ? 'Enabled' : 'Disabled'}`);
        console.log(`📈 Analytics: ${config.enableAnalytics ? 'Enabled' : 'Disabled'}`);
        console.log(`📁 Export: ${config.enableExport ? 'Enabled' : 'Disabled'}`);
      }
    });

    // Graceful shutdown
    process.on('SIGTERM', () => {
      console.log('🔄 SIGTERM received, shutting down gracefully...');
      server.close(() => {
        console.log('👋 Express server closed');
        process.exit(0);
      });
    });

    process.on('SIGINT', () => {
      console.log('🔄 SIGINT received, shutting down gracefully...');
      server.close(() => {
        console.log('👋 Express server closed');
        process.exit(0);
      });
    });
  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
}

startServer();
