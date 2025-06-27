# Express Backend - Enhanced Implementation

## Overview

The Express backend has been significantly improved to provide better integration with the OSINT agent, enhanced security, comprehensive monitoring, and robust error handling.

## Key Improvements

### 1. Enhanced Database Schema
- **Comprehensive tracking**: Research cases, agent runs, search queries, MCP verifications, user feedback
- **Performance optimization**: Proper indexes and foreign key relationships
- **Audit trail**: Complete tracking of all operations
- **Statistics support**: Built-in analytics and reporting capabilities

### 2. Security Enhancements
- **Helmet.js**: Security headers and content security policy
- **Rate limiting**: Prevents abuse and DoS attacks
- **Request validation**: Joi schema validation for all inputs
- **CORS configuration**: Environment-aware CORS settings
- **Input sanitization**: Protection against injection attacks

### 3. Monitoring & Health Checks
- **Health endpoint**: `/health` provides system status
- **AI Worker monitoring**: Real-time status of FastAPI worker
- **Database health**: Connection and performance monitoring
- **System statistics**: Performance metrics and usage analytics

### 4. API Improvements
- **RESTful design**: Proper HTTP methods and status codes
- **Comprehensive error handling**: Detailed error responses
- **Request/response logging**: Better debugging and monitoring
- **Backward compatibility**: Legacy endpoints still supported
- **Documentation**: Self-documenting API endpoints

### 5. Enhanced Integration
- **Stream handling**: Better error handling for streaming responses
- **Database tracking**: Complete tracking of agent execution
- **Feedback integration**: Comprehensive user feedback system
- **Memory management**: Improved approved memory retrieval

## New API Endpoints

### Core Endpoints
- `GET /api` - API documentation
- `GET /api/models` - Available LLM models
- `POST /api/agent/start` - Start OSINT investigation
- `POST /api/submit_feedback` - Submit user feedback

### Management Endpoints
- `GET /api/cases` - List research cases
- `GET /api/cases/:caseId` - Get case details
- `GET /api/stats` - System statistics
- `GET /health` - System health check

### Legacy Support
- `POST /api/agent/review` - Backward compatibility (deprecated)

## Database Schema

### New Tables
1. **research_cases** - Investigation topics and metadata
2. **agent_runs** - Individual agent executions
3. **search_queries** - Track all search operations
4. **mcp_verifications** - MCP tool execution tracking
5. **user_feedback** - Comprehensive feedback system
6. **system_health** - System monitoring data

### Key Features
- Proper foreign key relationships
- Performance indexes
- Automatic timestamp management
- Data integrity constraints

## Configuration

### Environment Variables
```bash
PORT=3000                           # Server port
NODE_ENV=development               # Environment mode
DATABASE_PATH=./osint_database.sqlite # Database location
FASTAPI_WORKER_URL=http://127.0.0.1:8001 # AI worker URL
ALLOWED_ORIGINS=http://localhost:3000 # CORS origins
RATE_LIMIT_MAX=1000               # Rate limit per 15 minutes
```

### Security Settings
- Helmet.js for security headers
- Rate limiting (100 req/15min in production)
- Request size limits (10MB)
- Input validation with Joi schemas

## Error Handling

### Graceful Degradation
- AI Worker offline detection
- Database connection recovery
- Stream error handling
- Request timeout management

### Error Response Format
```json
{
  "error": "Error description",
  "details": "Additional details",
  "timestamp": "2025-06-26T23:00:00.000Z",
  "path": "/api/endpoint",
  "method": "POST"
}
```

## Usage Examples

### Start Investigation
```bash
curl -X POST http://localhost:3000/api/agent/start \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Domain investigation for suspicious-site.com",
    "model_id": "gpt-4o-mini",
    "temperature": 0.3,
    "description": "Investigating potential malware distribution",
    "priority": 3
  }'
```

### Submit Feedback
```bash
curl -X POST http://localhost:3000/api/submit_feedback \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "123e4567-e89b-12d3-a456-426614174000",
    "feedback_type": "approve",
    "rating": 5,
    "comments": "Excellent analysis quality"
  }'
```

### Check System Health
```bash
curl http://localhost:3000/health
```

## Installation & Setup

1. **Install enhanced dependencies:**
   ```bash
   cd express-backend
   npm install
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start the server:**
   ```bash
   npm start
   # or for development
   npm run dev
   ```

## Monitoring

### Health Check Response
```json
{
  "status": "healthy",
  "timestamp": "2025-06-26T23:00:00.000Z",
  "environment": "development",
  "database": {
    "status": "connected",
    "stats": {
      "total_cases": 42,
      "active_cases": 15,
      "total_runs": 156,
      "successful_runs": 142,
      "success_rate": "91.03"
    }
  },
  "aiWorker": {
    "status": "online",
    "url": "http://127.0.0.1:8001",
    "responseTime": 145
  },
  "uptime": 3600,
  "memory": {
    "rss": 45678592,
    "heapTotal": 25165824,
    "heapUsed": 18543216
  }
}
```

## Performance Optimizations

1. **Database indexes** on frequently queried columns
2. **Connection pooling** for database operations
3. **Stream handling** for large responses
4. **Request validation** to prevent invalid processing
5. **Graceful shutdown** handling

## Security Features

1. **Helmet.js** - Security headers
2. **Rate limiting** - Prevent abuse
3. **Input validation** - Prevent injection
4. **CORS protection** - Cross-origin security
5. **Error sanitization** - No sensitive data leakage

## Troubleshooting

### Database Path Issues
If you encounter `SQLITE_CANTOPEN` errors during startup, check for conflicting environment variables:
```bash
# Clear any conflicting DATABASE_PATH environment variable
unset DATABASE_PATH

# Restart the server
npm start
```

The server expects to create the database in the project directory (`./osint_database.sqlite`).

### AI Worker Connection
If the AI worker is unavailable, the server will still start but some endpoints will return 503 status codes. Ensure the FastAPI worker is running on the expected port (8001 by default).

### Port Conflicts
If port 3000 is already in use, set a different port:
```bash
PORT=3001 npm start
```

The enhanced Express backend now provides enterprise-grade capabilities for OSINT investigations with robust monitoring, security, and integration features.
