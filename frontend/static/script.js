// --- Global State ---
let currentCaseId = null;
let currentTraceId = null;
let lastKnownFindings = "";
let agentState = {};
let progressSteps = ['planner', 'search', 'synthesis', 'mcp_identifier', 'mcp_executor', 'final_updater'];
let currentStep = 0;

// --- DOM Elements ---
const getEl = (id) => document.getElementById(id);
const researchTopicEl = getEl('research-topic');
const startResearchBtn = getEl('start-research-btn');
const statusSectionEl = getEl('status-section');
const liveLogEl = getEl('live-log');
const reviewSectionEl = getEl('review-section');
const findingsDisplayEl = getEl('findings-display');
const completionSectionEl = getEl('completion-section');
const completionMessageEl = getEl('completion-message');
const downloadLinkEl = getEl('download-link');
const llmModelSelectEl = getEl('llm-model-select');
const temperatureSliderEl = getEl('temperature-slider');
const temperatureValueEl = getEl('temperature-value');

// --- Enhanced Agent with Conversation Memory ---
let currentConversationId = null;
let conversationHistory = [];
let cachedFindings = {};
let isConversationMode = false;

// Enhanced DOM Elements
const startNewConversationBtn = getEl('start-new-conversation-btn');
const loadConversationBtn = getEl('load-conversation-btn');
const conversationIdInput = getEl('conversation-id-input');
const loadConversationConfirmBtn = getEl('load-conversation-confirm-btn');
const currentConversationInfo = getEl('current-conversation-info');
const conversationIdDisplay = getEl('conversation-id-display');
const conversationStats = getEl('conversation-stats');
const conversationHistoryEl = getEl('conversation-history');
const chatMessagesEl = getEl('chat-messages');
const enhancedInputSection = getEl('enhanced-input-section');
const enhancedResearchTopicEl = getEl('enhanced-research-topic');
const sendEnhancedQueryBtn = getEl('send-enhanced-query-btn');
const continueConversationBtn = getEl('continue-conversation-btn');
const contextIndicator = getEl('context-indicator');
const enhancedStatusSection = getEl('enhanced-status-section');
const enhancedLiveLogEl = getEl('enhanced-live-log');
const enhancedResultsSection = getEl('enhanced-results-section');
const enhancedFindingsDisplayEl = getEl('enhanced-findings-display');
const memoryStatsEl = getEl('memory-stats');
const cachedFindingsEl = getEl('cached-findings');
const cachedFindingsListEl = getEl('cached-findings-list');

// --- Functions ---

function resetUI() {
    statusSectionEl.classList.add('hidden');
    reviewSectionEl.classList.add('hidden');
    completionSectionEl.classList.add('hidden');
    downloadLinkEl.classList.add('hidden');
    liveLogEl.innerHTML = '';
    findingsDisplayEl.innerHTML = '';
    startResearchBtn.disabled = false;
    currentCaseId = null;
    currentTraceId = null;
    lastKnownFindings = "";
    agentState = {};
    currentStep = 0;
}

function logMessage(message, type = 'status') {
    const timestamp = new Date().toLocaleTimeString();
    const p = document.createElement('p');
    p.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;
    p.className = `log-${type}`;
    liveLogEl.appendChild(p);
    liveLogEl.scrollTop = liveLogEl.scrollHeight;
}

function updateProgressIndicator(step) {
    const stepNames = {
        'planner': '📋 Planning Research Strategy',
        'search': '🔍 Executing Search Queries', 
        'synthesis': '🧠 Synthesizing Intelligence',
        'mcp_identifier': '🎯 Identifying Verification Targets',
        'mcp_executor': '🛡️ Running Verification Tools',
        'final_updater': '📊 Compiling Final Report'
    };
    
    const stepName = stepNames[step] || step;
    logMessage(`<strong>${stepName}</strong>`, 'progress');
    
    const stepIndex = progressSteps.indexOf(step);
    if (stepIndex !== -1) {
        currentStep = stepIndex;
        updateProgress(stepIndex);
    }
}

async function populateModels() {
    try {
        const response = await fetch('/models');
        if (!response.ok) throw new Error('Failed to fetch models.');
        const models = await response.json();
        
        llmModelSelectEl.innerHTML = '';
        for (const modelId in models) {
            const modelInfo = models[modelId];
            const option = document.createElement('option');
            option.value = modelId;
            option.textContent = `${modelInfo.provider}: ${modelInfo.model_name}`;
            llmModelSelectEl.appendChild(option);
        }
        
        // Set default model if available
        if (Object.keys(models).length > 0) {
            llmModelSelectEl.value = Object.keys(models)[0];
        }
    } catch (error) {
        console.error('Error populating models:', error);
        llmModelSelectEl.innerHTML = '<option value="">Could not load models</option>';
        logMessage('⚠️ Could not load available models. Please check backend connection.', 'error');
    }
}

async function startResearch() {
    const topic = researchTopicEl.value.trim();
    const model_id = llmModelSelectEl.value;
    const temperature = parseFloat(temperatureSliderEl.value);

    // Enhanced input validation
    if (!topic || topic.length < 3) {
        alert('Please enter a research topic (minimum 3 characters).');
        researchTopicEl.focus();
        return;
    }
    
    if (!model_id) {
        alert('Please select an AI model.');
        llmModelSelectEl.focus();
        return;
    }
    
    if (temperature < 0 || temperature > 1) {
        alert('Temperature must be between 0.0 and 1.0.');
        temperatureSliderEl.focus();
        return;
    }

    // Generate a unique case ID
    currentCaseId = `case_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    resetUI();
    statusSectionEl.classList.remove('hidden');
    enhanceStatusSection(); // Add progress bar
    startResearchBtn.disabled = true;
    
    logMessage(`🚀 Initializing OSINT investigation: "${topic}"`, 'start');
    logMessage(`📡 Using AI Model: ${llmModelSelectEl.options[llmModelSelectEl.selectedIndex].text}`, 'config');
    logMessage(`🌡️ Temperature: ${temperature} (Creativity Level)`, 'config');

    try {
        const requestBody = { 
            topic, 
            case_id: currentCaseId,
            model_id, 
            temperature,
            long_term_memory: [] // Could be enhanced to store previous investigations
        };

        const response = await fetch('/run_agent_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok || !response.body) {
            throw new Error(`Failed to connect to agent stream. Status: ${response.status}`);
        }

        const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
        let buffer = '';
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) {
                logMessage('🏁 Agent investigation completed successfully.', 'complete');
                break;
            }
            
            buffer += value;
            let boundary = buffer.lastIndexOf('\n\n');
            if (boundary !== -1) {
                let messages = buffer.substring(0, boundary);
                buffer = buffer.substring(boundary + 2);
                processStreamChunk(messages);
            }
        }
    } catch (e) {
        console.error("Streaming failed:", e);
        logMessage(`❌ Failed to connect to agent: ${e.message}`, 'error');
        startResearchBtn.disabled = false;
    }
}

function processStreamChunk(chunk) {
    const lines = chunk.split('\n\n').filter(line => line.trim() !== '');
    for (const line of lines) {
        if (line.startsWith('event: ')) {
            try {
                const eventLine = line.substring('event: '.length);
                const eventType = eventLine.substring(0, eventLine.indexOf('\n'));
                const dataLine = eventLine.substring(eventLine.indexOf('\n') + 'data: '.length);
                const eventData = JSON.parse(dataLine);

                if (eventType === 'task_created') {
                    currentCaseId = eventData.case_id;
                    currentTraceId = eventData.trace_id;
                    logMessage(`✅ Investigation task created (ID: ${currentCaseId})`, 'created');
                    if (eventData.trace_id) {
                        logMessage(`📍 Langfuse trace ID: ${eventData.trace_id}`, 'trace');
                    }
                } else if (eventType === 'status_update') {
                    const message = eventData.message;
                    
                    // Extract step information from status messages
                    if (message.includes('Planning')) {
                        updateProgressIndicator('planner');
                    } else if (message.includes('search')) {
                        updateProgressIndicator('search');
                    } else if (message.includes('Synthesizing')) {
                        updateProgressIndicator('synthesis');
                    } else if (message.includes('verification')) {
                        updateProgressIndicator('mcp_identifier');
                    } else if (message.includes('MCPs')) {
                        updateProgressIndicator('mcp_executor');
                    } else if (message.includes('final') || message.includes('report')) {
                        updateProgressIndicator('final_updater');
                    }
                    
                    logMessage(`🤖 ${message}`, 'status');
                } else if (eventType === 'review_required') {
                    logMessage('✨ Investigation completed! Generated comprehensive intelligence report.', 'complete');
                    lastKnownFindings = eventData.synthesized_findings;
                    currentTraceId = eventData.trace_id;
                    agentState.success = eventData.success;
                    
                    displayIntelligenceReport(eventData);
                    reviewSectionEl.classList.remove('hidden');
                } else if (eventType === 'error') {
                    logMessage(`💥 Investigation error: ${eventData.message}`, 'error');
                    startResearchBtn.disabled = false;
                }
            } catch (e) {
                console.error("Failed to parse stream chunk:", line, e);
                logMessage(`⚠️ Error parsing agent response: ${e.message}`, 'error');
            }
        }
    }
}

function displayIntelligenceReport(eventData) {
    const findings = eventData.synthesized_findings || "No findings available.";
    const success = eventData.success !== false;
    
    // Create a rich display of the intelligence report
    const reportHtml = `
        <div class="intelligence-report">
            <div class="report-header">
                <h3>🛡️ Intelligence Assessment Report</h3>
                <div class="report-meta">
                    <span class="status-badge ${success ? 'success' : 'warning'}">
                        ${success ? '✅ Investigation Successful' : '⚠️ Partial Results'}
                    </span>
                    ${currentTraceId ? `<span class="trace-id">Trace: ${currentTraceId}</span>` : ''}
                </div>
            </div>
            <div class="report-content">
                <pre class="findings-text">${findings}</pre>
            </div>
            <div class="report-actions">
                <p class="review-prompt">
                    📋 Please review the intelligence assessment above. 
                    Approving will store this analysis in the system's long-term memory for future investigations.
                </p>
            </div>
        </div>
    `;
    
    findingsDisplayEl.innerHTML = reportHtml;
}

async function handleReview(isApproved) {
    if (!lastKnownFindings || !currentCaseId) {
        alert("No findings or case ID available to review.");
        return;
    }

    const feedbackType = isApproved ? "approve" : "reject";
    logMessage(`📝 Submitting ${feedbackType} feedback...`, 'status');
    reviewSectionEl.classList.add('hidden');

    try {
        const feedbackData = {
            case_id: currentCaseId,
            feedback_type: feedbackType,
            feedback_data: {
                findings: lastKnownFindings,
                approved: isApproved,
                timestamp: new Date().toISOString(),
                user_action: isApproved ? "approved_for_memory" : "rejected_findings"
            },
            trace_id: currentTraceId
        };

        const response = await fetch('/submit_feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(feedbackData),
        });

        if (!response.ok) throw new Error('Failed to submit feedback.');
        const data = await response.json();
        
        let message = "";
        if (isApproved) {
            message = "✅ Intelligence report approved and stored in long-term memory. The findings will be available for future investigations.";
            logMessage("💾 Report stored in agent memory successfully.", 'success');
        } else {
            message = "❌ Intelligence report rejected. No data was stored in memory.";
            logMessage("🗑️ Report rejected - no memory storage.", 'reject');
        }
        
        completionMessageEl.innerHTML = `
            <div class="completion-details">
                <p>${message}</p>
                <div class="completion-meta">
                    <p><strong>Case ID:</strong> ${currentCaseId}</p>
                    ${currentTraceId ? `<p><strong>Trace ID:</strong> ${currentTraceId}</p>` : ''}
                    <p><strong>Status:</strong> ${data.status}</p>
                    <p><strong>Timestamp:</strong> ${new Date().toLocaleString()}</p>
                </div>
            </div>
        `;
        
        // Hide download link as the current backend doesn't generate reports
        downloadLinkEl.classList.add('hidden');
        completionSectionEl.classList.remove('hidden');

    } catch (error) {
        console.error('Review handling failed:', error);
        logMessage(`💥 Error submitting feedback: ${error.message}`, 'error');
        
        // Show completion section with error message
        completionMessageEl.innerHTML = `
            <div class="error-details">
                <p>❌ Failed to submit review feedback.</p>
                <p><strong>Error:</strong> ${error.message}</p>
                <p>The intelligence report was generated successfully but could not be processed for storage.</p>
            </div>
        `;
        completionSectionEl.classList.remove('hidden');
    } finally {
        startResearchBtn.disabled = false;
    }
}

// --- Additional Utility Functions ---

function validateInput() {
    const topic = researchTopicEl.value.trim();
    const model = llmModelSelectEl.value;
    
    if (!topic) {
        researchTopicEl.classList.add('error');
        return false;
    } else {
        researchTopicEl.classList.remove('error');
    }
    
    if (!model) {
        llmModelSelectEl.classList.add('error');
        return false;
    } else {
        llmModelSelectEl.classList.remove('error');
    }
    
    return true;
}

function formatTimestamp(timestamp) {
    return new Date(timestamp).toLocaleString();
}

function createProgressBar() {
    const progressHtml = `
        <div class="progress-container">
            <div class="progress-bar">
                <div class="progress-fill" style="width: 0%"></div>
            </div>
            <div class="progress-steps">
                <span class="step active">Plan</span>
                <span class="step">Search</span>
                <span class="step">Analyze</span>
                <span class="step">Verify</span>
                <span class="step">Execute</span>
                <span class="step">Report</span>
            </div>
        </div>
    `;
    
    const progressContainer = document.createElement('div');
    progressContainer.innerHTML = progressHtml;
    statusSectionEl.insertBefore(progressContainer, liveLogEl);
}

function updateProgress(stepIndex) {
    const progressFill = document.querySelector('.progress-fill');
    const steps = document.querySelectorAll('.progress-steps .step');
    
    if (progressFill && steps.length > 0) {
        const progressPercent = ((stepIndex + 1) / steps.length) * 100;
        progressFill.style.width = `${progressPercent}%`;
        
        steps.forEach((step, index) => {
            if (index <= stepIndex) {
                step.classList.add('completed');
            } else {
                step.classList.remove('completed');
            }
        });
    }
}

function exportInvestigationLog() {
    const logEntries = Array.from(liveLogEl.children).map(entry => entry.textContent);
    const reportData = {
        caseId: currentCaseId,
        traceId: currentTraceId,
        topic: researchTopicEl.value.trim(),
        timestamp: new Date().toISOString(),
        logEntries: logEntries,
        findings: lastKnownFindings,
        agentState: agentState
    };
    
    const dataStr = JSON.stringify(reportData, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `osint_investigation_${currentCaseId}.json`;
    link.click();
    
    URL.revokeObjectURL(url);
}

// Enhanced status section with progress tracking
function enhanceStatusSection() {
    if (!document.querySelector('.progress-container')) {
        createProgressBar();
    }
}

// --- Enhanced Agent Functions ---
function resetEnhancedUI() {
    currentConversationInfo.classList.add('hidden');
    conversationHistoryEl.classList.add('hidden');
    enhancedInputSection.classList.add('hidden');
    enhancedStatusSection.classList.add('hidden');
    enhancedResultsSection.classList.add('hidden');
    cachedFindingsEl.classList.add('hidden');
    contextIndicator.classList.add('hidden');
    continueConversationBtn.classList.add('hidden');
    chatMessagesEl.innerHTML = '';
    enhancedLiveLogEl.innerHTML = '';
    enhancedFindingsDisplayEl.innerHTML = '';
    cachedFindingsListEl.innerHTML = '';
    currentConversationId = null;
    conversationHistory = [];
    cachedFindings = {};
    isConversationMode = false;
}

function logEnhancedMessage(message, type = 'status') {
    const timestamp = new Date().toLocaleTimeString();
    const p = document.createElement('p');
    p.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;
    p.className = `log-${type}`;
    enhancedLiveLogEl.appendChild(p);
    enhancedLiveLogEl.scrollTop = enhancedLiveLogEl.scrollHeight;
}

function addChatMessage(role, content, metadata = {}) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}`;
    
    const roleIcon = role === 'user' ? '👤' : role === 'assistant' ? '🤖' : '⚙️';
    const timestamp = new Date().toLocaleTimeString();
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <span class="role-icon">${roleIcon}</span>
            <span class="role-name">${role.charAt(0).toUpperCase() + role.slice(1)}</span>
            <span class="message-time">${timestamp}</span>
        </div>
        <div class="message-content">${content}</div>
        ${metadata.case_id ? `<div class="message-meta">Case: ${metadata.case_id}</div>` : ''}
    `;
    
    chatMessagesEl.appendChild(messageDiv);
    chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
}

function updateConversationStats(stats) {
    if (stats && conversationStats) {
        const totalMessages = stats.total_messages || 0;
        const cachedCount = stats.cached_findings_count || 0;
        conversationStats.textContent = `${totalMessages} messages • ${cachedCount} cached findings`;
    }
}

function displayMemoryStats(memoryInfo) {
    if (!memoryInfo) return;
    
    const stats = memoryInfo.memory_optimization || {};
    const cachedUtilized = stats.cached_findings_utilized || 0;
    const newCached = stats.new_findings_cached || 0;
    const contextScore = stats.context_reuse_score || 0;
    
    memoryStatsEl.innerHTML = `
        <div class="memory-stat">
            <span class="stat-label">Context Reuse Score:</span>
            <span class="stat-value">${(contextScore * 100).toFixed(0)}%</span>
        </div>
        <div class="memory-stat">
            <span class="stat-label">Cached Findings Used:</span>
            <span class="stat-value">${cachedUtilized}</span>
        </div>
        <div class="memory-stat">
            <span class="stat-label">New Findings Cached:</span>
            <span class="stat-value">${newCached}</span>
        </div>
    `;
}

function displayCachedFindings(findings) {
    if (!findings || Object.keys(findings).length === 0) {
        cachedFindingsEl.classList.add('hidden');
        return;
    }
    
    cachedFindingsEl.classList.remove('hidden');
    cachedFindingsListEl.innerHTML = '';
    
    Object.entries(findings).forEach(([key, finding]) => {
        const findingDiv = document.createElement('div');
        findingDiv.className = 'cached-finding-item';
        
        const content = typeof finding === 'object' ? finding.content || JSON.stringify(finding) : finding;
        const truncatedContent = content.length > 150 ? content.substring(0, 150) + '...' : content;
        
        findingDiv.innerHTML = `
            <div class="finding-key">${key}</div>
            <div class="finding-content">${truncatedContent}</div>
        `;
        
        cachedFindingsListEl.appendChild(findingDiv);
    });
}

async function startNewConversation() {
    try {
        resetEnhancedUI();
        logEnhancedMessage('🆕 Starting new conversation...', 'status');
        
        // Show the enhanced input immediately for new conversations
        enhancedInputSection.classList.remove('hidden');
        isConversationMode = true;
        
        logEnhancedMessage('✅ Ready for new conversation. Enter your query above.', 'success');
        
    } catch (error) {
        console.error('Error starting new conversation:', error);
        logEnhancedMessage(`❌ Error starting conversation: ${error.message}`, 'error');
    }
}

async function loadConversation() {
    const conversationId = conversationIdInput.value.trim();
    if (!conversationId) {
        alert('Please enter a conversation ID');
        return;
    }
    
    try {
        logEnhancedMessage(`📁 Loading conversation ${conversationId}...`, 'status');
        
        const response = await fetch(`/conversation/${conversationId}/history`);
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Failed to load conversation');
        }
        
        const historyData = result.data;
        currentConversationId = conversationId;
        conversationHistory = historyData.messages || [];
        cachedFindings = historyData.cached_findings || {};
        
        // Update UI
        conversationIdDisplay.textContent = `ID: ${conversationId}`;
        currentConversationInfo.classList.remove('hidden');
        conversationHistoryEl.classList.remove('hidden');
        enhancedInputSection.classList.remove('hidden');
        continueConversationBtn.classList.remove('hidden');
        contextIndicator.classList.remove('hidden');
        isConversationMode = true;
        
        // Display conversation history
        chatMessagesEl.innerHTML = '';
        conversationHistory.forEach(msg => {
            addChatMessage(msg.role, msg.content, msg.metadata || {});
        });
        
        // Update stats
        updateConversationStats(historyData.conversation_metadata);
        displayCachedFindings(cachedFindings);
        
        // Hide load controls
        conversationIdInput.classList.add('hidden');
        loadConversationConfirmBtn.classList.add('hidden');
        
        logEnhancedMessage(`✅ Loaded conversation with ${conversationHistory.length} messages`, 'success');
        
    } catch (error) {
        console.error('Error loading conversation:', error);
        logEnhancedMessage(`❌ Error loading conversation: ${error.message}`, 'error');
    }
}

async function sendEnhancedQuery() {
    const topic = enhancedResearchTopicEl.value.trim();
    if (!topic) {
        alert('Please enter a query or investigation topic');
        return;
    }
    
    const model_id = llmModelSelectEl.value;
    const temperature = parseFloat(temperatureSliderEl.value);
    
    try {
        enhancedStatusSection.classList.remove('hidden');
        enhancedResultsSection.classList.add('hidden');
        sendEnhancedQueryBtn.disabled = true;
        continueConversationBtn.disabled = true;
        
        // Add user message to chat
        addChatMessage('user', topic, {});
        
        logEnhancedMessage('🚀 Starting enhanced investigation...', 'status');
        
        let endpoint, requestBody;
        
        if (currentConversationId) {
            // Continue existing conversation
            endpoint = '/enhanced_agent/continue';
            requestBody = {
                conversation_id: currentConversationId,
                new_query: topic,
                model_id: model_id,
                temperature: temperature
            };
            logEnhancedMessage('🔄 Continuing existing conversation...', 'status');
        } else {
            // Start new conversation
            endpoint = '/enhanced_agent/start';
            requestBody = {
                topic: topic,
                model_id: model_id,
                temperature: temperature
            };
            logEnhancedMessage('🆕 Starting new investigation...', 'status');
        }
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Failed to run enhanced agent');
        }
        
        const data = result.data;
        
        // Update conversation state
        if (data.conversation_id) {
            currentConversationId = data.conversation_id;
            conversationIdDisplay.textContent = `ID: ${currentConversationId}`;
            currentConversationInfo.classList.remove('hidden');
            conversationHistoryEl.classList.remove('hidden');
            continueConversationBtn.classList.remove('hidden');
            
            if (data.context_reuse_applied) {
                contextIndicator.classList.remove('hidden');
                logEnhancedMessage('🧠 Applied conversation context optimization', 'success');
            }
        }
        
        // Display results
        enhancedResultsSection.classList.remove('hidden');
        enhancedFindingsDisplayEl.innerHTML = `
            <div class="findings-content">
                <h4>🎯 Investigation Results</h4>
                <div class="findings-text">${data.synthesized_findings || 'No findings available'}</div>
                
                ${data.synthesis_confidence ? `
                <div class="confidence-section">
                    <h5>📊 Confidence Assessment</h5>
                    <p>${data.synthesis_confidence}</p>
                </div>
                ` : ''}
                
                ${data.planner_reasoning ? `
                <div class="reasoning-section">
                    <h5>🤔 Planning Reasoning</h5>
                    <p>${data.planner_reasoning}</p>
                </div>
                ` : ''}
            </div>
        `;
        
        // Add assistant message to chat
        const assistantMessage = `Investigation completed. Found ${data.search_quality_metrics?.total_results || 0} results with ${data.memory_optimization?.new_findings_cached || 0} new findings cached.`;
        addChatMessage('assistant', assistantMessage, { case_id: data.case_id });
        
        // Display memory optimization info
        if (data.memory_optimization) {
            displayMemoryStats(data);
        }
        
        // Update cached findings if available
        if (data.cached_findings_count > 0) {
            // Refresh conversation history to get updated cached findings
            setTimeout(async () => {
                try {
                    const historyResponse = await fetch(`/conversation/${currentConversationId}/history`);
                    const historyResult = await historyResponse.json();
                    if (historyResult.success) {
                        cachedFindings = historyResult.data.cached_findings || {};
                        displayCachedFindings(cachedFindings);
                        updateConversationStats(historyResult.data.conversation_metadata);
                    }
                } catch (e) {
                    console.warn('Could not refresh cached findings:', e);
                }
            }, 1000);
        }
        
        logEnhancedMessage('✅ Enhanced investigation completed successfully', 'success');
        
        // Clear input
        enhancedResearchTopicEl.value = '';
        
    } catch (error) {
        console.error('Error running enhanced agent:', error);
        logEnhancedMessage(`❌ Error: ${error.message}`, 'error');
        
        // Add error message to chat
        addChatMessage('system', `Error: ${error.message}`, {});
        
    } finally {
        sendEnhancedQueryBtn.disabled = false;
        continueConversationBtn.disabled = false;
    }
}

// --- Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
    populateModels();
    
    // Add input validation
    researchTopicEl.addEventListener('input', () => {
        if (researchTopicEl.value.trim()) {
            researchTopicEl.classList.remove('error');
        }
    });
    
    llmModelSelectEl.addEventListener('change', () => {
        if (llmModelSelectEl.value) {
            llmModelSelectEl.classList.remove('error');
        }
    });
});

temperatureSliderEl.addEventListener('input', (e) => {
    temperatureValueEl.textContent = e.target.value;
});

startResearchBtn.addEventListener('click', () => {
    if (validateInput()) {
        startResearch();
    } else {
        logMessage('⚠️ Please fill in all required fields before starting the investigation.', 'error');
    }
});

getEl('approve-btn').addEventListener('click', () => handleReview(true));
getEl('reject-btn').addEventListener('click', () => handleReview(false));

// Enhanced Agent Event Listeners
startNewConversationBtn.addEventListener('click', startNewConversation);

loadConversationBtn.addEventListener('click', () => {
    conversationIdInput.classList.toggle('hidden');
    loadConversationConfirmBtn.classList.toggle('hidden');
    if (!conversationIdInput.classList.contains('hidden')) {
        conversationIdInput.focus();
    }
});

loadConversationConfirmBtn.addEventListener('click', loadConversation);

conversationIdInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        loadConversation();
    }
});

sendEnhancedQueryBtn.addEventListener('click', sendEnhancedQuery);
continueConversationBtn.addEventListener('click', sendEnhancedQuery);

enhancedResearchTopicEl.addEventListener('keypress', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        sendEnhancedQuery();
    }
});

// Enhanced keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) {
        switch(e.key) {
            case 'Enter':
                if (!startResearchBtn.disabled && validateInput()) {
                    e.preventDefault();
                    startResearch();
                }
                break;
            case 's':
                if (currentCaseId) {
                    e.preventDefault();
                    exportInvestigationLog();
                }
                break;
        }
    }
});
