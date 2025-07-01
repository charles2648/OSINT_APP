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
