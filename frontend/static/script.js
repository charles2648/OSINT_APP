// --- Global State ---
let currentCaseId = null;
let lastKnownFindings = "";

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
    lastKnownFindings = "";
}

function logMessage(message, type = 'status') {
    const p = document.createElement('p');
    p.textContent = message;
    p.className = `log-${type}`;
    liveLogEl.appendChild(p);
    liveLogEl.scrollTop = liveLogEl.scrollHeight;
}

async function populateModels() {
    try {
        const response = await fetch('/api/models');
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
    } catch (error) {
        console.error('Error populating models:', error);
        llmModelSelectEl.innerHTML = '<option value="">Could not load models</option>';
    }
}

async function startResearch() {
    const topic = researchTopicEl.value.trim();
    const model_id = llmModelSelectEl.value;
    const temperature = parseFloat(temperatureSliderEl.value);

    if (!topic || !model_id) {
        alert('Please enter a topic and select a model.');
        return;
    }

    resetUI();
    statusSectionEl.classList.remove('hidden');
    startResearchBtn.disabled = true;
    logMessage(`Initializing research on "${topic}" using ${model_id}...`);

    try {
        const response = await fetch('/api/agent/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic, model_id, temperature }),
        });

        if (!response.ok || !response.body) {
            throw new Error(`Failed to connect to agent stream. Status: ${response.status}`);
        }

        const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
        let buffer = '';
        while (true) {
            const { value, done } = await reader.read();
            if (done) {
                logMessage('Agent stream finished.', 'complete');
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
        logMessage(`Failed to connect to agent: ${e.message}`, 'error');
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
                    logMessage(`Agent task created with Case ID: ${currentCaseId}`, 'created');
                } else if (eventType === 'status_update') {
                    logMessage(`Agent: ${eventData.message}`);
                } else if (eventType === 'review_required') {
                    logMessage('Agent has completed its research. Please review.', 'review');
                    lastKnownFindings = eventData.synthesized_findings;
                    findingsDisplayEl.innerText = lastKnownFindings;
                    reviewSectionEl.classList.remove('hidden');
                } else if (eventType === 'error') {
                    logMessage(`An error occurred: ${eventData.message}`, 'error');
                    startResearchBtn.disabled = false;
                }
            } catch (e) {
                console.error("Failed to parse stream chunk:", line, e);
            }
        }
    }
}

async function handleReview(isApproved) {
    if (!lastKnownFindings || !currentCaseId) {
        alert("No findings or case ID available to review.");
        return;
    }

    logMessage('Submitting your review...', 'status');
    reviewSectionEl.classList.add('hidden');

    try {
        const response = await fetch(`/api/agent/review`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ caseId: currentCaseId, findings: lastKnownFindings, approved: isApproved }),
        });

        if (!response.ok) throw new Error('Failed to submit review.');
        const data = await response.json();
        
        completionMessageEl.textContent = data.message;
        if(data.report_path && isApproved) {
            downloadLinkEl.href = data.report_path;
            downloadLinkEl.classList.remove('hidden');
            completionMessageEl.textContent += " Report generated successfully."
        }
        completionSectionEl.classList.remove('hidden');

    } catch (error) {
        console.error('Review handling failed:', error);
        logMessage(`Error submitting review: ${error.message}`, 'error');
    } finally {
        startResearchBtn.disabled = false;
    }
}

// --- Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
    populateModels();
});

temperatureSliderEl.addEventListener('input', (e) => {
    temperatureValueEl.textContent = e.target.value;
});

startResearchBtn.addEventListener('click', startResearch);
getEl('approve-btn').addEventListener('click', () => handleReview(true));
getEl('reject-btn').addEventListener('click', () => handleReview(false));
