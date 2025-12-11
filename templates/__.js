// Status color constants
const STATUS_COLORS = {
    connected: {
        status: '#10b981', // Green
        user: '#10b981'    // Green
    },
    loading: {
        status: '#f59e0b', // Orange
        user: '#f59e0b'    // Orange
    },
    disconnected: {
        status: '#ef4444', // Red
        user: '#ef4444'    // Red
    }
};


let sessionStartTime = null;
let messageCount = 0;
let audioLevelsChart = null;
let isRecording = false;
let isAudioCurrentlyPlaying = false;
let configSaved = false;
let interruptRequested = false; 
let interruptInProgress = false;
let lastSeenGenId = 0;
let reconnecting = false;
let reconnectAttempts = 0;
let maxReconnectAttempts = 10;

const SESSION_ID = "default_" + Date.now();
console.log("chat.js loaded - Session ID:", SESSION_ID);

let micStream;
let selectedMicId = null;
let selectedOutputId = null;


let audioDataHistory = [];
let micAnalyser, micContext;
let activeGenId = 0;


// Initialize conversations as empty array - will be loaded from server
let conversations = [];
let currentFilter = localStorage.getItem('conversationFilter') || 'all';
let isFetchingConversations = false;
let conversationsLastUpdated = null;

// Audio playback variables
let audioContext = null;
let audioPlaybackQueue = [];
let currentAudioGenerationId = null;
let isPlayingAudio = false;
let currentAudioSource = null;
let audioScheduledTime = 0;

// Function to extract username from email
function extractUsernameFromEmail(email) {
    if (!email) return 'User';
    
    // Remove any whitespace and convert to lowercase
    const cleanEmail = email.trim().toLowerCase();
    
    // Extract the part before @
    const usernamePart = cleanEmail.split('@')[0];
    
    if (!usernamePart) return 'User';
    
    // Capitalize first letter
    const capitalized = usernamePart.charAt(0).toUpperCase() + usernamePart.slice(1);
    
    // Remove dots and underscores, replace with spaces
    const formatted = capitalized.replace(/[._-]/g, ' ');
    
    // If the name contains spaces, take only the first part
    const firstName = formatted.split(' ')[0];
    
    return firstName || 'User';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

const MAX_PREVIEW_LENGTH = 80;

function getPreviewText(text) {
    if (!text) return '';
    const firstSentenceMatch = text.match(/^[^.!?]*[.!?](?=\s|$)|^[^.!?]+/);
    let firstSentence = firstSentenceMatch ? firstSentenceMatch[0] : text;
    if (firstSentence.length <= MAX_PREVIEW_LENGTH) {
        return firstSentence;
    } else {
        return firstSentence.substring(0, MAX_PREVIEW_LENGTH).trim() + '…';
    }
}

// Function to fetch conversations from server
async function fetchConversations() {
    if (isFetchingConversations) return conversations;
    
    isFetchingConversations = true;
    try {
        const sessionToken = getCookie('session_token');
        const headers = sessionToken ? {
            'Authorization': `Bearer ${sessionToken}`
        } : {};
        
        const response = await fetch('/api/user/conversations', {
            headers: headers
        });
        
        if (response.ok) {
            const data = await response.json();
            
            // Transform server data to match our local format
            conversations = data.map(conv => ({
                id: conv.id,
                date: conv.timestamp, // Keep using timestamp as date
                user_message: conv.user_message || '',
                ai_message: conv.ai_message || '',
                starred: conv.starred || false, // Map the 'starred' field from the server response
                audio_path: conv.audio_path || '',
                server_id: conv.id
            }));
            
            conversationsLastUpdated = new Date();
            console.log(`Loaded ${conversations.length} conversations from server`);
            
            return conversations;
        } else if (response.status === 401) {
            console.log('User not authenticated, using local conversations only');
            return conversations;
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
    } catch (error) {
        console.error('Error fetching conversations:', error);
        return conversations;
    } finally {
        isFetchingConversations = false;
    }
}

// Function to refresh conversations
async function refreshConversations() {
    const spinner = document.createElement('div');
    spinner.innerHTML = `
        <div class="conversation-card" style="text-align: center; padding: 20px;">
            <div class="loading-spinner" style="margin: 0 auto 10px; width: 30px; height: 30px; border: 3px solid #f3f4f6; border-top: 3px solid #4f46e5; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            <div style="color: #6b7280;">Loading conversations...</div>
        </div>
    `;
    
    const mainContent = document.querySelector('.main-content');
    mainContent.innerHTML = '';
    mainContent.appendChild(spinner);
    
    await fetchConversations();
    renderConversations(currentFilter);
}

// Pulse animation control functions
function startPulseAnimation() {
    const pulseContainer = document.querySelector('.pulse-container');
    const dotsPulse = document.querySelector('.dots-pulse');
    
    if (pulseContainer) {
        pulseContainer.classList.add('pulsing');
        pulseContainer.classList.remove('connected', 'loading', 'disconnected');
    }
    
    if (dotsPulse) {
        dotsPulse.classList.add('pulsing');
        dotsPulse.classList.remove('connected', 'loading', 'disconnected');
    }
}

function stopPulseAnimation() {
    const pulseContainer = document.querySelector('.pulse-container');
    const dotsPulse = document.querySelector('.dots-pulse');
    
    if (pulseContainer) {
        pulseContainer.classList.remove('pulsing');
        // Restore the previous status-based class
        pulseContainer.classList.add(modelStatus);
    }
    
    if (dotsPulse) {
        dotsPulse.classList.remove('pulsing');
        // Restore the previous status-based class
        dotsPulse.classList.add(modelStatus);
    }
}

// Audio playback functions
async function initializeAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContext.state === 'suspended') {
        await audioContext.resume();
    }
    return audioContext;
}

function queueAudioChunk(audioData, sampleRate, genId, chunkNum) {
    console.log(`Queuing audio chunk ${chunkNum} for generation ${genId}`);
    
    if (genId !== currentAudioGenerationId) {
        // New audio generation, clear queue and reset
        audioPlaybackQueue = [];
        currentAudioGenerationId = genId;
        audioScheduledTime = 0;
        console.log('Starting new audio generation:', genId);
    }
    
    // Add chunk to queue
    audioPlaybackQueue.push({
        audioData: new Float32Array(audioData),
        sampleRate: sampleRate,
        genId: genId,
        chunkNum: chunkNum
    });
    
    // Update model status to show audio is streaming
    const modelStatusEl = document.getElementById('modelStatus');
    if (modelStatusEl) {
        modelStatusEl.textContent = `Streaming audio... (${chunkNum} chunks)`;
        modelStatusEl.style.color = '#3b82f6'; // Blue for streaming
    }
    
    // Start pulse animation when first chunk arrives
    if (audioPlaybackQueue.length === 1 && !isPlayingAudio) {
        startPulseAnimation();
    }
    
    // If not currently playing, start processing the queue
    if (!isPlayingAudio) {
        console.log('Starting audio playback from queue');
        processAudioPlaybackQueue();
    }
}

async function processAudioPlaybackQueue() {
    if (audioPlaybackQueue.length === 0 || isPlayingAudio) {
        return;
    }
    
    isPlayingAudio = true;
    
    try {
        await initializeAudioContext();
        
        // Process chunks in sequence
        while (audioPlaybackQueue.length > 0) {
            const chunk = audioPlaybackQueue.shift();
            await playAudioChunk(chunk.audioData, chunk.sampleRate, chunk.genId, chunk.chunkNum);
        }
        
    } catch (error) {
        console.error('Error processing audio queue:', error);
        isPlayingAudio = false;
        stopPulseAnimation();
        showNotification('Error playing audio: ' + error.message, 'error');
    }
}

async function playAudioChunk(audioData, sampleRate, genId, chunkNum) {
    return new Promise(async (resolve) => {
        if (genId !== currentAudioGenerationId) {
            console.log(`Skipping chunk ${chunkNum} - generation ID mismatch`);
            resolve();
            return;
        }
        
        try {
            console.log(`Playing audio chunk ${chunkNum}, samples: ${audioData.length}`);
            
            // Create audio buffer
            const audioBuffer = audioContext.createBuffer(1, audioData.length, sampleRate);
            audioBuffer.copyToChannel(audioData, 0);
            
            // Create source
            currentAudioSource = audioContext.createBufferSource();
            currentAudioSource.buffer = audioBuffer;
            currentAudioSource.connect(audioContext.destination);
            
            // Calculate when to play this chunk
            const now = audioContext.currentTime;
            const startTime = audioScheduledTime === 0 ? now : audioScheduledTime;
            
            // Calculate duration and update next scheduled time
            const duration = audioBuffer.duration;
            audioScheduledTime = startTime + duration;
            
            console.log(`Scheduling chunk ${chunkNum} at ${startTime.toFixed(2)}s, duration: ${duration.toFixed(2)}s`);
            
            // Set up event handlers
            currentAudioSource.onended = () => {
                console.log(`Audio chunk ${chunkNum} finished playing`);
                currentAudioSource = null;
                
                // If queue is empty after this chunk, mark as done
                if (audioPlaybackQueue.length === 0) {
                    isPlayingAudio = false;
                    console.log('Audio playback complete');
                    
                    // Stop pulse animation
                    stopPulseAnimation();
                    
                    // Update model status
                    const modelStatusEl = document.getElementById('modelStatus');
                    if (modelStatusEl) {
                        modelStatusEl.textContent = 'All models loaded';
                        modelStatusEl.style.color = '#10b981';
                    }
                    
                    showNotification('Audio playback complete', 'success');
                }
                
                resolve();
            };
            
            currentAudioSource.onerror = (error) => {
                console.error(`Audio chunk ${chunkNum} playback error:`, error);
                currentAudioSource = null;
                isPlayingAudio = false;
                stopPulseAnimation();
                showNotification('Audio playback failed', 'error');
                resolve();
            };
            
            // Start playback
            currentAudioSource.start(startTime);
            
            // Show notification for first chunk
            if (chunkNum === 1) {
                showNotification('Playing audio response...', 'info');
            }
            
        } catch (error) {
            console.error(`Error playing audio chunk ${chunkNum}:`, error);
            resolve();
        }
    });
}

function stopAudioPlayback() {
    if (currentAudioSource) {
        try {
            currentAudioSource.stop();
            console.log('Audio playback stopped');
        } catch (e) {
            console.log('Audio source already stopped');
        }
        currentAudioSource = null;
    }
    isPlayingAudio = false;
    audioPlaybackQueue = [];
    currentAudioGenerationId = null;
    audioScheduledTime = 0;
    
    // Stop pulse animation
    stopPulseAnimation();
}

function handleAudioChunk(data) {
    console.log('Received audio chunk:', data.chunk_num, 'for generation:', data.gen_id);
    
    try {
        // Queue the audio chunk for playback
        queueAudioChunk(data.audio, data.sample_rate || 24000, data.gen_id, data.chunk_num);
        
    } catch (error) {
        console.error('Error processing audio chunk:', error);
        showNotification('Error processing audio chunk', 'error');
    }
}

function handleAudioStatus(data) {
    console.log('Audio status:', data.status, 'Gen ID:', data.gen_id);
    
    const modelStatusEl = document.getElementById('modelStatus');
    
    switch(data.status) {
        case 'generating':
            showNotification('Generating audio response...', 'info');
            if (modelStatusEl) {
                modelStatusEl.textContent = 'Generating audio...';
                modelStatusEl.style.color = '#f59e0b';
            }
            break;
        case 'first_chunk':
            showNotification('Audio generation started', 'info');
            if (modelStatusEl) {
                modelStatusEl.textContent = 'Audio streaming...';
                modelStatusEl.style.color = '#3b82f6';
            }
            break;
        case 'complete':
            showNotification('Audio generation complete', 'success');
            if (modelStatusEl) {
                modelStatusEl.textContent = 'Audio complete';
                modelStatusEl.style.color = '#10b981';
            }
            
            // If we have chunks but haven't started playing, start now
            setTimeout(() => {
                if (audioPlaybackQueue.length > 0 && !isPlayingAudio) {
                    console.log('Starting delayed playback after completion');
                    processAudioPlaybackQueue();
                }
            }, 500);
            break;
        case 'interrupted':
            stopAudioPlayback();
            // stopPulseAnimation() will be called by stopAudioPlayback()
            showNotification('Audio generation interrupted', 'warning');
            if (modelStatusEl) {
                modelStatusEl.textContent = 'Audio interrupted';
                modelStatusEl.style.color = '#ef4444';
            }
            break;
        case 'preparing_generation':
            showNotification('Preparing audio generation...', 'info');
            break;
    }
}

// Updated renderConversations function to use extracted username
function renderConversations(filter = currentFilter) {
    const mainContent = document.querySelector('.main-content');
    mainContent.innerHTML = '';
    
    let filteredConvs = conversations;
    if (filter === 'starred') {
        filteredConvs = conversations.filter(c => c.starred);
    } else if (filter === 'recent') {
        const now = new Date();
        const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000);
        filteredConvs = conversations.filter(c => new Date(c.date) > yesterday);
    }
    
    if (filteredConvs.length === 0) {
        let message = '';
        let emoji = '';
        switch(filter) {
            case 'starred':
                message = 'No starred conversations yet';
                emoji = '⭐';
                break;
            case 'recent':
                message = 'No recent conversations';
                emoji = '🕒';
                break;
            default:
                message = 'No conversations yet';
                emoji = '💬';
        }
        
        const emptyState = document.createElement('div');
        emptyState.style.cssText = `
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 300px;
            text-align: center;
            color: #6b7280;
        `;
        emptyState.innerHTML = `
            <div style="font-size: 4rem; margin-bottom: 1rem;">${emoji}</div>
            <div style="font-size: 1.25rem; font-weight: 500; margin-bottom: 0.5rem;">${message}</div>
            <div style="font-size: 0.875rem;">Start a new conversation to see it here</div>
            <button id="refreshConversationsBtn" style="margin-top: 1rem; padding: 0.5rem 1rem; background-color: #4f46e5; color: white; border: none; border-radius: 0.375rem; cursor: pointer; transition: background-color 0.2s;">
                Refresh Conversations
            </button>
        `;
        mainContent.appendChild(emptyState);
        
        const refreshBtn = document.getElementById('refreshConversationsBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', refreshConversations);
        }
        return;
    }
    
    // Sort conversations by date (newest first)
    filteredConvs.sort((a, b) => new Date(b.date) - new Date(a.date));
    
    // Get current user email and extract username
    const currentUserEmail = document.getElementById('currentUserEmail')?.textContent || '';
    const userName = extractUsernameFromEmail(currentUserEmail);
    
    filteredConvs.forEach(conv => {
        const fullUser = escapeHtml(conv.user_message);
        const fullAi = escapeHtml(conv.ai_message);
        const previewUser = escapeHtml(getPreviewText(conv.user_message));
        const previewAi = escapeHtml(getPreviewText(conv.ai_message));
        
        const card = document.createElement('div');
        card.className = 'conversation-card';
        card.dataset.fullUser = fullUser;
        card.dataset.fullAi = fullAi;
        card.dataset.previewUser = previewUser;
        card.dataset.previewAi = previewAi;
        card.dataset.id = conv.id;
        
        const isStarred = conv.starred;
        
        card.innerHTML = `
            <div class="flex justify-between items-start mb-4">
                <div class="text-sm text-gray-500">${formatDate(conv.date)}</div>
                <div class="text-xs bg-gray-100 px-2 py-1 rounded">Conversation #${conv.id}</div>
            </div>
            <div class="conversation-sections">
                <div class="message-section user-section">
                    <div class="message-header">
                        <div class="icon-container user-icon">
                            <svg xmlns="http://www.w3.org/2000/svg" class="icon-svg" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd" />
                            </svg>
                        </div>
                        <div class="text-blue-700 font-semibold message-label">${userName}</div>
                    </div>
                    <div class="message-content">
                        <div class="message-text">${previewUser}</div>
                    </div>
                </div>
                <div class="message-section ai-section">
                    <div class="message-header">
                        <div class="icon-container ai-icon">
                            <svg xmlns="http://www.w3.org/2000/svg" class="icon-svg" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
                            </svg>
                        </div>
                        <div class="text-green-700 font-semibold message-label">Basqui-R1-14B</div>
                    </div>
                    <div class="message-content">
                        <div class="message-text">${previewAi}</div>
                    </div>
                </div>
            </div>
            <div class="action-icons">
                <div class="action-icon star-icon ${isStarred ? 'active' : ''}" data-id="${conv.id}">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                        <path fill-rule="evenodd" d="M10.788 3.21c.448-1.077 1.976-1.077 2.424 0l2.082 5.007 5.404.433c1.164.093 1.636 1.545.749 2.305l-4.117 3.527 1.257 5.273c.271 1.136-.964 2.033-1.96 1.425L12 18.354 7.373 21.18c-.996.608-2.231-.29-1.96-1.425l1.257-5.273-4.117-3.527c-.887-.76-.415-2.212.749-2.305l5.404-.433 2.082-5.006z" clip-rule="evenodd" />
                    </svg>
                </div>
                <div class="action-icon expand-icon" data-id="${conv.id}">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                        <path fill-rule="evenodd" d="M15 3.75a.75.75 0 01.75-.75h4.5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0V5.56l-3.97 3.97a.75.75 0 11-1.06-1.06l3.97-3.97h-2.69a.75.75 0 01-.75-.75zm-12 0A.75.75 0 013.75 3h4.5a.75.75 0 010 1.5H5.56l3.97 3.97a.75.75 0 01-1.06 1.06L4.5 5.56v2.69a.75.75 0 01-1.5 0v-4.5zm11.47 14.78a.75.75 0 111.06-1.06l3.97 3.97v-2.69a.75.75 0 011.5 0v4.5a.75.75 0 01-1.5 0v-4.5a.75.75 0 010-1.5h4.5a.75.75 0 010 1.5h-2.69l-3.97-3.97zm-4.94-1.06a.75.75 0 010 1.06L5.56 19.5h2.69a.75.75 0 010 1.5h-4.5a.75.75 0 01-.75-.75v-4.5a.75.75 0 011.5 0v2.69l3.97-3.97a.75.75 0 011.06 0z" clip-rule="evenodd" />
                    </svg>
                </div>
            </div>
        `;
        
        mainContent.appendChild(card);
    });
    
    // Add event listeners for star icons
    document.querySelectorAll('.star-icon').forEach(icon => {
        icon.addEventListener('click', function() {
            const id = parseInt(this.getAttribute('data-id'), 10);
            handleStarToggle(id);
            // const conv = conversations.find(c => c.id === id);
            // if (conv) {
            //     conv.starred = !conv.starred;
            //     this.classList.toggle('active');
            // }
        });
    });
    
    // Add event listeners for expand icons
    document.querySelectorAll('.expand-icon').forEach(icon => {
        icon.addEventListener('click', function() {
            const id = this.getAttribute('data-id');
            const card = this.closest('.conversation-card');
            const isExpanded = card.classList.contains('expanded');
            
            card.classList.toggle('expanded');
            const userContent = card.querySelector('.user-section .message-text');
            const aiContent = card.querySelector('.ai-section .message-text');
            
            if (!isExpanded) {
                userContent.innerHTML = card.dataset.fullUser;
                aiContent.innerHTML = card.dataset.fullAi;
                this.querySelector('svg').innerHTML = `<path fill-rule="evenodd" d="M4.5 12a.75.75 0 01.75-.75h13.5a.75.75 0 010 1.5H5.25a.75.75 0 01-.75-.75z" clip-rule="evenodd" />`;
            } else {
                userContent.innerHTML = card.dataset.previewUser;
                aiContent.innerHTML = card.dataset.previewAi;
                this.querySelector('svg').innerHTML = `<path fill-rule="evenodd" d="M15 3.75a.75.75 0 01.75-.75h4.5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0V5.56l-3.97 3.97a.75.75 0 11-1.06-1.06l3.97-3.97h-2.69a.75.75 0 01-.75-.75zm-12 0A.75.75 0 013.75 3h4.5a.75.75 0 010 1.5H5.56l3.97 3.97a.75.75 0 01-1.06 1.06L4.5 5.56v2.69a.75.75 0 01-1.5 0v-4.5zm11.47 14.78a.75.75 0 111.06-1.06l3.97 3.97v-2.69a.75.75 0 011.5 0v4.5a.75.75 0 01-1.5 0v-4.5a.75.75 0 010-1.5h4.5a.75.75 0 010 1.5h-2.69l-3.97-3.97zm-4.94-1.06a.75.75 0 010 1.06L5.56 19.5h2.69a.75.75 0 010 1.5h-4.5a.75.75 0 01-.75-.75v-4.5a.75.75 0 011.5 0v2.69l3.97-3.97a.75.75 0 011.06 0z" clip-rule="evenodd" />`;
            }
        });
    });
}

// Model Status Management
let modelStatus = 'loading'; // loading, connected, disconnected
let ws = null;
let statusCheckInterval = null;

function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
}

function updateConnectionStatus() {
    const statusEl = document.getElementById('connectionStatus');
    const modelStatusEl = document.getElementById('modelStatus');
    const userEmailEl = document.getElementById('currentUserEmail');
    
    if (!statusEl || !modelStatusEl || !userEmailEl) return;
    
    switch(modelStatus) {
        case 'connected':
            statusEl.textContent = 'Connected';
            statusEl.style.color = STATUS_COLORS.connected.status;
            userEmailEl.style.color = STATUS_COLORS.connected.user;
            modelStatusEl.textContent = 'Ready';
            modelStatusEl.style.color = STATUS_COLORS.connected.status;
            break;
        case 'loading':
            statusEl.textContent = 'Connecting';
            statusEl.style.color = STATUS_COLORS.loading.status;
            userEmailEl.style.color = STATUS_COLORS.loading.user;
            modelStatusEl.textContent = 'Loading...';
            modelStatusEl.style.color = STATUS_COLORS.loading.status;
            break;
        case 'disconnected':
            statusEl.textContent = 'Disconnected';
            statusEl.style.color = STATUS_COLORS.disconnected.status;
            userEmailEl.style.color = STATUS_COLORS.disconnected.user;
            modelStatusEl.textContent = 'Offline';
            modelStatusEl.style.color = STATUS_COLORS.disconnected.status;
            break;
        default:
            statusEl.textContent = 'Connecting';
            statusEl.style.color = STATUS_COLORS.loading.status;
            userEmailEl.style.color = STATUS_COLORS.loading.user;
            modelStatusEl.textContent = 'Checking...';
            modelStatusEl.style.color = STATUS_COLORS.loading.status;
    }
}

async function checkModelStatus() {
    try {
        const response = await fetch('/api/status');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.models_loaded) {
            modelStatus = 'connected';
        } else if (data.whisper_loaded || data.llm_loaded || data.rag_loaded) {
            modelStatus = 'connected';
        } else {
            modelStatus = 'disconnected';
        }
        
        updateConnectionStatus();
        updatePulseAnimation();
        
        return data;
    } catch (error) {
        console.error('Error checking model status:', error);
        modelStatus = 'disconnected';
        updateConnectionStatus();
        updatePulseAnimation();
        return null;
    }
}

function updatePulseAnimation() {
    const pulseContainer = document.querySelector('.pulse-container');
    const dotsPulse = document.querySelector('.dots-pulse');
    
    // Don't update if currently pulsing (audio is playing)
    if (pulseContainer && pulseContainer.classList.contains('pulsing')) return;
    if (dotsPulse && dotsPulse.classList.contains('pulsing')) return;
    
    if (pulseContainer) {
        pulseContainer.classList.remove('connected', 'loading', 'disconnected');
        pulseContainer.classList.add(modelStatus);
    }
    
    if (dotsPulse) {
        dotsPulse.classList.remove('connected', 'loading', 'disconnected');
        dotsPulse.classList.add(modelStatus);
    }
}

// Notification system
function showNotification(message, type = 'info') {
    const colors = {
        info: '#3b82f6',
        success: '#10b981',
        warning: '#f59e0b',
        error: '#ef4444'
    };
    
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${colors[type] || colors.info};
        color: white;
        padding: 12px 18px;
        border-radius: 6px;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        animation: slideIn 0.3s ease;
    `;
    
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

function setupWebSocket() {
    const sessionToken = getCookie('session_token');
    const wsUrl = sessionToken 
        ? `ws://${window.location.host}/ws?session_token=${encodeURIComponent(sessionToken)}`
        : `ws://${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        modelStatus = 'loading';
        updateConnectionStatus();
        
        checkModelStatus();
        if (statusCheckInterval) clearInterval(statusCheckInterval);
        statusCheckInterval = setInterval(checkModelStatus, 10000);
        
        ws.send(JSON.stringify({ type: 'test', message: 'Connection test' }));
        ws.send(JSON.stringify({ type: 'request_conversation_history' }));
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        modelStatus = 'disconnected';
        updateConnectionStatus();
        updatePulseAnimation();
        
        setTimeout(() => {
            console.log('Attempting to reconnect...');
            setupWebSocket();
        }, 5000);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        modelStatus = 'disconnected';
        updateConnectionStatus();
        updatePulseAnimation();
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            console.log('WebSocket message received:', data.type);
            
            switch(data.type) {
                case 'test_response':
                    console.log('Server test response:', data.message);
                    if (data.user_email && data.user_email !== 'anonymous') {
                        const emailEl = document.getElementById('currentUserEmail');
                        if (emailEl) {
                            emailEl.textContent = data.user_email;
                            updateConnectionStatus();
                        }
                    }
                    break;
                    
                case 'connection_established':
                    console.log('Connection established with session:', data.session_id);
                    if (data.user_email && data.user_email !== 'anonymous') {
                        const emailEl = document.getElementById('currentUserEmail');
                        if (emailEl) {
                            emailEl.textContent = data.user_email;
                            updateConnectionStatus();
                        }
                    }
                    break;
                    
                case 'audio_chunk':
                    handleAudioChunk(data);
                    break;
                    
                case 'audio_status':
                    handleAudioStatus(data);
                    break;
                    
                case 'response':
                    handleAIResponse(data);
                    break;
                    
                case 'conversation_history':
                    updateConversationsFromServer(data.conversations);
                    break;
                    
                case 'error':
                    console.error('Server error:', data.message);
                    modelStatus = 'disconnected';
                    updateConnectionStatus();
                    updatePulseAnimation();
                    break;
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };
}

// Function to update conversations from server data
function updateConversationsFromServer(serverConversations) {
    if (!Array.isArray(serverConversations)) return;
    
    const newConversations = serverConversations.map(conv => ({
        id: conv.id,
        date: conv.timestamp,
        user_message: conv.user_message || '',
        ai_message: conv.ai_message || '',
        starred: conversations.find(c => c.server_id === conv.id)?.starred || false,
        audio_path: conv.audio_path || '',
        server_id: conv.id
    }));
    
    conversations = newConversations;
    conversationsLastUpdated = new Date();
    
    console.log(`Updated ${conversations.length} conversations from server`);
    renderConversations(currentFilter);
}

function handleAIResponse(data) {
    const newConversation = {
        id: conversations.length > 0 ? Math.max(...conversations.map(c => c.id)) + 1 : 1,
        date: new Date().toISOString().replace('T', ' ').substring(0, 19),
        user_message: window.lastUserMessage || 'User message',
        ai_message: data.text,
        starred: false,
        audio_path: data.audio_path || '',
        server_id: null
    };
    
    conversations.unshift(newConversation);
    window.lastUserMessage = null;
    renderConversations(currentFilter);
    
    setTimeout(() => {
        fetchConversations().then(() => {
            renderConversations(currentFilter);
        });
    }, 2000);
}

// Text input handling
function setupTextInput() {
    const textInput = document.getElementById('textInput');
    const sendBtn = document.getElementById('sendTextBtn');
    const charCount = document.getElementById('charCount');
    
    if (!textInput || !sendBtn || !charCount) return;
    
    textInput.addEventListener('input', () => {
        const length = textInput.value.length;
        charCount.textContent = `${length}/500`;
        
        sendBtn.disabled = length === 0 || length > 500 || modelStatus !== 'connected';
        
        if (length > 450) {
            charCount.style.color = '#ef4444';
        } else if (length > 400) {
            charCount.style.color = '#f59e0b';
        } else {
            charCount.style.color = '#6b7280';
        }
    });
    
    textInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled && ws && ws.readyState === WebSocket.OPEN) {
                sendTextMessage();
            }
        } else if (e.key === 'Escape') {
            sendInterrupt();
        }
    });
    
    sendBtn.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            sendTextMessage();
        }
    });
    
    function sendTextMessage() {
        const message = textInput.value.trim();
        if (message && ws && ws.readyState === WebSocket.OPEN && modelStatus === 'connected') {
            window.lastUserMessage = message;
            
            ws.send(JSON.stringify({
                type: 'text_message',
                text: message,
                timestamp: new Date().toISOString()
            }));
            
            textInput.value = '';
            charCount.textContent = '0/500';
            sendBtn.disabled = true;
            charCount.style.color = '#6b7280';
            
            showNotification('Sending message...', 'info');
        }
    }
    
    function sendInterrupt() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'interrupt'
            }));
            stopAudioPlayback();
            showNotification('Interrupt sent', 'warning');
        }
    }
}

// Auto-refresh conversations
function startAutoRefresh() {
    setInterval(async () => {
        if (document.visibilityState === 'visible') {
            await fetchConversations();
            renderConversations(currentFilter);
        }
    }, 30000);
}





// Add this new function to handle toggling the star via the API
async function handleStarToggle(id) {
    console.log(`Toggling star for conversation ID: ${id}`);
    try {
        const sessionToken = getCookie('session_token');
        const headers = sessionToken ? {
            'Authorization': `Bearer ${sessionToken}`,
            'Content-Type': 'application/json'
        } : {
            'Content-Type': 'application/json'
        };

        // Determine the new starred state based on the current UI state
        const conv = conversations.find(c => c.id === id);
        if (!conv) {
            console.error(`Conversation with ID ${id} not found in local array.`);
            return;
        }
        const newStarredValue = !conv.starred;

        const response = await fetch(`/api/user/conversations/${id}/star`, {
            method: 'PUT', // Use PUT to update the conversation
            headers: headers,
            body: JSON.stringify({ starred: newStarredValue })
        });

        if (response.ok) {
            const result = await response.json();
            console.log(`Conversation ${id} star status updated on server: ${result.starred}`);

            // Update the local conversation object
            conv.starred = result.starred;

            // Update the UI (star icon)
            const starIcon = document.querySelector(`.star-icon[data-id="${id}"]`);
            if (starIcon) {
                if (conv.starred) {
                    starIcon.classList.add('active');
                } else {
                    starIcon.classList.remove('active');
                }
            }

            // If currently viewing the 'starred' filter, re-render to reflect changes
            if (currentFilter === 'starred') {
                renderConversations(currentFilter);
            }
        } else if (response.status === 401) {
            console.error('Authentication required to star conversation.');
            showNotification('Please log in to star conversations.', 'error');
        } else {
            console.error(`Failed to update star status on server: ${response.status}`);
            showNotification('Failed to update star status.', 'error');
            // Revert UI change if server failed
            const starIcon = document.querySelector(`.star-icon[data-id="${id}"]`);
            if (starIcon) {
                 if (conv.starred) { // If it was previously starred locally
                    starIcon.classList.add('active'); // Keep it active
                 } else {
                    starIcon.classList.remove('active'); // Keep it inactive
                 }
            }
        }
    } catch (error) {
        console.error('Error updating star status:', error);
        showNotification('Error updating star status.', 'error');
        // Revert UI change if request failed
        const starIcon = document.querySelector(`.star-icon[data-id="${id}"]`);
        if (starIcon) {
             if (conv.starred) { // If it was previously starred locally
                starIcon.classList.add('active'); // Keep it active
             } else {
                starIcon.classList.remove('active'); // Keep it inactive
             }
        }
    }
}





// Add this function to handle copying the link with a fallback
async function copyShareLink(event) {
    const urlToCopy = "https://voice.porturs.com/";

    try {
        // Try the modern Clipboard API first
        await navigator.clipboard.writeText(urlToCopy);
        console.log("Share link copied to clipboard (via Clipboard API):", urlToCopy);
        showNotification('Share link copied to clipboard!', 'success');
    } catch (err) {
        console.error('Clipboard API failed: ', err);
        console.log('Attempting fallback copy method...');

        // Fallback for older browsers or when Clipboard API is denied
        try {
            // Create a temporary textarea element
            const textArea = document.createElement('textarea');
            textArea.value = urlToCopy;
            // Make it invisible and not affect layout
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '-999999px';
            textArea.style.opacity = '0';
            textArea.style.pointerEvents = 'none';

            document.body.appendChild(textArea);

            // Select the text inside the textarea
            textArea.focus();
            textArea.select();
            // Use the older execCommand
            const successful = document.execCommand('copy');

            // Clean up by removing the temporary textarea
            document.body.removeChild(textArea);

            if (successful) {
                console.log("Share link copied to clipboard (via execCommand):", urlToCopy);
                showNotification('Share link copied to clipboard!', 'success');
            } else {
                console.error('execCommand copy failed.');
                showNotification('Failed to copy link. Please try again.', 'error');
            }
        } catch (fallbackErr) {
            console.error('Error during fallback copy: ', fallbackErr);
            showNotification('Failed to copy link. Please try again.', 'error');
        }
    }
}

// Add this code inside the DOMContentLoaded event listener
document.addEventListener('DOMContentLoaded', async () => {
    // ... (existing code up to the end of DOMContentLoaded) ...

    // Select all share icons using their common attributes (like the title)
    const shareIcons = document.querySelectorAll('div[title="Share"]');

    // Add the click event listener to each share icon found
    shareIcons.forEach(icon => {
        icon.addEventListener('click', copyShareLink);
    });

    // ... (rest of your existing DOMContentLoaded code) ...
    // Setup WebSocket, text input, etc.
    setupWebSocket();
    setupTextInput();
    checkModelStatus();
    startAutoRefresh();

    window.addEventListener('beforeunload', () => {
        if (ws) {
            ws.close();
        }
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
        }
        stopAudioPlayback();
    });
    // ... (end of DOMContentLoaded code) ...
});




// Add this function to handle navigating to the feedback page
function navigateToFeedback(event) {
    // Navigate to the feedback page
    window.open('/feedback', '_blank'); // Adjust the path if your feedback page is served elsewhere
}

// Add this code inside the DOMContentLoaded event listener, after the Share icon listener
document.addEventListener('DOMContentLoaded', async () => {
    // ... (existing code) ...

    // Select all share icons using their common attributes (like the title)
    const shareIcons = document.querySelectorAll('div[title="Share"]');

    // Add the click event listener to each share icon found
    shareIcons.forEach(icon => {
        icon.addEventListener('click', copyShareLink);
    });

    // Select all feedback icons using their common attributes (like the title)
    const feedbackIcons = document.querySelectorAll('div[title="Feedback"]');

    // Add the click event listener to each feedback icon found
    feedbackIcons.forEach(icon => {
        icon.addEventListener('click', navigateToFeedback);
    });

    // ... (rest of your existing DOMContentLoaded code) ...
    // Setup WebSocket, text input, etc.
    setupWebSocket();
    setupTextInput();
    checkModelStatus();
    startAutoRefresh();

    window.addEventListener('beforeunload', () => {
        if (ws) {
            ws.close();
        }
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
        }
        stopAudioPlayback();
    });
    // ... (end of DOMContentLoaded code) ...
});



// Initialize everything
document.addEventListener('DOMContentLoaded', async () => {
    // Add notification styles
    const notificationStyles = document.createElement('style');
    notificationStyles.textContent = `
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        @keyframes slideOut {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(notificationStyles);
    
    // Initialize filter buttons
    const filterButtons = document.querySelectorAll('.filter-btn');
    filterButtons.forEach(btn => {
        const filter = btn.getAttribute('data-filter');
        if (filter === currentFilter) {
            btn.classList.add('active');
        }
    });
    
    filterButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            filterButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentFilter = btn.getAttribute('data-filter');
            localStorage.setItem('conversationFilter', currentFilter);
            renderConversations(currentFilter);
        });
    });
    
    // Initial render with loading state
    const mainContent = document.querySelector('.main-content');
    mainContent.innerHTML = `
        <div class="conversation-card" style="text-align: center; padding: 40px;">
            <div class="loading-spinner" style="margin: 0 auto 20px; width: 40px; height: 40px; border: 4px solid #f3f4f6; border-top: 4px solid #4f46e5; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            <div style="color: #6b7280; font-size: 1rem;">Loading conversations...</div>
        </div>
    `;
    
    // Set initial status
    modelStatus = 'loading';
    updateConnectionStatus();
    updatePulseAnimation();
    
    // Update username placeholder
    const userEmailEl = document.getElementById('currentUserEmail');
    if (userEmailEl && (!userEmailEl.textContent || userEmailEl.textContent.trim() === '')) {
        userEmailEl.textContent = 'loading';
        userEmailEl.style.color = STATUS_COLORS.loading.user;
    }
    
    // Load conversations from server
    await fetchConversations();
    
    // Initial render with loaded conversations
    renderConversations(currentFilter);
    
    // Setup WebSocket and status checking
    setupWebSocket();
    
    // Setup text input
    setupTextInput();
    
    // Check model status immediately
    checkModelStatus();
    
    // Start auto-refresh
    startAutoRefresh();
    
    // Clean up on page unload
    window.addEventListener('beforeunload', () => {
        if (ws) {
            ws.close();
        }
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
        }
        stopAudioPlayback();
    });
});







































// Function to setup menu
function setupMenu() {
    const menuIcons = document.querySelectorAll('.menu-icon');
    const menuOverlay = document.getElementById('menuOverlay');
    const menuContainer = document.getElementById('menuContainer');
    const closeMenuBtn = document.getElementById('closeMenu');
    const chatToggle = document.getElementById('chatToggle');
    const callToggle = document.getElementById('callToggle');
    const feedbackBtn = document.querySelector('[data-action="feedback"]');
    const shareBtn = document.querySelector('[data-action="share"]');
    const chatInputContainer = document.getElementById('chatInputContainer');
    const voiceInputContainer = document.getElementById('voiceInputContainer');
    
    if (feedbackBtn) {
        feedbackBtn.addEventListener('click', function() {
            window.open('/feedback', '_blank');
            closeMenu();
        });
    }
    
    if (shareBtn) {
        shareBtn.addEventListener('click', function() {
            const urlToCopy = "https://voice.porturs.com/";
            navigator.clipboard.writeText(urlToCopy)
                .then(() => {
                    console.log("Share link copied to clipboard:", urlToCopy);
                    showNotification('Share link copied to clipboard!', 'success');
                })
                .catch(err => {
                    console.error('Failed to copy share link: ', err);
                    showNotification('Failed to copy link. Please try again.', 'error');
                });
            closeMenu();
        });
    }
    
    
    // Function to update UI based on toggle states
    function updateInputUI() {
        if (voiceInputContainer) {
            if (callToggle.checked) {
                // Call is ON - hide voice input container
                voiceInputContainer.style.display = 'flex';
                console.log('Voice input container hidden (Call mode)');
            } else {
                // Call is OFF - show voice input container
                voiceInputContainer.style.display = 'none'; // or whatever your original display value is
                console.log('Voice input container shown (Chat mode)');
            }
        }


        if (chatInputContainer) {
            if (chatToggle.checked) {
                // Call is ON - hide voice input container
                chatInputContainer.style.display = 'flex';
                console.log('Voice input container hidden (Call mode)');
            } else {
                // Call is OFF - show voice input container
                chatInputContainer.style.display = 'none';
                console.log('Voice input container shown (Chat mode)');
            }
        }
    }
    
    // Function to open menu
    function openMenu() {
        menuOverlay.classList.add('active');
        menuContainer.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
    
    // Function to close menu
    function closeMenu() {
        menuOverlay.classList.remove('active');
        menuContainer.classList.remove('active');
        document.body.style.overflow = '';
    }
    
    // Add click event to all menu icons
    menuIcons.forEach(icon => {
        const newIcon = icon.cloneNode(true);
        icon.parentNode.replaceChild(newIcon, icon);
        
        newIcon.addEventListener('click', openMenu);
        newIcon.setAttribute('role', 'button');
        newIcon.setAttribute('aria-label', 'Open menu');
        newIcon.setAttribute('aria-expanded', 'false');
        newIcon.setAttribute('tabindex', '0');
        
        newIcon.addEventListener('click', function() {
            const isExpanded = menuContainer.classList.contains('active');
            this.setAttribute('aria-expanded', isExpanded);
        });
    });
    
    // Close menu when overlay is clicked
    menuOverlay.addEventListener('click', closeMenu);
    
    // Close menu when close button is clicked
    closeMenuBtn.addEventListener('click', closeMenu);
    
    // Close menu with Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && menuContainer.classList.contains('active')) {
            closeMenu();
        }
    });
    
    // Load saved toggle states
    const savedChatState = localStorage.getItem('chatEnabled');
    const savedCallState = localStorage.getItem('callEnabled');

    // Set default state
    if (savedChatState === null) {
        chatToggle.checked = true;
        callToggle.checked = false;
    } else {
        chatToggle.checked = savedChatState === 'true';
        callToggle.checked = savedCallState === 'true';
    }

    // Apply initial UI state
    updateInputUI();

    // Apply mutual exclusivity on initial load
    if (chatToggle.checked && callToggle.checked) {
        callToggle.checked = false;
        localStorage.setItem('callEnabled', 'false');
        updateInputUI();
    }

    // Chat toggle event listener
    chatToggle.addEventListener('change', function() {
        if (this.checked) {
            callToggle.checked = false;
            localStorage.setItem('callEnabled', 'false');
            showNotification('Call feature disabled', 'warning');
        } else {
            callToggle.checked = true;
            localStorage.setItem('callEnabled', 'true');
            showNotification('Call feature enabled', 'success');
        }
        localStorage.setItem('chatEnabled', this.checked);
        
        // Update UI after state change
        updateInputUI();
        
        console.log('Chat enabled:', this.checked);
        if (this.checked) {
            showNotification('Chat feature enabled', 'success');
        }

        closeMenu();
    });
    
    // Call toggle event listener
    callToggle.addEventListener('change', function() {
        if (this.checked) {
            chatToggle.checked = false;
            localStorage.setItem('chatEnabled', 'false');
            showNotification('Chat feature disabled', 'warning');
        } else {
            chatToggle.checked = true;
            localStorage.setItem('chatEnabled', 'true');
            showNotification('Chat feature enabled', 'success');
        }
        localStorage.setItem('callEnabled', this.checked);
        
        // Update UI after state change
        updateInputUI();
        
        console.log('Call enabled:', this.checked);
        if (this.checked) {
            showNotification('Call feature enabled', 'success');
        }

        closeMenu();
    });
    
    // Initialize toggle states in UI
    updateToggleStates();
}

// Function to update toggle states based on current settings
function updateToggleStates() {
    const chatToggle = document.getElementById('chatToggle');
    const callToggle = document.getElementById('callToggle');
    
    // You can add logic here to sync with your app's state
    // For example, if chat is currently active in your app
    if (typeof window.isChatEnabled !== 'undefined') {
        chatToggle.checked = window.isChatEnabled;
    }
    
    if (typeof window.isCallEnabled !== 'undefined') {
        callToggle.checked = window.isCallEnabled;
    }
}

// Update your DOMContentLoaded event to include menu setup
document.addEventListener('DOMContentLoaded', async () => {
    // ... your existing DOMContentLoaded code ...
    
    // Add menu setup
    setupMenu();
    
    // ... rest of your existing code ...
});




































////HUNDLE VOICE
// State variables
let isCallActive = false;
let isMuted = false;
let hasMicrophonePermission = false;


// Call Dial Button Click Handler 

document.getElementById('callDialBtn').addEventListener('click', async function() {
    
    console.log("Hello Benjamin?");
    // If button is inactive, do nothing
    if (this.classList.contains('inactive')) return;
    
        // Start call
        try {
            
            // Activate all buttons for call controls
            startRecording();
            
            // Change call button to active state
            this.innerHTML = `
                <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M6.62 10.79c1.44 2.83 3.76 5.14 6.59 6.59l2.2-2.2c.27-.27.67-.36 1.02-.24 1.12.37 2.33.57 3.57.57.55 0 1 .45 1 1v3.49c0 .55-.45 1-1 1-9.39 0-17-7.61-17-17 0-.55.45-1 1-1h3.5c.55 0 1 .45 1 1 0 1.25.2 2.45.57 3.57.11.35.03.74-.25 1.02l-2.2 2.2z"/>
                </svg>
            `;
            
            
        } catch (error) {
            console.error('Failed to start call:', error);
            alert('Could not access microphone. Please check your permissions.');
            activateCallButtonOnly(); // Reset to initial state
        }
});


document.getElementById('muteBtn').addEventListener('click', function() {
    // Only work if not inactive
    if (this.classList.contains('inactive')) return;

});


// Decline Button Click Handler
document.getElementById('declineBtn').addEventListener('click', function() {
    // Only work if not inactive
    if (this.classList.contains('inactive')) return;
    
    stopRecording()
    console.log('Call declined');

    // ADD THIS: Mark as inactive
    this.classList.add('inactive');
    
    // Optional: Disable further clicks
    this.disabled = true;
});

// Function to activate all buttons during an active call
function activateAllButtonsForCall() {
    const buttons = document.querySelectorAll('.call-control-btn');
    buttons.forEach(button => {
        button.classList.remove('inactive');
    });
}



// Optional: Add visual indicator for permission state
const style = document.createElement('style');
style.textContent = `
    .call-dial-btn.active-call {
        background-color: #2196F3 !important;
    }
    
    .call-dial-btn.active-call:hover {
        background-color: #1976D2 !important;
    }
    
    /* Visual indicator for permission-granted call button */
    .call-dial-btn:not(.inactive) {
        position: relative;
    }
    
    .call-dial-btn:not(.inactive)::after {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 50%;
        transform: translateX(-50%);
        width: 6px;
        height: 6px;
        background-color: #4CAF50;
        border-radius: 50%;
    }
    
    /* Visual indicator for permission-request mic button */
    .mute-btn:not(.inactive):not(.muted) {
        position: relative;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(240, 240, 240, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(240, 240, 240, 0); }
        100% { box-shadow: 0 0 0 0 rgba(240, 240, 240, 0); }
    }
`;
document.head.appendChild(style);


async function startRecording() {

  
  if (isRecording) return;
  try {
    const constraints = {
      audio: selectedMicId ? {deviceId:{exact:selectedMicId}} : true
    };
    micStream = await navigator.mediaDevices.getUserMedia(constraints);

    if (!audioContext) audioContext = new (AudioContext||webkitAudioContext)();
    activateAllButtonsForCall();
    const src = audioContext.createMediaStreamSource(micStream);
    const proc = audioContext.createScriptProcessor(4096,1,1);
    src.connect(proc); proc.connect(audioContext.destination);

     
    proc.onaudioprocess = e => {
      const samples = Array.from(e.inputBuffer.getChannelData(0));
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify({
            type:'audio',
            audio:samples,
            sample_rate:audioContext.sampleRate,
            session_id:SESSION_ID
          }));
        } catch (error) {
          console.error("Error sending audio data:", error);
          stopRecording();
        }
      }
    };

    window._micProcessor = proc;        
    isRecording = true;
    document.getElementById('muteBtn').style.display = 'block';
  } catch (err) {
    console.error("Microphone access error:", err);
    showNotification('Microphone access denied','error');
  }
}

function stopRecording() {
  if (!isRecording) return;
  try {
    if (window._micProcessor) {
      window._micProcessor.disconnect();
      window._micProcessor = null;
    }
    if (micStream) {
      micStream.getTracks().forEach(t => t.stop());
      micStream = null;
    }
  } catch(e) {
    console.warn("Error stopping recording:", e);
  }
  isRecording = false;
  
  const micStatus = document.getElementById('micStatus');
  if (micStatus) {
    micStatus.textContent = 'Click to speak';
  }
    document.getElementById('muteBtn').style.display = 'none';
}