/**
 * Interactive LiveAvatar - Client Application
 */

class AvatarClient {
    constructor() {
        this.ws = null;
        this.sessionId = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        
        this.initializeElements();
        this.setupEventListeners();
        this.connect();
    }
    
    initializeElements() {
        // Video elements
        this.avatarVideo = document.getElementById('avatarVideo');
        this.videoPlaceholder = document.getElementById('videoPlaceholder');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.loadingText = document.getElementById('loadingText');
        
        // Chat elements
        this.chatContainer = document.getElementById('chatContainer');
        this.textInput = document.getElementById('textInput');
        this.sendButton = document.getElementById('sendButton');
        
        // Microphone elements
        this.micButton = document.getElementById('micButton');
        
        // Status elements
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.connectionStatus = document.getElementById('connectionStatus');
        this.sessionIdElem = document.getElementById('sessionId');
        
        // Settings
        this.settingsToggle = document.getElementById('settingsToggle');
        this.settingsContent = document.getElementById('settingsContent');
    }
    
    setupEventListeners() {
        // Microphone button (hold to speak)
        this.micButton.addEventListener('mousedown', () => this.startRecording());
        this.micButton.addEventListener('mouseup', () => this.stopRecording());
        this.micButton.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startRecording();
        });
        this.micButton.addEventListener('touchend', (e) => {
            e.preventDefault();
            this.stopRecording();
        });
        
        // Text input
        this.sendButton.addEventListener('click', () => this.sendTextMessage());
        this.textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendTextMessage();
            }
        });
        
        // Settings toggle
        this.settingsToggle.addEventListener('click', () => {
            this.settingsContent.classList.toggle('open');
        });
    }
    
    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        console.log('Connecting to:', wsUrl);
        this.updateStatus('connecting', 'Connecting...');
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateStatus('connected', 'Connected');
            this.connectionStatus.textContent = 'Connected';
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('error', 'Connection Error');
            this.connectionStatus.textContent = 'Error';
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket closed');
            this.updateStatus('disconnected', 'Disconnected');
            this.connectionStatus.textContent = 'Disconnected';
            
            // Try to reconnect after 3 seconds
            setTimeout(() => this.connect(), 3000);
        };
    }
    
    handleMessage(data) {
        console.log('Received message:', data);
        
        switch (data.type) {
            case 'connection':
                this.sessionId = data.session_id;
                this.sessionIdElem.textContent = data.session_id;
                this.addSystemMessage(data.message);
                break;
            
            case 'status':
                this.updateLoadingStatus(data.status, data.message);
                break;
            
            case 'transcription':
                this.addUserMessage(data.text);
                break;
            
            case 'response':
                this.addAssistantMessage(data.text);
                break;
            
            case 'video_ready':
                this.playAvatarVideo(data.video_url);
                this.hideLoading();
                break;
            
            case 'error':
                this.addSystemMessage(`Error: ${data.message}`, 'error');
                this.hideLoading();
                break;
        }
    }
    
    async startRecording() {
        if (this.isRecording) return;
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                this.sendAudioData(audioBlob);
                
                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            this.micButton.classList.add('recording');
            this.micButton.querySelector('.mic-text').textContent = 'Recording...';
            
            console.log('Recording started');
        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Could not access microphone. Please check permissions.');
        }
    }
    
    stopRecording() {
        if (!this.isRecording || !this.mediaRecorder) return;
        
        this.mediaRecorder.stop();
        this.isRecording = false;
        this.micButton.classList.remove('recording');
        this.micButton.querySelector('.mic-text').textContent = 'Hold to Speak';
        
        console.log('Recording stopped');
    }
    
    sendAudioData(audioBlob) {
        if (this.ws.readyState === WebSocket.OPEN) {
            console.log('Sending audio data:', audioBlob.size, 'bytes');
            this.ws.send(audioBlob);
            this.showLoading('Processing audio...');
        } else {
            console.error('WebSocket not connected');
            this.addSystemMessage('Not connected to server', 'error');
        }
    }
    
    sendTextMessage() {
        const text = this.textInput.value.trim();
        if (!text) return;
        
        if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'text_input',
                text: text
            }));
            
            this.addUserMessage(text);
            this.textInput.value = '';
            this.showLoading('Processing...');
        } else {
            this.addSystemMessage('Not connected to server', 'error');
        }
    }
    
    playAvatarVideo(videoUrl) {
        // Add timestamp to prevent caching issues
        const url = `${videoUrl}?t=${Date.now()}`;
        this.avatarVideo.src = url;
        this.avatarVideo.load();
        this.avatarVideo.play();
        
        // Hide placeholder, show video
        this.videoPlaceholder.style.display = 'none';
        this.avatarVideo.style.display = 'block';
        
        console.log('Playing video:', url);
    }
    
    addUserMessage(text) {
        this.addMessage('user', text);
    }
    
    addAssistantMessage(text) {
        this.addMessage('assistant', text);
    }
    
    addSystemMessage(text, className = '') {
        this.addMessage('system', text, className);
    }
    
    addMessage(role, text, extraClass = '') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role} ${extraClass}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const textP = document.createElement('p');
        textP.textContent = text;
        
        contentDiv.appendChild(textP);
        messageDiv.appendChild(contentDiv);
        this.chatContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }
    
    showLoading(message) {
        this.loadingText.textContent = message;
        this.loadingOverlay.style.display = 'flex';
    }
    
    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }
    
    updateLoadingStatus(status, message) {
        this.loadingText.textContent = message;
        this.loadingOverlay.style.display = 'flex';
    }
    
    updateStatus(status, text) {
        this.statusDot.className = `status-dot ${status}`;
        this.statusText.textContent = text;
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new AvatarClient();
});
