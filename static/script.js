

let videoStream = null;
let webcamCanvas = null;
let webcamInterval = null;
let isDatabaseConnected = false;
let currentStreamId = '';
let progressInterval;
let isWebcam = false;
let selectedFiles = [];
let videoStartTime = null;
let videoEndTime = null;
let videoName = '';


// FIXED: Proper button creation with correct CSS class

function createNewProcessingButton(containerId) {
    const buttonsContainer = document.getElementById(containerId);
    if (buttonsContainer) {
        buttonsContainer.innerHTML = '';
        const newSessionButton = document.createElement('button');
        // FIXED: Use the CSS class that exists in style.css
        newSessionButton.className = 'start-new-session-btn';
        newSessionButton.textContent = 'New Processing';
        newSessionButton.onclick = startNewSession;
        buttonsContainer.appendChild(newSessionButton);
    }
}

// Database configuration
async function configureDatabase() {
    const dbUrl = document.getElementById('dbUrl').value.trim();
    const statusDiv = document.getElementById('dbStatus');
    
    if (!dbUrl) {
        statusDiv.innerHTML = '<p style="color: red;">Please enter a database URL</p>';
        statusDiv.style.display = 'block';
        return;
    }
    
    statusDiv.innerHTML = '<p>Connecting...</p>';
    statusDiv.style.display = 'block';
    
    try {
        const formData = new FormData();
        formData.append('database_url', dbUrl);
        
        const response = await fetch('/configure-database/', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            isDatabaseConnected = true;
            statusDiv.innerHTML = '<p style="color: green;">✅ ' + result.message + '</p>';
            setTimeout(() => {
                document.getElementById('dbConfigModal').style.display = 'none';
            }, 2000);
        } else {
            isDatabaseConnected = false;
            statusDiv.innerHTML = '<p style="color: red;">❌ ' + result.error + '</p>';
        }
    } catch (error) {
        isDatabaseConnected = false;
        statusDiv.innerHTML = '<p style="color: red;">❌ Error: ' + error.message + '</p>';
    }
}

async function skipDatabase() {
    try {
        const formData = new FormData();
        formData.append('skip_database', 'true');
        
        const response = await fetch('/configure-database/', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        isDatabaseConnected = false;
        alert('⚠️ ' + result.message);
        document.getElementById('dbConfigModal').style.display = 'none';
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

// Modify downloadDashboardStats to use new endpoint
async function downloadDashboardStats() {
    if (!currentStreamId) {
        alert('❌ No active session');
        return;
    }
    
    try {
        window.location.href = `/download-dashboard-report/${currentStreamId}`;
        alert('✅ Report downloaded successfully!');
    } catch (error) {
        alert('❌ Error: ' + error.message);
    }
}

// File selection handler
document.getElementById('videoInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileInfo').style.display = 'block';
        document.getElementById('webcamInfo').style.display = 'none';
        isWebcam = false;
        videoName = file.name;
    }
});

// Start webcam button
function startWebcam() {
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('webcamInfo').style.display = 'block';
    isWebcam = true;
    videoName = 'Webcam';
}

// Get selected camera view for video
function getSelectedCameraView() {
    const selectedView = document.querySelector('input[name="cameraView"]:checked');
    return selectedView ? selectedView.value : 'Front';
}

// Get selected camera view for webcam
function getSelectedWebcamView() {
    const selectedView = document.querySelector('input[name="webcamView"]:checked');
    return selectedView ? selectedView.value : 'Front';
}
// FIXED: Start webcam processing with browser capture
async function startWebcamProcessing() {
    const startBtn = document.getElementById('startWebcamBtn');
    startBtn.disabled = true;
    startBtn.textContent = 'Starting Webcam...';

    const cameraView = getSelectedWebcamView();
    videoStartTime = new Date();

    try {
        console.log('🎥 Requesting browser camera access...');
        
        // Request camera access from browser
        videoStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 640 },
                facingMode: 'user'
            } 
        });
        
        console.log('✅ Camera access granted');
        
        // Create hidden video element to capture frames
        const videoElement = document.createElement('video');
        videoElement.srcObject = videoStream;
        videoElement.autoplay = true;
        videoElement.playsInline = true;
        
        // Wait for video to be ready
        await new Promise((resolve, reject) => {
            videoElement.onloadedmetadata = () => {
                videoElement.play();
                console.log(`📹 Video ready: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
                resolve();
            };
            videoElement.onerror = reject;
            setTimeout(reject, 5000); // 5 second timeout
        });
        
        // Initialize webcam session on server
        const formData = new FormData();
        formData.append('camera_view', cameraView);

        console.log('🌐 Initializing server session...');
        const response = await fetch('/start-webcam/', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        if (response.ok) {
            currentStreamId = result.session_id;
            isWebcam = true;
            console.log(`✅ Webcam session initialized: ${currentStreamId}`);
            
            // Show processing UI
            showProcessingUI();
            startLiveStream();
            startProgressMonitoring();
            
            // Start capturing and sending frames
            startFrameCapture(videoElement, result.session_id);
            
            startBtn.textContent = '🎬 Start Webcam Processing';
        } else {
            throw new Error(result.detail || 'Failed to initialize webcam session');
        }
    } catch (error) {
        console.error('❌ Webcam error:', error);
        
        if (error.name === 'NotAllowedError') {
            alert('❌ Camera access denied. Please allow camera access and try again.');
        } else if (error.name === 'NotFoundError') {
            alert('❌ No camera found on your device.');
        } else if (error.name === 'NotReadableError') {
            alert('❌ Camera is already in use by another application.');
        } else {
            alert('❌ Error accessing camera: ' + error.message);
        }
        
        stopVideoStream();
        startBtn.textContent = '🎬 Start Webcam Processing';
    } finally {
        startBtn.disabled = false;
    }
}
// FIXED: Capture frames from browser and send to server
function startFrameCapture(videoElement, sessionId) {
    // Create canvas for frame capture
    webcamCanvas = document.createElement('canvas');
    const ctx = webcamCanvas.getContext('2d');
    
    // Set canvas size to match video
    webcamCanvas.width = videoElement.videoWidth || 640;
    webcamCanvas.height = videoElement.videoHeight || 640;
    
    console.log(`🎨 Canvas created: ${webcamCanvas.width}x${webcamCanvas.height}`);
    
    let frameCount = 0;
    let lastFrameTime = Date.now();
    const targetFPS = 15;
    const frameInterval = 1000 / targetFPS;
    
    // Capture and send frames
    webcamInterval = setInterval(async () => {
        const now = Date.now();
        const elapsed = now - lastFrameTime;
        
        // Throttle to target FPS
        if (elapsed < frameInterval) {
            return;
        }
        
        lastFrameTime = now;
        
        if (!videoElement.srcObject) {
            console.log('⚠️ Video stream lost');
            stopFrameCapture();
            return;
        }
        
        try {
            // Draw current video frame to canvas
            ctx.drawImage(videoElement, 0, 0, webcamCanvas.width, webcamCanvas.height);
            
            // Convert canvas to base64 JPEG
            const frameDataUrl = webcamCanvas.toDataURL('image/jpeg', 0.8);
            
            // Send frame to server
            const formData = new FormData();
            formData.append('frame_data', frameDataUrl);
            
            const response = await fetch(`/process-webcam-frame/${sessionId}`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error(`❌ Frame ${frameCount} failed:`, response.status, errorText);
            } else {
                const result = await response.json();
                if (frameCount % 30 === 0) { // Log every 30 frames
                    console.log(`✅ Frame ${frameCount} processed`);
                }
            }
            
            frameCount++;
        } catch (error) {
            console.error('❌ Error capturing/sending frame:', error);
        }
    }, 50); // Check every 50ms, but throttle to targetFPS
    
    console.log(`🎬 Frame capture started at ${targetFPS} FPS`);
}


// Stop frame capture
function stopFrameCapture() {
    if (webcamInterval) {
        clearInterval(webcamInterval);
        webcamInterval = null;
        console.log('⏹️ Frame capture stopped');
    }
    stopVideoStream();
}

// Stop video stream
function stopVideoStream() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => {
            track.stop();
            console.log(`🛑 Track stopped: ${track.kind}`);
        });
        videoStream = null;
    }
}
// Show webcam stopped with download option
function showWebcamStopped(videoPath) {
    const stoppedMessage = document.getElementById('stoppedMessage');
    const videoDownloadSection = document.getElementById('videoDownloadSection');
    const partialDownloadLink = document.getElementById('partialDownloadLink');
    
    stoppedMessage.textContent = 'Webcam processing stopped. Download your processed video below.';
    videoDownloadSection.style.display = 'block';
    
    if (videoPath) {
        partialDownloadLink.href = videoPath;
        partialDownloadLink.download = `webcam_processed_${currentStreamId}.mp4`;
    }
    
    document.getElementById('stoppedSection').style.display = 'block';
    document.getElementById('progressSection').style.display = 'none';
    document.getElementById('liveFramesSection').style.display = 'none';
    document.getElementById('resultSection').style.display = 'none';
    
    createNewProcessingButton('stoppedSectionButtons');
}
// Upload and process video
async function uploadVideo() {
    const fileInput = document.getElementById('videoInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showError('Please select a video file first');
        return;
    }

    const uploadBtn = document.getElementById('uploadBtn');
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Uploading...';

    const cameraView = getSelectedCameraView();
    videoStartTime = new Date();

    const formData = new FormData();
    formData.append('file', file);
    formData.append('camera_view', cameraView);

    try {
        const response = await fetch('/upload-video/', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        if (response.ok) {
            currentStreamId = result.filename;
            console.log(`Video uploaded with camera view: ${result.camera_view}`);
            isWebcam = false;
            showProcessingUI();
            startLiveStream();
            startProgressMonitoring();
        } else {
            showError(result.detail || 'Upload failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Start Video Processing';
    }
}

// Stop processing
async function stopProcessing() {
    const stopBtn = document.getElementById('stopBtn');
    stopBtn.disabled = true;
    stopBtn.textContent = 'Stopping...';

    videoEndTime = new Date();

    try {
        if (isWebcam) {
            // Stop frame capture first
            stopFrameCapture();
            
            // Stop webcam session on server
            const response = await fetch(`/stop-webcam/${currentStreamId}`, {
                method: 'POST'
            });

            const result = await response.json();
            
            if (response.ok) {
                showWebcamStopped(result.video_path);
            } else {
                showError('Failed to stop processing');
            }
        } else {
            // Original video stop logic
            const response = await fetch(`/stop-processing/${currentStreamId}`, {
                method: 'POST'
            });

            const result = await response.json();
            
            if (response.ok) {
                showStopped();
            } else {
                showError('Failed to stop processing');
            }
        }
    } catch (error) {
        showError('Error stopping processing: ' + error.message);
    } finally {
        stopBtn.disabled = false;
        stopBtn.textContent = 'Stop Processing';
    }
}
// Start live stream
function startLiveStream() {
    const streamImg = document.getElementById('liveStream');
    streamImg.src = `/live-stream/${currentStreamId}`;
    
    streamImg.onload = function() {
        document.getElementById('frameStatus').textContent = 'Stream connected - processing in real-time';
    };
    
    streamImg.onerror = function() {
        document.getElementById('frameStatus').textContent = 'Reconnecting...';
        setTimeout(() => {
            streamImg.src = `/live-stream/${currentStreamId}?t=${new Date().getTime()}`;
        }, 2000);
    };
}
// Start progress monitoring with dashboard updates
function startProgressMonitoring() {
    if (progressInterval) {
        clearInterval(progressInterval);
    }
    
    progressInterval = setInterval(async () => {
        try {
            const response = await fetch(`/stream-progress/${currentStreamId}`);
            if (!response.ok) {
                throw new Error('Failed to fetch progress');
            }
            const status = await response.json();
            
            if (status.dashboard_data) {
                updateDashboard(status.dashboard_data);
            }
            
            updateProgressUI(status);
            
            if (status.status === 'completed') {
                videoEndTime = new Date();
                handleCompletion(status);
                clearInterval(progressInterval);
            } else if (status.status === 'stopped') {
                showStopped();
                clearInterval(progressInterval);
            } else if (status.status === 'error') {
                showError(status.error || 'Processing failed');
                clearInterval(progressInterval);
            }
            
        } catch (error) {
            console.error('Error checking progress:', error);
        }
    }, 500);
}

// Update dashboard with real-time data
function updateDashboard(data) {
    document.getElementById('totalFaces').textContent = data.total_faces || 0;
    document.getElementById('knownFaces').textContent = data.known_faces_num || 0;
    document.getElementById('unknownFaces').textContent = data.unknown_faces_num || 0;
}
// Update progress UI
function updateProgressUI(status) {
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const frameCount = document.getElementById('frameCount');
    const sourceInfo = document.getElementById('sourceInfo');
    const processingTitle = document.getElementById('processingTitle');
    
    if (status.stream_type === 'webcam') {
        processingTitle.textContent = `Webcam Processing (${status.camera_view} View)`;
        sourceInfo.textContent = `Source: Webcam (Live) - ${status.camera_view} View`;
        frameCount.textContent = `Frames: ${status.processed_frames}`;
        progressText.textContent = `Live processing: ${status.processed_frames} frames`;
        progressBar.style.width = '100%';
    } else {
        processingTitle.textContent = `Video Processing (${status.camera_view} View)`;
        sourceInfo.textContent = `Source: Uploaded Video - ${status.camera_view} View`;
        const progress = status.progress || 0;
        progressBar.style.width = progress + '%';
        progressText.textContent = `Processing: ${progress.toFixed(1)}%`;
        frameCount.textContent = `Frames: ${status.processed_frames}/${status.total_frames}`;
    }
    
    document.getElementById('stopBtn').disabled = !status.can_stop;
}

// Handle completion
function handleCompletion(status) {
    if (status.stream_type === 'video') {
        showFinalResult();
    } else {
        showStopped();
    }
}

// Make function available globally
window.downloadDashboardStats = downloadDashboardStats;

// Download dashboard statistics
async function downloadDashboardStats() {
    console.log('Download button clicked!');
    console.log('Current stream ID:', currentStreamId);
    
    if (!currentStreamId) {
        alert('❌ No active session. Please start processing first.');
        return;
    }
    
    try {
        console.log('Fetching dashboard data...');
        const response = await fetch(`/dashboard-data/${currentStreamId}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Dashboard data received:', data);
        
        // Create text content for the report
        const reportContent = generateReportText(data);
        console.log('Report generated, length:', reportContent.length);
        
        // Create a blob and download
        const blob = new Blob([reportContent], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const fileName = `dashboard_report_${videoName.replace(/\.[^/.]+$/, '')}_${timestamp}.txt`;
        a.download = fileName;
        
        console.log('Downloading file:', fileName);
        
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        console.log('✅ Report downloaded successfully');
        alert('✅ Report downloaded successfully!');
    } catch (error) {
        console.error('Error downloading report:', error);
        alert('❌ Error downloading report: ' + error.message);
    }
}

// Generate report text
function generateReportText(data) {
    const startStr = videoStartTime ? videoStartTime.toLocaleString() : 'N/A';
    const endStr = videoEndTime ? videoEndTime.toLocaleString() : 'N/A';
    
    let report = `========================================\n`;
    report += `   FACE RECOGNITION DASHBOARD REPORT\n`;
    report += `========================================\n\n`;
    
    report += `Video/Source Name: ${videoName}\n`;
    report += `Start Time: ${startStr}\n`;
    report += `End Time: ${endStr}\n\n`;
    
    report += `========================================\n`;
    report += `   STATISTICS SUMMARY\n`;
    report += `========================================\n\n`;
    
    report += `Total Faces Detected: ${data.total_faces || 0}\n`;
    report += `Known Faces: ${data.known_faces_num || 0}\n`;
    report += `Unknown Faces: ${data.unknown_faces_num || 0}\n\n`;
    
    // List known person IDs
    if (data.known_faces && data.known_faces.length > 0) {
        report += `========================================\n`;
        report += `   KNOWN PERSONS DETECTED\n`;
        report += `========================================\n\n`;
        
        const uniquePersons = {};
        data.known_faces.forEach(face => {
            const personId = face.subject;
            if (!uniquePersons[personId]) {
                uniquePersons[personId] = {
                    count: 0,
                    bestScore: face.score || 0
                };
            }
            uniquePersons[personId].count++;
            if (face.score && face.score < uniquePersons[personId].bestScore) {
                uniquePersons[personId].bestScore = face.score;
            }
        });
        
        Object.keys(uniquePersons).sort((a, b) => a - b).forEach(personId => {
            const info = uniquePersons[personId];
            report += `Person ID: ${personId}\n`;
            report += `  - Appearances: ${info.count}\n`;
            report += `  - Best Match Score: ${info.bestScore.toFixed(3)}\n\n`;
        });
    }
    
    report += `========================================\n`;
    report += `   END OF REPORT\n`;
    report += `========================================\n`;
    
    return report;
}

// Show known faces modal
async function showKnownFaces() {
    if (!currentStreamId) {
        alert('No active processing session');
        return;
    }
    
    try {
        const response = await fetch(`/known-faces/${currentStreamId}`);
        if (!response.ok) {
            throw new Error('Failed to fetch known faces');
        }
        const data = await response.json();
        
        const modal = document.getElementById('facesModal');
        const modalTitle = document.getElementById('modalTitle');
        const modalSubtitle = document.getElementById('modalSubtitle');
        const gallery = document.getElementById('facesGallery');
        
        modalTitle.textContent = '✅ Known Faces Gallery';
        modalSubtitle.textContent = `Showing ${data.known_faces.length} recognized faces`;
        gallery.innerHTML = '';
        
        if (data.known_faces.length === 0) {
            gallery.innerHTML = '<div class="no-faces-message"><p>No known faces detected yet.</p><p>Keep processing to see recognized faces here.</p></div>';
        } else {
            data.known_faces.forEach(face => {
                const faceItem = document.createElement('div');
                faceItem.className = 'face-item';
                
                let scoreText = 'N/A';
                if (face.score !== 'N/A' && typeof face.score === 'number') {
                    scoreText = face.score.toFixed(3);
                } else if (face.score !== 'N/A' && !isNaN(parseFloat(face.score))) {
                    scoreText = parseFloat(face.score).toFixed(3);
                }
                
                faceItem.innerHTML = `
                    <img src="${face.face_image}" alt="${face.subject}" class="face-image">
                    <div class="face-info">
                        <div class="face-subject">ID: ${face.subject}</div>
                        <div class="face-details">
                            <div>Score: ${scoreText}</div>
                            <div>Track ID: ${face.track_id || 'N/A'}</div>
                            <div>Frame: ${face.frame_id || 'N/A'}</div>
                        </div>
                    </div>
                `;
                gallery.appendChild(faceItem);
            });
        }
        
        modal.style.display = 'block';
    } catch (error) {
        console.error('Error loading known faces:', error);
        alert('Error loading known faces: ' + error.message);
    }
}

// Show unknown faces modal (removed click to register functionality)
async function showUnknownFaces() {
    if (!currentStreamId) {
        alert('No active processing session');
        return;
    }
    
    try {
        const response = await fetch(`/unknown-faces/${currentStreamId}`);
        if (!response.ok) {
            throw new Error('Failed to fetch unknown faces');
        }
        const data = await response.json();
        
        const modal = document.getElementById('facesModal');
        const modalTitle = document.getElementById('modalTitle');
        const modalSubtitle = document.getElementById('modalSubtitle');
        const gallery = document.getElementById('facesGallery');
        
        modalTitle.textContent = '❓ Unknown Faces Gallery';
        modalSubtitle.textContent = `Showing ${data.unknown_faces.length} unrecognized faces`;
        gallery.innerHTML = '';
        
        if (data.unknown_faces.length === 0) {
            gallery.innerHTML = '<div class="no-faces-message"><p>No unknown faces detected yet.</p><p>All detected faces have been recognized.</p></div>';
        } else {
            data.unknown_faces.forEach(face => {
                const faceItem = document.createElement('div');
                faceItem.className = 'face-item';
                
                let scoreText = 'N/A';
                if (face.score !== 'N/A' && typeof face.score === 'number') {
                    scoreText = face.score.toFixed(3);
                } else if (face.score !== 'N/A' && !isNaN(parseFloat(face.score))) {
                    scoreText = parseFloat(face.score).toFixed(3);
                }
                
                faceItem.innerHTML = `
                    <img src="${face.face_image}" alt="Unknown Face" class="face-image">
                    <div class="face-info">
                        <div class="face-unknown">Unknown Person</div>
                        <div class="face-details">
                            <div>Track ID: ${face.track_id || 'N/A'}</div>
                            <div>Frame: ${face.frame_id || 'N/A'}</div>
                            <div>Score: ${scoreText}</div>
                        </div>
                    </div>
                `;
                gallery.appendChild(faceItem);
            });
        }
        
        modal.style.display = 'block';
    } catch (error) {
        console.error('Error loading unknown faces:', error);
        alert('Error loading unknown faces: ' + error.message);
    }
}

// Close modal
function closeModal() {
    document.getElementById('facesModal').style.display = 'none';
}

// Register modal functions
function openRegisterModal() {
    console.log('Opening register modal');
    const registerModal = document.getElementById('registerModal');
    if (registerModal) {
        registerModal.style.display = 'block';
        document.getElementById('personId').value = '';
        selectedFiles = [];
        updatePreview();
    }
}

function closeRegisterModal() {
    document.getElementById('registerModal').style.display = 'none';
    document.getElementById('personId').value = '';
    document.getElementById('faceImages').value = '';
    selectedFiles = [];
    updatePreview();
}

document.addEventListener('DOMContentLoaded', function() {
    const faceImagesInput = document.getElementById('faceImages');
    
    if (faceImagesInput) {
        faceImagesInput.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            handleFiles(files);
        });
    }
    
    window.removeImage = function(index) {
        selectedFiles.splice(index, 1);
        updatePreview();
    };
});

function handleFiles(files) {
    const imageFiles = files.filter(file => 
        file.type.match('image/(jpeg|jpg|png|bmp)')
    );
    
    if (selectedFiles.length + imageFiles.length > 5) {
        alert('❌ Maximum 5 images allowed');
        return;
    }
    
    selectedFiles = selectedFiles.concat(imageFiles);
    updatePreview();
}

function updatePreview() {
    const previewContainer = document.getElementById('previewContainer');
    const imageCount = document.getElementById('imageCount');
    const registerBtn = document.getElementById('registerBtnModal');
    
    if (!previewContainer || !imageCount || !registerBtn) return;
    
    previewContainer.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const reader = new FileReader();
        
        reader.onload = (e) => {
            const previewItem = document.createElement('div');
            previewItem.className = 'preview-item';
            previewItem.innerHTML = `
                <img src="${e.target.result}" alt="Preview ${index + 1}">
                <button type="button" class="remove-preview-btn" onclick="window.removeImage(${index})">✕</button>
            `;
            previewContainer.appendChild(previewItem);
        };
        
        reader.readAsDataURL(file);
    });
    
    if (selectedFiles.length > 0) {
        const pluralText = selectedFiles.length > 1 ? 's' : '';
        imageCount.textContent = `✅ ${selectedFiles.length} image${pluralText} selected`;
        if (selectedFiles.length === 5) {
            imageCount.textContent += ' (Maximum reached)';
            imageCount.style.color = '#ef4444';
            imageCount.style.background = '#fee2e2';
        } else {
            imageCount.style.color = '#10b981';
            imageCount.style.background = '#d1fae5';
        }
        imageCount.style.display = 'block';
    } else {
        imageCount.textContent = 'No images selected';
        imageCount.style.display = 'block';
        imageCount.style.color = '#6b7280';
        imageCount.style.background = '#f3f4f6';
    }
    
    registerBtn.disabled = selectedFiles.length === 0;
}

async function registerNewFace() {
    const personId = document.getElementById('personId').value.trim();
    
    if (!personId) {
        alert('❌ Please enter the person ID');
        return;
    }
    
    if (selectedFiles.length === 0) {
        alert('❌ Please select at least one face image');
        return;
    }
    
    // ADD THIS CHECK:
    if (!isDatabaseConnected) {
        const confirmMsg = '⚠️ You are in temporary mode (no database).\n\n' +
                          'The face will be registered temporarily and will be lost when you refresh the page.\n\n' +
                          'Do you want to continue?';
        if (!confirm(confirmMsg)) {
            return;
        }
    }
    
    const registerBtn = document.getElementById('registerBtnModal');
    registerBtn.disabled = true;
    registerBtn.textContent = 'Registering...';
    
    try {
        const formData = new FormData();
        formData.append('person_id', personId);
        
        selectedFiles.forEach((file) => {
            formData.append('images', file);
        });
        
        const response = await fetch('/register-face-multiple/', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            const storageType = result.details.storage_type || (isDatabaseConnected ? 'database' : 'memory');
            const storageMsg = storageType === 'database' 
                ? '\n\n✅ Saved to database permanently.' 
                : '\n\n⚠️ Saved temporarily (will be lost on refresh).';
            
            alert(`✅ Successfully registered person ID ${personId}!` +
                  `\n\nImages processed: ${result.details.images_processed}/${result.details.images_provided}` +
                  storageMsg);
            
            document.getElementById('personId').value = '';
            document.getElementById('faceImages').value = '';
            selectedFiles = [];
            updatePreview();
            closeRegisterModal();
        } else {
            alert(`❌ Registration failed: ${result.detail || 'Unknown error'}`);
        }
    } catch (error) {
        alert(`❌ Error: ${error.message}`);
    } finally {
        registerBtn.disabled = false;
        registerBtn.textContent = 'Register Face';
    }
}

// Quick register from unknown face image
async function openQuickRegister(imageDataUrl) {
    closeModal();
    
    const personId = prompt('Enter the person ID (numeric):');
    
    if (!personId || personId.trim() === '') {
        alert('❌ Registration cancelled - person ID is required');
        return;
    }
    
    try {
        const loadingMsg = document.createElement('div');
        loadingMsg.id = 'loadingMsg';
        loadingMsg.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:rgba(0,0,0,0.8);color:white;padding:20px 40px;border-radius:10px;z-index:10000;font-size:1.2em;';
        loadingMsg.textContent = `Registering person ID ${personId.trim()}...`;
        document.body.appendChild(loadingMsg);
        
        // Extract base64 from data URL
        const base64Data = imageDataUrl.split(',')[1];
        
        const formData = new FormData();
        formData.append('person_id', personId.trim());
        formData.append('image_data_list', JSON.stringify([imageDataUrl]));
        
        const response = await fetch('/register-face-from-images/', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        document.body.removeChild(loadingMsg);
        
        if (response.ok) {
            alert(`✅ Successfully registered person ID ${personId.trim()}!`);
        } else {
            alert(`❌ Registration failed: ${result.detail || 'Unknown error'}`);
        }
    } catch (error) {
        const loadingMsg = document.getElementById('loadingMsg');
        if (loadingMsg) {
            document.body.removeChild(loadingMsg);
        }
        alert(`❌ Error: ${error.message}`);
    }
}

// Delete modal functions
function openDeleteModal() {
    const deleteModal = document.getElementById('deleteModal');
    if (deleteModal) {
        deleteModal.style.display = 'block';
        document.getElementById('deletePersonId').value = '';
    }
}

function closeDeleteModal() {
    const deleteModal = document.getElementById('deleteModal');
    if (deleteModal) {
        deleteModal.style.display = 'none';
    }
}
async function deletePerson() {
    const personId = document.getElementById('deletePersonId').value.trim();
    
    if (!personId) {
        alert('❌ Please enter a person ID');
        return;
    }
    
    // ADD THIS CHECK:
    const storageMsg = isDatabaseConnected 
        ? '\n\nThis will permanently delete from the database!'
        : '\n\nThis will delete from temporary memory!';
    
    const confirmed = confirm(`⚠️ Are you sure you want to delete person ID ${personId}?${storageMsg}\n\nThis action cannot be undone!`);
    
    if (!confirmed) return;
    
    const deleteBtn = document.getElementById('deleteBtnModal');
    deleteBtn.disabled = true;
    deleteBtn.textContent = 'Deleting...';
    
    try {
        const response = await fetch(`/delete-person/${personId}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (response.ok) {
            const storageType = result.storage_type || (isDatabaseConnected ? 'database' : 'memory');
            const msg = storageType === 'database'
                ? `✅ Successfully deleted person ID ${personId} from database`
                : `✅ Successfully deleted person ID ${personId} from temporary memory`;
            alert(msg);
            closeDeleteModal();
        } else {
            alert(`❌ Deletion failed: ${result.detail || 'Unknown error'}`);
        }
    } catch (error) {
        alert(`❌ Error: ${error.message}`);
    } finally {
        deleteBtn.disabled = false;
        deleteBtn.textContent = 'Delete Person';
    }
}

// ATTENDANCE REPORT FUNCTIONS - FIXED: Check database connection
async function downloadAttendanceReport(reportType) {
    console.log(`Downloading ${reportType} attendance report...`);
    
    // ADD THESE LINES AT THE START:
    if (!isDatabaseConnected) {
        alert('❌ Database not connected. Please configure database first to use attendance reports.');
        return;
    }
    
    try {
        // First check if database is connected
        const statusResponse = await fetch('/database-status/');
        const dbStatus = await statusResponse.json();
        
        if (!dbStatus.connected) {
            alert('❌ Database not connected. Please configure database to use attendance reports.');
            return;
        }
        
        // Show loading message
        const loadingMsg = document.createElement('div');
        loadingMsg.id = 'reportLoadingMsg';
        loadingMsg.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:rgba(0,0,0,0.8);color:white;padding:20px 40px;border-radius:10px;z-index:10000;font-size:1.2em;';
        loadingMsg.textContent = `Generating ${reportType} report...`;
        document.body.appendChild(loadingMsg);
        
        // Call the API to generate and download the report
        const response = await fetch(`/generate-attendance-report/?report_type=${reportType}`);
        
        // Remove loading message
        document.body.removeChild(loadingMsg);
        
        if (response.ok) {
            // Get the file blob
            const blob = await response.blob();
            
            // Extract filename from Content-Disposition header or use default
            const contentDisposition = response.headers.get('content-disposition');
            let filename = `${reportType}_attendance_report.csv`;
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="?(.+)"?/i);
                if (filenameMatch) {
                    filename = filenameMatch[1];
                }
            }
            
            // Create a temporary download link
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            
            // Cleanup
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            alert(`✅ ${reportType === 'monthly' ? 'Monthly' : 'Detailed'} attendance report downloaded successfully!`);
        } else {
            const error = await response.json();
            alert(`❌ Error generating report: ${error.detail || 'Unknown error'}`);
        }
    } catch (error) {
        const loadingMsg = document.getElementById('reportLoadingMsg');
        if (loadingMsg) {
            document.body.removeChild(loadingMsg);
        }
        console.error('Report download error:', error);
        alert(`❌ Error: ${error.message}`);
    }
}



// Close modal when clicking outside
window.onclick = function(event) {
    const facesModal = document.getElementById('facesModal');
    const registerModal = document.getElementById('registerModal');
    const deleteModal = document.getElementById('deleteModal');
    
    if (event.target === facesModal) closeModal();
    if (event.target === registerModal) closeRegisterModal();
    if (event.target === deleteModal) closeDeleteModal();
}

// Add reset function after processing ends


// Show processing UI
function showProcessingUI() {
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('webcamInfo').style.display = 'none';
    document.getElementById('dashboardStatistics').style.display = 'block';
    document.getElementById('progressSection').style.display = 'block';
    document.getElementById('liveFramesSection').style.display = 'block';
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('stoppedSection').style.display = 'none';
    document.getElementById('errorSection').style.display = 'none';
}
// Show final result
function showFinalResult() {
    const videoElement = document.getElementById('processedVideo');
    const downloadLink = document.getElementById('downloadLink');
    
    if (currentStreamId) {
        videoElement.src = `/video/${currentStreamId}`;
        downloadLink.href = `/video/${currentStreamId}`;
        downloadLink.download = `processed_${currentStreamId}`;
    }
    
    document.getElementById('resultSection').style.display = 'block';
    document.getElementById('progressSection').style.display = 'none';
    document.getElementById('liveFramesSection').style.display = 'none';
    
    // Create button with new text
    createNewProcessingButton('resultSectionButtons');
}
// Show stopped state
function showStopped() {
    const stoppedMessage = document.getElementById('stoppedMessage');
    const videoDownloadSection = document.getElementById('videoDownloadSection');
    const partialDownloadLink = document.getElementById('partialDownloadLink');
    
    if (isWebcam) {
        stoppedMessage.textContent = 'Webcam processing was stopped by user.';
        videoDownloadSection.style.display = 'none';
    } else {
        stoppedMessage.textContent = 'Video processing was stopped by user. You can download the partially processed video.';
        videoDownloadSection.style.display = 'block';
        if (currentStreamId) {
            partialDownloadLink.href = `/video/${currentStreamId}`;
            partialDownloadLink.download = `partial_${currentStreamId}`;
        }
    }
    
    document.getElementById('stoppedSection').style.display = 'block';
    document.getElementById('progressSection').style.display = 'none';
    document.getElementById('liveFramesSection').style.display = 'none';
    document.getElementById('resultSection').style.display = 'none';
    
    // Create button with new text
    createNewProcessingButton('stoppedSectionButtons');
}

// Show error



function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorSection').style.display = 'block';
    
    document.getElementById('progressSection').style.display = 'none';
    document.getElementById('liveFramesSection').style.display = 'none';
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('stoppedSection').style.display = 'none';
    document.getElementById('dashboardStatistics').style.display = 'none';
    
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    
    // Create button with new text
    createNewProcessingButton('errorSectionButtons');
}

// Add reset function after processing ends
function resetProcessingState() {
    console.log("Resetting processing state - Manual reset by user");
    stopFrameCapture();
    // Reset video input
    document.getElementById('videoInput').value = '';
    document.getElementById('fileName').textContent = '';
    
    // Show upload box
    document.getElementById('uploadBox').style.display = 'block';
    
    // Hide all other sections
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('webcamInfo').style.display = 'none';
    document.getElementById('dashboardStatistics').style.display = 'none';
    document.getElementById('progressSection').style.display = 'none';
    document.getElementById('liveFramesSection').style.display = 'none';
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('stoppedSection').style.display = 'none';
    document.getElementById('errorSection').style.display = 'none';
    
    // Clear button containers
    const buttonContainers = ['resultSectionButtons', 'stoppedSectionButtons', 'errorSectionButtons'];
    buttonContainers.forEach(containerId => {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = '';
        }
    });
    
    // Remove any reset buttons that were added directly
    const resetButtons = ['resetButton', 'resetButtonResult', 'resetButtonError'];
    resetButtons.forEach(buttonId => {
        const button = document.getElementById(buttonId);
        if (button) {
            button.remove();
        }
    });
    
    // Clear stream image
    const streamImg = document.getElementById('liveStream');
    streamImg.src = '';
    
    // Reset progress bar
    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('progressText').textContent = 'Processing: 0.0%';
    document.getElementById('frameCount').textContent = 'Frames: 0/0';
    document.getElementById('frameStatus').textContent = 'Connecting to stream...';
    
    // Reset dashboard
    document.getElementById('totalFaces').textContent = '0';
    document.getElementById('knownFaces').textContent = '0';
    document.getElementById('unknownFaces').textContent = '0';
    
    // Clear interval if exists
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    
    // Reset variables
    currentStreamId = '';
    isWebcam = false;
    selectedFiles = [];
    videoStartTime = null;
    videoEndTime = null;
    videoName = '';
    
    console.log("✅ Processing state reset complete - Ready for new processing");
}
function startNewSession() {
    console.log('Starting new processing session...');
    
    stopFrameCapture();
    
    resetProcessingState();
    // Reset video input
    document.getElementById('videoInput').value = '';
    document.getElementById('fileName').textContent = '';
    
    // Show upload box
    document.getElementById('uploadBox').style.display = 'block';
    
    // Hide all other sections
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('webcamInfo').style.display = 'none';
    document.getElementById('dashboardStatistics').style.display = 'none';
    document.getElementById('progressSection').style.display = 'none';
    document.getElementById('liveFramesSection').style.display = 'none';
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('stoppedSection').style.display = 'none';
    document.getElementById('errorSection').style.display = 'none';
    
    // Clear button containers
    const buttonContainers = ['resultSectionButtons', 'stoppedSectionButtons', 'errorSectionButtons'];
    buttonContainers.forEach(containerId => {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = '';
        }
    });
    
    // Clear stream image
    const streamImg = document.getElementById('liveStream');
    streamImg.src = '';
    
    // Reset progress bar
    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('progressText').textContent = 'Processing: 0.0%';
    document.getElementById('frameCount').textContent = 'Frames: 0/0';
    document.getElementById('frameStatus').textContent = 'Connecting to stream...';
    
    // Reset dashboard
    document.getElementById('totalFaces').textContent = '0';
    document.getElementById('knownFaces').textContent = '0';
    document.getElementById('unknownFaces').textContent = '0';
    
    // Clear interval if exists
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    
    // Reset variables
    currentStreamId = '';
    isWebcam = false;
    selectedFiles = [];
    videoStartTime = null;
    videoEndTime = null;
    videoName = '';
    
    console.log('✅ New processing session ready');
}

function createNewProcessingButton(containerId) {
    const buttonsContainer = document.getElementById(containerId);
    if (buttonsContainer) {
        buttonsContainer.innerHTML = '';
        const newSessionButton = document.createElement('button');
        newSessionButton.className = 'start-new-session-btn';
        newSessionButton.textContent = 'New Processing'; // CHANGED TEXT
        newSessionButton.onclick = startNewSession;
        buttonsContainer.appendChild(newSessionButton);
    }
}


// Optional: Add this notification function if not already in your script.js
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">×</button>
    `;
    
    // Style the notification
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        z-index: 10000;
        display: flex;
        justify-content: space-between;
        align-items: center;
        min-width: 300px;
        max-width: 400px;
        animation: slideIn 0.3s ease;
    `;
    
    // Add styles for notification button
    const closeBtn = notification.querySelector('button');
    closeBtn.style.cssText = `
        background: transparent;
        border: none;
        color: white;
        font-size: 20px;
        cursor: pointer;
        margin-left: 15px;
        padding: 0;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        transition: background 0.3s;
    `;
    
    closeBtn.onmouseover = function() {
        this.style.background = 'rgba(255,255,255,0.2)';
    };
    
    closeBtn.onmouseout = function() {
        this.style.background = 'transparent';
    };
    
    // Add animation
    const style = document.createElement('style');
    style.textContent = `
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
    `;
    document.head.appendChild(style);
    
    // Add to body
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.animation = 'slideIn 0.3s ease reverse';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

// Clean up when page is closed
window.addEventListener('beforeunload', function() {
    if (progressInterval) {
        clearInterval(progressInterval);
    }
});