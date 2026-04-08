// Global State
let currentTab = 'exudate';
let isImageUploaded = false;
let stream = null;

// Modules Data
const modulesData = {
    exudate: { name: 'Retinal', color: 'var(--accent-blue)' },
    cataract: { name: 'Cataract', color: 'var(--accent-purple)' },
    jaundice: { name: 'Jaundice', color: 'var(--accent-amber)' }
};

// DOM Elements
const nav = document.getElementById('navbar');
const counters = document.querySelectorAll('.stat-counter');
const sections = document.querySelectorAll('section');
const navLinks = document.querySelectorAll('.nav-links a');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const imagePreview = document.getElementById('image-preview');
const uploadPrompt = document.getElementById('upload-prompt');
const removeBtn = document.getElementById('remove-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const cameraFeed = document.getElementById('camera-feed');
const resultsPanel = document.getElementById('results-panel');
const loader = document.getElementById('loader');

let tfliteModel;

// script.js
async function initializeModel() {
    try {
        // Core TF path synchronization (0.0.1-alpha.8 is the most stable for this model type)
        tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.8/dist/');
        
        tfliteModel = await tflite.loadTFLiteModel('./eyedetecter.tflite');
        console.log("✅ ExuDetect AI Model Initialized!");
    } catch (error) {
        console.error("❌ Model load error:", error);
    }
}

// --- 1. UI & Animations --- //

// Scroll Logic for Nav and Active Links
window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        if (scrollY >= sectionTop - 150) {
            current = section.getAttribute('id');
        }
    });

    if (scrollY > 50) {
        nav.classList.add('scrolled');
    } else {
        nav.classList.remove('scrolled');
    }

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// Intersection Observer for Counters
const counterObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const target = entry.target;
            const finalVal = parseInt(target.getAttribute('data-target'));
            animateCounter(target, finalVal);
            observer.unobserve(target);
        }
    });
}, { threshold: 0.5 });

counters.forEach(counter => counterObserver.observe(counter));

function animateCounter(el, target) {
    let val = 0;
    const inc = Math.max(1, target / 50);
    const updateCount = () => {
        val += inc;
        if (val < target) {
            el.innerText = Math.ceil(val) + (target > 1000 ? '+' : '');
            setTimeout(updateCount, 30);
        } else {
            el.innerText = target + (target > 1000 ? '+' : '');
        }
    };
    updateCount();
}

// FAQ Toggle
function toggleFaq(el) {
    const answer = el.querySelector('.faq-answer');
    const icon = el.querySelector('span');
    if (answer.style.maxHeight) {
        answer.style.maxHeight = null;
        icon.style.transform = 'rotate(0deg)';
    } else {
        answer.style.maxHeight = answer.scrollHeight + "px";
        icon.style.transform = 'rotate(45deg)';
    }
}

// --- 2. Scanner Logic --- //

function switchTab(tabId) {
    currentTab = tabId;
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        btn.style.borderBottomColor = 'transparent';
        btn.style.color = 'var(--text-muted)';
    });
    const activeBtn = document.getElementById(`tab-${tabId}`);
    activeBtn.classList.add('active');
    activeBtn.style.borderBottomColor = modulesData[tabId].color;
    activeBtn.style.color = 'var(--text-main)';

    document.getElementById('jaundice-toggle').style.display = tabId === 'jaundice' ? 'flex' : 'none';
    resetScanner();
}

function openScanner(tabId) {
    document.getElementById('scanner').scrollIntoView({ behavior: 'smooth' });
    switchTab(tabId);
}

// Dropzone Setup
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = modulesData[currentTab].color;
    dropzone.style.background = 'rgba(255,255,255,0.05)';
});
dropzone.addEventListener('dragleave', () => {
    dropzone.style.borderColor = 'rgba(255,255,255,0.2)';
    dropzone.style.background = 'rgba(0,0,0,0.2)';
});
dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'rgba(255,255,255,0.2)';
    dropzone.style.background = 'rgba(0,0,0,0.2)';
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) return alert('Please upload an image file.');
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'block';
        showUploadState();
    };
    reader.readAsDataURL(file);
}

async function startCamera() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            cameraFeed.srcObject = stream;
            cameraFeed.style.display = 'block';
            cameraFeed.play();

            // Add capture button temporarily
            const captureBtn = document.createElement('button');
            captureBtn.className = 'btn btn-primary';
            captureBtn.id = 'temp-capture-btn';
            captureBtn.innerText = '⚡ Capture';
            captureBtn.style.position = 'absolute';
            captureBtn.style.bottom = '20px';
            captureBtn.style.left = '50%';
            captureBtn.style.transform = 'translateX(-50%)';
            captureBtn.onclick = (e) => {
                e.stopPropagation();
                captureImage();
            };
            dropzone.appendChild(captureBtn);

            uploadPrompt.style.display = 'none';
            removeBtn.style.display = 'block';
        } catch (err) {
            alert("Camera access denied or unavailable.");
        }
    } else {
        alert("Camera API not supported in this browser.");
    }
}

function captureImage() {
    const canvas = document.createElement('canvas');
    canvas.width = cameraFeed.videoWidth;
    canvas.height = cameraFeed.videoHeight;
    canvas.getContext('2d').drawImage(cameraFeed, 0, 0);
    imagePreview.src = canvas.toDataURL('image/jpeg');

    stopCamera();
    cameraFeed.style.display = 'none';
    imagePreview.style.display = 'block';
    showUploadState();
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    const capBtn = document.getElementById('temp-capture-btn');
    if (capBtn) capBtn.remove();
}

function showUploadState() {
    uploadPrompt.style.display = 'none';
    removeBtn.style.display = 'block';
    analyzeBtn.disabled = false;
    isImageUploaded = true;
    resultsPanel.style.display = 'none';
}

function resetScanner() {
    stopCamera();
    imagePreview.style.display = 'none';
    cameraFeed.style.display = 'none';
    uploadPrompt.style.display = 'block';
    removeBtn.style.display = 'none';
    analyzeBtn.disabled = true;
    fileInput.value = '';
    isImageUploaded = false;
    resultsPanel.style.display = 'none';
}

async function analyzeImage() {
    if (!isImageUploaded) return;

    analyzeBtn.style.display = 'none';
    loader.style.display = 'block';
    resultsPanel.style.display = 'none';

    try {
        const imageBlob = await fetch(imagePreview.src).then(res => res.blob());
        const formData = new FormData();
        formData.append('image', imageBlob, 'eye_scan.jpg');

        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Backend Offline');

        const scores = await response.json();
        
      
        console.log("DEBUG: Backend se ye data aaya ->", scores);
        console.log("DEBUG: Current Tab active hai ->", currentTab);

        let finalScore = 0;
        
        
        if (currentTab === 'cataract') {
            finalScore = scores.cataract;
        } else if (currentTab === 'exudate') { 
            
            finalScore = scores.retinopathy; 
        } else if (currentTab === 'jaundice') {
            finalScore = scores.jaundice;
        }

        console.log("DEBUG: Final Score processed ->", finalScore);
        generateResults(finalScore);

    } catch (err) {
        console.error("Fetch Error:", err);
        alert("Server Error! Check if app.py is running.");
    } finally {
        loader.style.display = 'none';
        analyzeBtn.style.display = 'inline-flex';
    }
}
function generateResults(score, aiSpecificData = null) {
    const isPositive = score > 0.1; 
    const conf = (score * 100).toFixed(1);

    // Safe Data Retrieval
    const activeModule = modulesData[currentTab] || { name: 'Analysis', color: '#00d4ff' };

    const statusEl = document.getElementById('res-status');
    const sevEl = document.getElementById('res-severity');
    const confText = document.getElementById('res-confidence-text');
    const confBar = document.getElementById('res-confidence-bar');
    const specEl = document.getElementById('res-specific');
    const recEl = document.getElementById('res-recommendation');

    // UI Updates using activeModule
    confText.innerText = `${conf}%`;
    confBar.style.width = `${conf}%`;
    confBar.style.background = `linear-gradient(90deg, ${activeModule.color}, #fff)`;
    confText.style.color = activeModule.color;

    let resultCategory = isPositive ? 'Detected' : 'Normal';
    let severity = 'None';
    let specific = '';
    let rec = '';

    if (isPositive) {
        statusEl.innerText = "Attention Required";
        statusEl.style.background = "rgba(239, 68, 68, 0.2)";
        statusEl.style.color = "var(--danger-color)";
        statusEl.style.borderColor = "rgba(239, 68, 68, 0.5)";

        // Severity Logic
        if (score > 0.85) severity = 'Severe';
        else if (score > 0.65) severity = 'Moderate';
        else severity = 'Mild';

        if (currentTab === 'exudate') {
            specific = `Retinopathy Indicated: YES\nAI Confidence: ${conf}%`;
            rec = severity === 'Severe' ? 'Refer immediately to an ophthalmologist. High risk of vision loss.' : 'Schedule comprehensive dilated eye exam within 2 weeks.';
        } else if (currentTab === 'cataract') {
            specific = `Lens Opacity Detected\nSeverity Grade: ${severity}`;
            rec = severity === 'Severe' ? 'Surgical intervention strongly recommended.' : 'Monitor visual acuity changes closely.';
        } else if (currentTab === 'jaundice') {
            const bLevel = (score * 15).toFixed(1); // Mock mapping for demo
            specific = `Est. Bilirubin Level: ~${bLevel} mg/dL`;
            rec = bLevel > 10 ? 'Requires immediate serum bilirubin blood test.' : 'Monitor closely and stay hydrated.';
        }
    } else {
        statusEl.innerText = "No Anomalies Detected";
        statusEl.style.background = "rgba(16, 185, 129, 0.2)";
        statusEl.style.color = "var(--success-color)";
        statusEl.style.borderColor = "rgba(16, 185, 129, 0.5)";
        sevEl.innerText = "None";
        specific = "All biomarkers within normal range.";
        rec = "Maintain routine screening schedule.";
    }

    sevEl.innerText = isPositive ? severity : 'None';
    specEl.innerText = specific;
    recEl.innerText = rec;

    resultsPanel.style.display = 'block';

    // Save to history (Local Storage)
    saveHistory(currentTab, resultCategory, isPositive ? severity : 'None', conf);
}
// --- 3. Data & Storage --- //

function loadProfile() {
    const p = JSON.parse(localStorage.getItem('mediscan_profile'));
    if (p) {
        document.getElementById('p-name').value = p.name;
        document.getElementById('p-age').value = p.age;
        document.getElementById('p-gender').value = p.gender;
        document.getElementById('p-id').value = p.id;
        document.getElementById('p-notes').value = p.notes;
    }
}

function saveProfile(e) {
    e.preventDefault();
    const profile = {
        name: document.getElementById('p-name').value,
        age: document.getElementById('p-age').value,
        gender: document.getElementById('p-gender').value,
        id: document.getElementById('p-id').value,
        notes: document.getElementById('p-notes').value
    };
    localStorage.setItem('mediscan_profile', JSON.stringify(profile));
    alert("Patient profile saved locally!");
}

function loadHistory() {
    const h = JSON.parse(localStorage.getItem('mediscan_history')) || [];
    const tbody = document.getElementById('history-body');
    tbody.innerHTML = '';

    if (h.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; padding: 20px; color: var(--text-muted);">No scans in history yet.</td></tr>';
        return;
    }

    h.slice().reverse().forEach(scan => {
        const tr = document.createElement('tr');
        tr.style.borderBottom = '1px solid rgba(255,255,255,0.05)';
        const resColor = scan.result === 'Detected' ? 'var(--danger-color)' : 'var(--success-color)';
        tr.innerHTML = `
                    <td style="padding: 12px 10px;">${scan.date}</td>
                    <td style="padding: 12px 10px; text-transform: capitalize;">${scan.module}</td>
                    <td style="padding: 12px 10px; color: ${resColor};">${scan.result}</td>
                    <td style="padding: 12px 10px;">${scan.severity}</td>
                    <td style="padding: 12px 10px;">${scan.confidence}%</td>
                `;
        tbody.appendChild(tr);
    });
}

function saveHistory(mod, res, sev, conf) {
    const h = JSON.parse(localStorage.getItem('mediscan_history')) || [];
    h.push({
        date: new Date().toLocaleDateString() + ' ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        module: mod,
        result: res,
        severity: sev,
        confidence: conf
    });
    localStorage.setItem('mediscan_history', JSON.stringify(h));
    loadHistory();
}

function clearHistory() {
    if (confirm('Are you sure you want to clear all locally saved scan history?')) {
        localStorage.removeItem('mediscan_history');
        loadHistory();
    }
}

// Download and Share
function downloadReport() {
    const content = `MEDISCAN AI REPORT\nDate: ${new Date().toLocaleString()}\nModule: ${currentTab.toUpperCase()}\nStatus: ${document.getElementById('res-status').innerText}\nSeverity: ${document.getElementById('res-severity').innerText}\nConfidence: ${document.getElementById('res-confidence-text').innerText}\nMetrics: ${document.getElementById('res-specific').innerText}\nRecommendation: ${document.getElementById('res-recommendation').innerText}\n\nDisclaimer: Auto-generated by MediScan AI Prototype. Not a clinical diagnosis.`;

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `MediScan_Report_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

function shareReport() {
    if (navigator.share) {
        navigator.share({
            title: 'MediScan Analysis Report',
            text: `Module: ${currentTab.toUpperCase()} | Status: ${document.getElementById('res-status').innerText} | Conf: ${document.getElementById('res-confidence-text').innerText}`,
        }).catch(err => console.log('Error sharing:', err));
    } else {
        alert('Web Share API implies secure context (HTTPS) and supported browser.');
    }
}

// --- 4. Init --- //
window.onload = async () => {
    loadProfile();
    loadHistory();

    // Disclaimer logic
    if (!localStorage.getItem('mediscan_disclaimer_accepted')) {
        document.getElementById('disclaimer-modal').style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }

    // Initialize TF Lite Model
    await initializeModel();
};

function acceptDisclaimer() {
    localStorage.setItem('mediscan_disclaimer_accepted', 'true');
    document.getElementById('disclaimer-modal').style.display = 'none';
    document.body.style.overflow = 'auto';
}