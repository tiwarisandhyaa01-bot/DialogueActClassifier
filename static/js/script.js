// ============================================
// DIALOGUE ACT CLASSIFIER - JAVASCRIPT
// Premium Interactive Features
// ============================================

// ===== GLOBAL STATE =====
let modelInfo = null;
let examples = null;

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Dialogue Act Classifier Initialized');
    
    // Initialize theme
    initializeTheme();
    
    // Load model info
    loadModelInfo();
    
    // Load examples
    loadExamples();
    
    // Setup event listeners
    setupEventListeners();
    
    // Add entrance animations
    addEntranceAnimations();
});

// ===== THEME MANAGEMENT =====
function initializeTheme() {
    const themeToggle = document.getElementById('themeToggle');
    const savedTheme = localStorage.getItem('theme') || 'dark';
    
    if (savedTheme === 'light') {
        document.body.classList.remove('dark-mode');
    }
    
    themeToggle.addEventListener('click', toggleTheme);
}

function toggleTheme() {
    const body = document.body;
    body.classList.toggle('dark-mode');
    
    const isDark = body.classList.contains('dark-mode');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    
    // Add rotation animation
    const themeToggle = document.getElementById('themeToggle');
    themeToggle.style.transform = 'rotate(360deg)';
    setTimeout(() => {
        themeToggle.style.transform = 'rotate(0deg)';
    }, 300);
}

// ===== EVENT LISTENERS SETUP =====
function setupEventListeners() {
    // Tab switching
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.addEventListener('click', () => switchTab(button.dataset.tab));
    });
    
    // Single prediction
    const predictBtn = document.getElementById('predictBtn');
    predictBtn.addEventListener('click', handleSinglePrediction);
    
    // Enter key in textarea
    const textInput = document.getElementById('textInput');
    textInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            handleSinglePrediction();
        }
    });
    
    // File upload
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleFileDrop);
    fileInput.addEventListener('change', handleFileSelect);
}

// ===== TAB SWITCHING =====
function switchTab(tabName) {
    // Update tab buttons
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    
    // Update tab content
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => {
        content.classList.remove('active');
    });
    
    const activeTab = document.getElementById(tabName + 'Tab');
    if (activeTab) {
        activeTab.classList.add('active');
    }
    
    // Load specific content if needed
    if (tabName === 'model' && !modelInfo) {
        loadModelInfo();
    } else if (tabName === 'examples' && !examples) {
        loadExamples();
    }
}

// ===== LOAD MODEL INFO =====
async function loadModelInfo() {
    try {
        const response = await fetch('/api/model-info');
        const data = await response.json();
        modelInfo = data;
        
        // Update header stats
        document.getElementById('modelAccuracy').textContent = data.overall_accuracy + '%';
        document.getElementById('numClasses').textContent = data.num_classes;
        
        // Display model info in tab
        displayModelInfo(data);
    } catch (error) {
        console.error('Error loading model info:', error);
        showNotification('Failed to load model information', 'error');
    }
}

function displayModelInfo(data) {
    const container = document.getElementById('modelInfoContent');
    
    const html = `
        <div class="model-stat-grid">
            <div class="model-stat-card">
                <span class="model-stat-value">${data.overall_accuracy}%</span>
                <span class="model-stat-label">Overall Accuracy</span>
            </div>
            <div class="model-stat-card">
                <span class="model-stat-value">${data.num_classes}</span>
                <span class="model-stat-label">Dialogue Acts</span>
            </div>
            <div class="model-stat-card">
                <span class="model-stat-value">${data.model_type}</span>
                <span class="model-stat-label">Model Type</span>
            </div>
            <div class="model-stat-card">
                <span class="model-stat-value">${data.training_date.split(' ')[0]}</span>
                <span class="model-stat-label">Trained On</span>
            </div>
        </div>
        
        <h3 style="margin-bottom: 16px; color: var(--text-primary); font-size: 18px;">Class-wise Performance Metrics</h3>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Dialogue Act</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                </tr>
            </thead>
            <tbody>
                ${data.class_metrics.map(metric => `
                    <tr>
                        <td>${metric.label}</td>
                        <td>${metric.precision}%</td>
                        <td>${metric.recall}%</td>
                        <td>${metric.f1_score}%</td>
                        <td>${metric.support}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    
    container.innerHTML = html;
}

// ===== LOAD EXAMPLES =====
async function loadExamples() {
    try {
        const response = await fetch('/api/examples');
        const data = await response.json();
        examples = data;
        displayExamples(data);
    } catch (error) {
        console.error('Error loading examples:', error);
        showNotification('Failed to load examples', 'error');
    }
}

function displayExamples(data) {
    const container = document.getElementById('examplesList');
    
    let html = '';
    let delay = 0;
    
    Object.keys(data).forEach(category => {
        data[category].forEach((text, index) => {
            html += `
                <div class="example-card" onclick="useExample('${escapeHtml(text)}')" style="animation-delay: ${delay}s">
                    <span class="example-category">${category}</span>
                    <p class="example-text">${text}</p>
                </div>
            `;
            delay += 0.05;
        });
    });
    
    container.innerHTML = html;
}

function useExample(text) {
    // Switch to single prediction tab
    switchTab('single');
    
    // Fill the textarea
    const textInput = document.getElementById('textInput');
    textInput.value = text;
    
    // Scroll to textarea
    textInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    // Focus and add highlight effect
    setTimeout(() => {
        textInput.focus();
        textInput.style.boxShadow = '0 0 0 4px rgba(99, 102, 241, 0.3)';
        setTimeout(() => {
            textInput.style.boxShadow = '';
        }, 1000);
    }, 500);
}

// ===== SINGLE PREDICTION =====
async function handleSinglePrediction() {
    const textInput = document.getElementById('textInput');
    const text = textInput.value.trim();
    
    if (!text) {
        showNotification('Please enter some text to classify', 'warning');
        textInput.focus();
        return;
    }
    
    // Show loading
    showLoading();
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayPredictionResult(data);
        } else {
            showNotification(data.error || 'Prediction failed', 'error');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification('Failed to get prediction. Please try again.', 'error');
    } finally {
        hideLoading();
    }
}

function displayPredictionResult(data) {
    const resultsSection = document.getElementById('resultsSection');
    const resultBadge = document.getElementById('resultBadge');
    const predictedLabel = document.getElementById('predictedLabel');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    const probabilitiesList = document.getElementById('probabilitiesList');
    
    // Update badge
    resultBadge.textContent = 'SUCCESS';
    
    // Update predicted label
    predictedLabel.textContent = data.predicted_label;
    
    // Update confidence bar with animation
    setTimeout(() => {
        confidenceFill.style.width = data.confidence + '%';
        confidenceText.textContent = data.confidence + '%';
    }, 100);
    
    // Update probabilities list
    let probabilitiesHTML = '';
    data.sorted_probabilities.forEach((item, index) => {
        probabilitiesHTML += `
            <div class="probability-item" style="animation-delay: ${index * 0.05}s">
                <span class="probability-label">${item.label}</span>
                <span class="probability-value">${item.probability}%</span>
            </div>
        `;
    });
    probabilitiesList.innerHTML = probabilitiesHTML;
    
    // Show results with animation
    resultsSection.style.display = 'block';
    
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 300);
    
    showNotification('Classification completed successfully!', 'success');
}

// ===== BATCH PROCESSING =====
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');
}

function handleFileDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processBatchFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processBatchFile(files[0]);
    }
}

async function processBatchFile(file) {
    if (!file.name.endsWith('.csv')) {
        showNotification('Please upload a CSV file', 'warning');
        return;
    }
    
    showLoading();
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/predict-batch', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayBatchResults(data);
            showNotification(`Successfully processed ${data.total_predictions} messages`, 'success');
        } else {
            showNotification(data.error || 'Batch processing failed', 'error');
        }
    } catch (error) {
        console.error('Batch processing error:', error);
        showNotification('Failed to process file. Please try again.', 'error');
    } finally {
        hideLoading();
        // Reset file input
        document.getElementById('fileInput').value = '';
    }
}

function displayBatchResults(data) {
    const batchResults = document.getElementById('batchResults');
    const batchCount = document.getElementById('batchCount');
    const batchResultsList = document.getElementById('batchResultsList');
    
    // Update count
    batchCount.textContent = `${data.total_predictions} predictions`;
    
    // Display results
    let html = '';
    data.results.forEach((result, index) => {
        html += `
            <div class="batch-result-item" style="animation-delay: ${index * 0.02}s">
                <div class="batch-result-text">${escapeHtml(result.text)}</div>
                <div class="batch-result-prediction">
                    <span class="batch-result-label">${result.predicted_label}</span>
                    <span class="batch-result-confidence">${result.confidence}% confidence</span>
                </div>
            </div>
        `;
    });
    batchResultsList.innerHTML = html;
    
    // Show results
    batchResults.style.display = 'block';
    
    // Scroll to results
    setTimeout(() => {
        batchResults.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 300);
}

// ===== UTILITY FUNCTIONS =====
function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 24px;
        padding: 16px 24px;
        background: ${type === 'success' ? 'var(--success)' : type === 'error' ? 'var(--error)' : type === 'warning' ? 'var(--warning)' : 'var(--accent-primary)'};
        color: white;
        border-radius: 12px;
        box-shadow: var(--shadow-lg);
        font-weight: 600;
        font-size: 14px;
        z-index: 10000;
        animation: slideInRight 0.3s ease;
        max-width: 400px;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function addEntranceAnimations() {
    // Add stagger animation to initial elements
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
    });
}

// ===== ADDITIONAL ANIMATIONS (CSS IN JS) =====
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideOutRight {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100px);
        }
    }
`;
document.head.appendChild(style);

// ===== EASTER EGG =====
let clickCount = 0;
document.querySelector('.logo-icon').addEventListener('click', function() {
    clickCount++;
    if (clickCount === 5) {
        showNotification('🎉 Made with ❤️ for an impressive college project!', 'success');
        clickCount = 0;
    }
});

console.log('%c🚀 Dialogue Act Classifier ', 'background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white; padding: 10px 20px; font-size: 16px; font-weight: bold; border-radius: 8px;');
console.log('%cBuilt with DistilBERT + Flask + Premium UI', 'color: #818cf8; font-size: 12px;');