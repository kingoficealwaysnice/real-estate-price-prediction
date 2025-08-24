// Real Estate Price Prediction App - Clean Version

// DOM Elements
const predictionForm = document.getElementById('predictionForm');
const predictionResult = document.getElementById('predictionResult');
const modelStatus = document.getElementById('modelStatus');
const featureImportance = document.getElementById('featureImportance');
const trainModelsBtn = document.getElementById('trainModelsBtn');
const evaluateModelsBtn = document.getElementById('evaluateModelsBtn');

// API Base URL
const API_BASE_URL = '/api';

// Initialize App
document.addEventListener('DOMContentLoaded', function() {
    loadModelStatus();
    loadFeatureImportance();
});

// Load Model Status
async function loadModelStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/models`);
        const data = await response.json();
        displayModelStatus(data);
    } catch (error) {
        console.error('Error loading model status:', error);
        modelStatus.innerHTML = '<p class="text-danger">Error loading model status</p>';
    }
}

// Display Model Status
function displayModelStatus(models) {
    modelStatus.innerHTML = '';
    
    Object.entries(models).forEach(([modelName, status]) => {
        const statusDiv = document.createElement('div');
        statusDiv.className = `model-status-item ${status.loaded ? 'success' : 'error'}`;
        
        statusDiv.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <span>${modelName}</span>
                <span class="badge ${status.loaded ? 'bg-success' : 'bg-danger'}">
                    ${status.loaded ? 'Loaded' : 'Not Loaded'}
                </span>
            </div>
        `;
        
        modelStatus.appendChild(statusDiv);
    });
}

// Load Feature Importance
async function loadFeatureImportance() {
    try {
        const response = await fetch(`${API_BASE_URL}/features/importance`);
        if (response.ok) {
            const data = await response.json();
            displayFeatureImportance(data);
        } else {
            featureImportance.innerHTML = '<p class="text-muted">Feature importance not available. Train models first.</p>';
        }
    } catch (error) {
        console.error('Error loading feature importance:', error);
        featureImportance.innerHTML = '<p class="text-danger">Error loading feature importance</p>';
    }
}

// Display Feature Importance
function displayFeatureImportance(importanceData) {
    featureImportance.innerHTML = '';
    
    // Access the nested feature_importance object
    const features = importanceData.feature_importance || importanceData;
    
    Object.entries(features)
        .sort(([, a], [, b]) => b - a)
        .forEach(([feature, importance]) => {
            const featureDiv = document.createElement('div');
            featureDiv.className = 'feature-item';
            
            featureDiv.innerHTML = `
                <span class="feature-name">${feature}</span>
                <span class="feature-value">${parseFloat(importance).toFixed(3)}</span>
            `;
            
            featureImportance.appendChild(featureDiv);
        });
}

// Handle Prediction Form Submission
predictionForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(predictionForm);
    const data = Object.fromEntries(formData.entries());
    
    // Convert string values to appropriate types
    data.bedrooms = parseInt(data.bedrooms);
    data.area = parseFloat(data.area);
    data.bathrooms = parseInt(data.bathrooms);
    data.age = parseInt(data.age);
    data.parking_spaces = parseInt(data.parking_spaces);
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayPredictionResult(result);
        } else {
            displayError(result.error || 'Prediction failed');
        }
    } catch (error) {
        displayError('Network error: ' + error.message);
    }
});

// Display Prediction Result
function displayPredictionResult(result) {
    predictionResult.style.display = 'block';
    
    const predictedPrice = document.getElementById('predictedPrice');
    const confidence = document.getElementById('confidence');
    const modelUsed = document.getElementById('modelUsed');
    const mae = document.getElementById('mae');
    const rmse = document.getElementById('rmse');
    
    predictedPrice.textContent = `₹${result.predicted_price.toLocaleString()}`;
    confidence.textContent = `${(result.confidence * 100).toFixed(1)}%`;
    modelUsed.textContent = result.model_used || 'ensemble';
    
    // Safely handle missing MAE and RMSE values
    if (mae && typeof result.mae !== 'undefined') {
        mae.textContent = result.mae.toFixed(2);
    } else if (mae) {
        mae.textContent = 'N/A';
    }
    
    if (rmse && typeof result.rmse !== 'undefined') {
        rmse.textContent = result.rmse.toFixed(2);
    } else if (rmse) {
        rmse.textContent = 'N/A';
    }
}

// Display Error
function displayError(message) {
    predictionResult.style.display = 'block';
    predictionResult.innerHTML = `
        <div class="card">
            <div class="card-header bg-danger">
                <h5 class="card-title mb-0">Prediction Error</h5>
            </div>
            <div class="card-body">
                <p class="text-danger">${message}</p>
            </div>
        </div>
    `;
}

// Train Models
trainModelsBtn.addEventListener('click', async function() {
    trainModelsBtn.disabled = true;
    trainModelsBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Training...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/models/train`, {
            method: 'POST',
        });
        
        const result = await response.json();
        
        if (response.ok) {
            alert('Models trained successfully!');
            loadModelStatus();
            loadFeatureImportance();
        } else {
            alert('Error training models: ' + (result.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Network error: ' + error.message);
    } finally {
        trainModelsBtn.disabled = false;
        trainModelsBtn.innerHTML = '<i class="fas fa-robot me-2"></i>Train Models';
    }
});

// Evaluate Models
evaluateModelsBtn.addEventListener('click', async function() {
    evaluateModelsBtn.disabled = true;
    evaluateModelsBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Evaluating...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/models/evaluate`, {
            method: 'POST',
        });
        
        const result = await response.json();
        
        if (response.ok) {
            alert('Models evaluated successfully! Check console for details.');
            console.log('Model Evaluation Results:', result);
        } else {
            alert('Error evaluating models: ' + (result.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Network error: ' + error.message);
    } finally {
        evaluateModelsBtn.disabled = false;
        evaluateModelsBtn.innerHTML = '<i class="fas fa-chart-line me-2"></i>Evaluate Models';
    }
});

// Smooth Scroll for Navigation Links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});