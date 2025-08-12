document.addEventListener('DOMContentLoaded', () => {
    initializeFileUploads();
    initializeTrainingForm();
    initializePredictionForms();
    initializeDashboard();
});

// Utility Functions
function showLoading() {
    document.getElementById('loading-spinner')?.classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading-spinner')?.classList.add('hidden');
}

function showError(message) {
    const errorMessage = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');
    if (errorMessage && errorText) {
        errorText.textContent = message;
        errorMessage.classList.remove('hidden');
        document.getElementById('success-message')?.classList.add('hidden');
    }
}

function showSuccess() {
    const successMessage = document.getElementById('success-message');
    const errorMessage = document.getElementById('error-message');
    if (successMessage && errorMessage) {
        successMessage.classList.remove('hidden');
        errorMessage.classList.add('hidden');
    }
}

// File Upload Handling
function initializeFileUploads() {
    const uploadForm = document.getElementById('upload-form');
    if (!uploadForm) return;
    
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const filename = document.getElementById('filename');
    const uploadButton = document.getElementById('upload-button');
    const buttonText = document.getElementById('button-text');
    const loadingSpinner = document.getElementById('loading-spinner');
    const uploadProgress = document.getElementById('upload-progress');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    function resetForm() {
        uploadForm.reset();
        fileInfo.classList.add('hidden');
        uploadProgress.classList.add('hidden');
        progressBar.style.width = '0%';
        progressText.textContent = 'Uploading...';
        buttonText.textContent = 'Upload File';
        loadingSpinner.classList.add('hidden');
        uploadButton.disabled = false;
    }
    
    function handleFileSelection(files) {
        if (files.length > 0) {
            const file = files[0];
            const validTypes = ['.csv', '.xlsx'];
            
            if (!validTypes.some(type => file.name.toLowerCase().endsWith(type))) {
                showError('Please upload a CSV or Excel (.xlsx) file');
                return false;
            }

            fileInput.files = files;
            updateFileInfo();
            document.getElementById('error-message')?.classList.add('hidden');
            document.getElementById('success-message')?.classList.add('hidden');
            return true;
        }
        return false;
    }
    
    // Handle drag and drop
    const dropZone = uploadForm.querySelector('.border-dashed');
    if (dropZone) {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('border-blue-500', 'bg-blue-50');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('border-blue-500', 'bg-blue-50');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-blue-500', 'bg-blue-50');
            handleFileSelection(e.dataTransfer.files);
        });
    }
    
    // Handle file selection
    fileInput.addEventListener('change', () => {
        handleFileSelection(fileInput.files);
    });
    
    function updateFileInfo() {
        if (fileInput.files.length > 0) {
            filename.textContent = fileInput.files[0].name;
            fileInfo.classList.remove('hidden');
        }
    }
    
    // Handle form submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            showError('Please select a file to upload');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        
        // Update UI for upload start
        uploadButton.disabled = true;
        buttonText.textContent = 'Uploading...';
        loadingSpinner.classList.remove('hidden');
        uploadProgress.classList.remove('hidden');
        document.getElementById('error-message')?.classList.add('hidden');
        document.getElementById('success-message')?.classList.add('hidden');
        
        try {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/upload', true);
            
            // Handle progress
            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    progressBar.style.width = percentComplete + '%';
                    progressText.textContent = `Uploading... ${Math.round(percentComplete)}%`;
                }
            };
            
            // Handle response
            xhr.onload = () => {
                try {
                    const response = JSON.parse(xhr.responseText);
                    if (xhr.status === 200) {
                        showSuccess();
                        // Store the file data for use in training
                        sessionStorage.setItem('currentFile', response.filename);
                        sessionStorage.setItem('columns', JSON.stringify(response.columns));
                        
                        // Show success message and redirect after a short delay
                        setTimeout(() => {
                            window.location.href = response.nextUrl;
                        }, 1500);
                    } else {
                        const error = response.error || 'Upload failed';
                        showError(error);
                        uploadButton.disabled = false;
                        buttonText.textContent = 'Try Again';
                        loadingSpinner.classList.add('hidden');
                    }
                } catch (e) {
                    showError('Invalid server response');
                    uploadButton.disabled = false;
                    buttonText.textContent = 'Try Again';
                    loadingSpinner.classList.add('hidden');
                }
            };
            
            // Handle network errors
            xhr.onerror = () => {
                showError('Network error occurred. Please try again.');
                uploadButton.disabled = false;
                buttonText.textContent = 'Try Again';
                loadingSpinner.classList.add('hidden');
            };
            
            xhr.send(formData);
        } catch (error) {
            showError('An unexpected error occurred');
            uploadButton.disabled = false;
            buttonText.textContent = 'Try Again';
            loadingSpinner.classList.add('hidden');
        }
    });
}

// Training Form Handling
function initializeTrainingForm() {
    const formContainer = document.getElementById('training-form-container');
    const noFileWarning = document.getElementById('no-file-warning');
    const form = document.getElementById('training-form');
    const fileBadge = document.getElementById('file-badge');
    const currentFile = document.getElementById('current-file');
    
    if (!form || !formContainer || !noFileWarning) return;
    
    // Check if we have a file to work with
    const filename = sessionStorage.getItem('currentFile');
    const columns = JSON.parse(sessionStorage.getItem('columns') || '[]');
    
    if (!filename || !columns.length) {
        formContainer.classList.add('hidden');
        noFileWarning.classList.remove('hidden');
        return;
    }
    
    // Show the current file name
    currentFile.textContent = filename;
    fileBadge.classList.remove('hidden');
    
    const targetSelect = document.getElementById('target-column');
    const featureList = document.getElementById('feature-list');
    const selectAll = document.getElementById('select-all');
    const deselectAll = document.getElementById('deselect-all');
    const trainButton = document.getElementById('train-button');
    const buttonText = document.getElementById('button-text');
    const loadingSpinner = document.getElementById('loading-spinner');
    
    // Populate columns
    columns.forEach(column => {
        // Add to target dropdown
        const option = document.createElement('option');
        option.value = column;
        option.textContent = column;
        targetSelect.appendChild(option);
        
        // Add to feature list
        const div = document.createElement('div');
        div.className = 'flex items-center space-x-2 py-1';
        div.innerHTML = `
            <input type="checkbox" id="feature-${column}" name="features" 
                   value="${column}" class="rounded border-gray-300 text-blue-600 
                   focus:border-blue-500 focus:ring-blue-500">
            <label for="feature-${column}" class="text-gray-700">${column}</label>
        `;
        featureList.appendChild(div);
    });
    
    // Handle select/deselect all
    selectAll.addEventListener('click', () => {
        featureList.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            if (checkbox.value !== targetSelect.value) {
                checkbox.checked = true;
            }
        });
    });
    
    deselectAll.addEventListener('click', () => {
        featureList.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.checked = false;
        });
    });
    
    // Disable selected target from features
    targetSelect.addEventListener('change', () => {
        const selectedTarget = targetSelect.value;
        featureList.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            if (checkbox.value === selectedTarget) {
                checkbox.checked = false;
                checkbox.disabled = true;
            } else {
                checkbox.disabled = false;
            }
        });
    });
    
    // Handle form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const target = targetSelect.value;
        const features = Array.from(featureList.querySelectorAll('input[type="checkbox"]:checked'))
            .map(checkbox => checkbox.value);
        
        if (!target) {
            showError('Please select a target variable');
            return;
        }
        
        if (features.length === 0) {
            showError('Please select at least one feature');
            return;
        }
        
        // Update UI for training start
        trainButton.disabled = true;
        buttonText.textContent = 'Training Models...';
        loadingSpinner.classList.remove('hidden');
        
        try {
            const response = await fetch('/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    target,
                    features,
                    filename
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Store the selected features for later use
                sessionStorage.setItem('target', target);
                sessionStorage.setItem('features', JSON.stringify(features));
                
                // Redirect to dashboard to view training results
                window.location.href = '/dashboard';
            } else {
                showError(data.error || 'Training failed');
                trainButton.disabled = false;
                buttonText.textContent = 'Train Models';
                loadingSpinner.classList.add('hidden');
            }
        } catch (error) {
            showError('An unexpected error occurred');
            trainButton.disabled = false;
            buttonText.textContent = 'Train Models';
            loadingSpinner.classList.add('hidden');
        }
    });
}

// Prediction Forms Handling
function initializePredictionForms() {
    const batchForm = document.getElementById('batch-predict-form');
    const singleForm = document.getElementById('single-predict-form');
    
    if (!batchForm && !singleForm) return;

    // Prediction forms initialization code will be added here
}

// Dashboard Initialization
function initializeDashboard() {
    // Detect dashboard by presence of metrics container
    const metricsEl = document.getElementById('model-metrics');
    if (!metricsEl) return;

    const bestModelEl = document.getElementById('best-model');
    const r2El = document.getElementById('r2-score');
    const rmseEl = document.getElementById('rmse-score');
    const featChartCanvas = document.getElementById('feature-importance-chart');

    // Load model metrics
    fetch('/metrics')
        .then(async (res) => {
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed to load metrics');

            bestModelEl && (bestModelEl.textContent = data.model_type || '-');
            const metrics = data.metrics || {};
            r2El && (r2El.textContent = typeof metrics.test_r2 === 'number' ? metrics.test_r2.toFixed(4) : '-');
            rmseEl && (rmseEl.textContent = typeof metrics.test_rmse === 'number' ? metrics.test_rmse.toFixed(4) : '-');

            // Feature importance chart (if available)
            if (featChartCanvas && data.feature_importance) {
                const entries = Object.entries(data.feature_importance);
                entries.sort((a, b) => b[1] - a[1]);
                const top = entries.slice(0, 15);
                const labels = top.map(([k]) => k);
                const values = top.map(([, v]) => v);

                new Chart(featChartCanvas.getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels,
                        datasets: [{
                            label: 'Importance',
                            data: values,
                            backgroundColor: 'rgba(37, 99, 235, 0.5)',
                            borderColor: 'rgba(37, 99, 235, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            x: { ticks: { autoSkip: false } },
                            y: { beginAtZero: true }
                        }
                    }
                });
            }
        })
        .catch((err) => {
            showError(err.message || 'Failed to load dashboard metrics');
        });

    // Load dataset information
    fetch('/dataset-info')
        .then(async (res) => {
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed to load dataset info');

            const datasetInfo = data.dataset_info;
            
            // Update dataset overview
            const totalRecordsEl = document.getElementById('total-records');
            const totalFeaturesEl = document.getElementById('total-features');
            const targetVariableEl = document.getElementById('target-variable');
            
            if (totalRecordsEl) totalRecordsEl.textContent = datasetInfo.n_rows || '-';
            if (totalFeaturesEl) totalFeaturesEl.textContent = datasetInfo.n_columns || '-';
            if (targetVariableEl) targetVariableEl.textContent = datasetInfo.target_column || '-';

            // Update missing values
            const missingValuesEl = document.getElementById('missing-values');
            if (missingValuesEl && datasetInfo.missing_values) {
                const missingHtml = Object.entries(datasetInfo.missing_values)
                    .filter(([col, count]) => count > 0)
                    .map(([col, count]) => `<p>${col}: <span class="font-medium">${count}</span></p>`)
                    .join('');
                missingValuesEl.innerHTML = missingHtml || '<p>No missing values</p>';
            }

            // Create correlation heatmap
            const correlationCanvas = document.getElementById('correlation-heatmap');
            if (correlationCanvas && datasetInfo.correlations) {
                createCorrelationHeatmap(correlationCanvas, datasetInfo.correlations);
            }

            // Create prediction trend chart (placeholder for now)
            const predictionTrendCanvas = document.getElementById('prediction-trend');
            if (predictionTrendCanvas) {
                createPredictionTrendChart(predictionTrendCanvas);
            }
        })
        .catch((err) => {
            console.warn('Failed to load dataset info:', err.message);
        });
}

// Create correlation heatmap
function createCorrelationHeatmap(canvas, correlations) {
    const ctx = canvas.getContext('2d');
    const labels = Object.keys(correlations);
    
    // Create a matrix of correlation values
    const correlationMatrix = labels.map(label1 => 
        labels.map(label2 => correlations[label1]?.[label2] || 0)
    );

    // Create a custom heatmap using rectangles
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;
    const cellWidth = canvasWidth / labels.length;
    const cellHeight = canvasHeight / labels.length;

    // Clear canvas
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // Draw correlation matrix
    correlationMatrix.forEach((row, i) => {
        row.forEach((value, j) => {
            const x = j * cellWidth;
            const y = i * cellHeight;
            
            // Color based on correlation value
            let color;
            if (value >= 0) {
                // Positive correlation - blue
                const intensity = Math.abs(value);
                color = `rgba(59, 130, 246, ${intensity})`;
            } else {
                // Negative correlation - red
                const intensity = Math.abs(value);
                color = `rgba(239, 68, 68, ${intensity})`;
            }
            
            ctx.fillStyle = color;
            ctx.fillRect(x, y, cellWidth, cellHeight);
            
            // Add border
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 1;
            ctx.strokeRect(x, y, cellWidth, cellHeight);
            
            // Add correlation value text
            ctx.fillStyle = 'white';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(value.toFixed(2), x + cellWidth/2, y + cellHeight/2);
        });
    });

    // Add axis labels
    ctx.fillStyle = 'black';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    
    // X-axis labels
    labels.forEach((label, i) => {
        const x = i * cellWidth + cellWidth / 2;
        const y = canvasHeight + 20;
        ctx.fillText(label, x, y);
    });
    
    // Y-axis labels
    ctx.save();
    ctx.translate(0, canvasHeight);
    ctx.rotate(-Math.PI / 2);
    labels.forEach((label, i) => {
        const x = -(i * cellHeight + cellHeight / 2);
        const y = -20;
        ctx.fillText(label, x, y);
    });
    ctx.restore();
}

// Create prediction trend chart
function createPredictionTrendChart(canvas) {
    const ctx = canvas.getContext('2d');
    
    // This is a placeholder chart - in a real implementation, you would
    // load actual prediction data from the model
    const labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
    const actualData = [120, 190, 300, 500, 200, 300];
    const predictedData = [110, 180, 290, 480, 210, 320];

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Actual',
                    data: actualData,
                    borderColor: 'rgba(59, 130, 246, 1)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Predicted',
                    data: predictedData,
                    borderColor: 'rgba(239, 68, 68, 1)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}
