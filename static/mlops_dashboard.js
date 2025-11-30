// static/mlops_dashboard.js - MLOps Specific Dashboard
let trendsChart = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    refreshAll();
    
    // Auto-refresh every 30 seconds
    setInterval(refreshAll, 30000);
});

// Refresh all data
async function refreshAll() {
    await loadSystemStatus();
    await loadPerformanceMetrics();
    await loadTrends();
    await loadActiveSessions();
    await checkAlerts();
}

// Load system status
async function loadSystemStatus() {
    try {
        const response = await fetch('/mlops/status');
        const status = await response.json();
        
        let statusHtml = '';
        if (status.status === 'error') {
            statusHtml = `<div class="status status-error">
                <strong>Error:</strong> ${status.error || 'Unknown error'}
            </div>`;
        } else {
            statusHtml = `
                <div class="status status-${status.status}">
                    <strong>Overall Status:</strong> ${status.status.toUpperCase()}<br>
                    <strong>Scheduler:</strong> ${status.scheduler_running ? '✅ Running' : '❌ Stopped'}<br>
                    <strong>MLflow:</strong> ${status.mlflow_active ? '✅ Active' : '❌ Inactive'}<br>
                    <strong>Active Sessions:</strong> ${status.metrics?.total_predictions || 0}<br>
                    <strong>Last Update:</strong> ${new Date(status.timestamp).toLocaleString()}
                </div>
            `;
            
            // Show alerts if any
            if (status.alerts && status.alerts.length > 0) {
                statusHtml += `<div class="alert-banner">
                    <strong>⚠️ Active Alerts:</strong> ${status.alerts.length} issue(s) detected
                </div>`;
            }
        }
        
        document.getElementById('system-status').innerHTML = statusHtml;
    } catch (error) {
        console.error('Error loading system status:', error);
        document.getElementById('system-status').innerHTML = 
            '<div class="status status-error">Error loading system status</div>';
    }
}

// Load performance metrics
async function loadPerformanceMetrics() {
    try {
        const response = await fetch('/mlops/metrics');
        const metrics = await response.json();
        
        let metricsHtml = '';
        if (metrics.error) {
            metricsHtml = `<div class="status status-error">${metrics.error}</div>`;
        } else if (Object.keys(metrics).length === 0) {
            metricsHtml = `<div class="status status-degraded">No metrics data available</div>`;
        } else {
            metricsHtml = `
                <div class="metric-card">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value ${getTrendClass(metrics.accuracy, 0.95)}">
                        ${(metrics.accuracy * 100).toFixed(2)}%
                    </div>
                    <div>Threshold: 95%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">FAR</div>
                    <div class="metric-value ${getTrendClass(metrics.far, 0.01, false)}">
                        ${(metrics.far * 100).toFixed(2)}%
                    </div>
                    <div>Threshold: 1%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">FRR</div>
                    <div class="metric-value ${getTrendClass(metrics.frr, 0.05, false)}">
                        ${(metrics.frr * 100).toFixed(2)}%
                    </div>
                    <div>Threshold: 5%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value">
                        ${metrics.f1_score ? metrics.f1_score.toFixed(4) : 'N/A'}
                    </div>
                    <div>Overall Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Predictions</div>
                    <div class="metric-value">
                        ${metrics.total_predictions || 0}
                    </div>
                    <div>Total Count</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Similarity</div>
                    <div class="metric-value">
                        ${metrics.avg_similarity ? metrics.avg_similarity.toFixed(4) : 'N/A'}
                    </div>
                    <div>Score Average</div>
                </div>
            `;
        }
        
        document.getElementById('performance-metrics').innerHTML = metricsHtml;
    } catch (error) {
        console.error('Error loading metrics:', error);
        document.getElementById('performance-metrics').innerHTML = 
            '<div class="status status-error">Error loading metrics</div>';
    }
}

// Load performance trends
async function loadTrends() {
    try {
        const response = await fetch('/mlops/metrics/trends?days=7');
        const trends = await response.json();
        
        if (trends.error || !trends.dates || trends.dates.length === 0) {
            document.getElementById('trends-chart').innerHTML = 
                '<div style="text-align: center; padding: 20px;">No trend data available</div>';
            return;
        }
        
        const ctx = document.getElementById('trends-chart').getContext('2d');
        
        // Destroy existing chart if it exists
        if (trendsChart) {
            trendsChart.destroy();
        }
        
        trendsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: trends.dates.map(date => new Date(date).toLocaleDateString()),
                datasets: [
                    {
                        label: 'Accuracy',
                        data: trends.accuracy,
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Avg Similarity',
                        data: trends.avg_similarity,
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading trends:', error);
        document.getElementById('trends-chart').innerHTML = 
            '<div style="text-align: center; padding: 20px;">Error loading trends</div>';
    }
}

// Load active sessions
async function loadActiveSessions() {
    try {
        // This would need to be implemented in your backend
        // For now, we'll use a placeholder
        const response = await fetch('/mlops/status');
        const status = await response.json();
        
        let sessionsHtml = '';
        if (status.metrics && status.metrics.total_predictions > 0) {
            sessionsHtml = `
                <div class="status status-healthy">
                    <strong>Active Processing:</strong> System is processing faces<br>
                    <strong>Total Predictions:</strong> ${status.metrics.total_predictions}<br>
                    <strong>System Load:</strong> Normal
                </div>
            `;
        } else {
            sessionsHtml = `
                <div class="status status-degraded">
                    No active processing sessions detected
                </div>
            `;
        }
        
        document.getElementById('active-sessions').innerHTML = sessionsHtml;
    } catch (error) {
        console.error('Error loading sessions:', error);
        document.getElementById('active-sessions').innerHTML = 
            '<div class="status status-error">Error loading sessions</div>';
    }
}

// Check for alerts
async function checkAlerts() {
    try {
        const response = await fetch('/mlops/status');
        const status = await response.json();
        
        let alertsHtml = '';
        if (status.alerts && status.alerts.length > 0) {
            alertsHtml = status.alerts.map(alert => 
                `<div class="alert-banner">
                    <strong>⚠️ ${alert.type || 'Alert'}:</strong> ${alert.message || 'Unknown issue'}
                </div>`
            ).join('');
        } else {
            alertsHtml = `
                <div class="status status-healthy">
                    ✅ No active alerts - System operating normally
                </div>
            `;
        }
        
        document.getElementById('alerts-container').innerHTML = alertsHtml;
    } catch (error) {
        console.error('Error checking alerts:', error);
        document.getElementById('alerts-container').innerHTML = 
            '<div class="status status-error">Error loading alerts</div>';
    }
}

// Trigger manual retraining
async function triggerRetraining() {
    try {
        const response = await fetch('/mlops/retrain', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            alert('✅ Retraining completed successfully!\n\n' + 
                  `New threshold: ${result.results.threshold?.toFixed(4) || 'N/A'}\n` +
                  `Persons: ${result.results.num_persons || 'N/A'}\n` +
                  `Samples: ${result.results.num_samples || 'N/A'}`);
        } else if (result.status === 'not_needed') {
            alert('ℹ️ Retraining not needed at this time.\n\n' + result.message);
        } else {
            alert('❌ Retraining failed!\n\n' + result.message);
        }
        
        refreshAll();
    } catch (error) {
        console.error('Error triggering retraining:', error);
        alert('❌ Error triggering retraining: ' + error.message);
    }
}

// Generate performance report
async function generateReport() {
    await loadDetailedReport();
}

// Load detailed report
async function loadDetailedReport() {
    try {
        const response = await fetch('/mlops/report');
        const result = await response.json();
        
        const reportDiv = document.getElementById('detailed-report');
        if (result.report) {
            reportDiv.textContent = result.report;
            reportDiv.style.display = 'block';
        } else if (result.error) {
            reportDiv.textContent = 'Error: ' + result.error;
            reportDiv.style.display = 'block';
        }
    } catch (error) {
        console.error('Error loading report:', error);
        document.getElementById('detailed-report').textContent = 'Error loading report: ' + error.message;
        document.getElementById('detailed-report').style.display = 'block';
    }
}

// Helper function to determine trend class
function getTrendClass(value, threshold, higherIsBetter = true) {
    if (higherIsBetter) {
        return value >= threshold ? 'trend-down' : 'trend-up';
    } else {
        return value <= threshold ? 'trend-down' : 'trend-up';
    }
}