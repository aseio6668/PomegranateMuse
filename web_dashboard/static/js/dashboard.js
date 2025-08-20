// PomegranteMuse Dashboard JavaScript

class PomegranateDashboard {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.projects = [];
        this.plugins = [];
        this.systemStats = {};
        
        this.initializeComponents();
        this.connectWebSocket();
        this.loadInitialData();
        this.setupEventListeners();
    }
    
    initializeComponents() {
        // Initialize tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
        
        // Show loading states
        this.showLoadingState();
    }
    
    connectWebSocket() {
        try {
            this.socket = io();
            
            this.socket.on('connect', () => {
                console.log('Connected to dashboard server');
                this.isConnected = true;
                this.updateConnectionStatus(true);
            });
            
            this.socket.on('disconnect', () => {
                console.log('Disconnected from dashboard server');
                this.isConnected = false;
                this.updateConnectionStatus(false);
            });
            
            this.socket.on('connection_established', (data) => {
                console.log('Connection established:', data);
                this.sessionId = data.session_id;
            });
            
            this.socket.on('update', (data) => {
                this.handleRealtimeUpdate(data);
            });
            
            this.socket.on('error', (error) => {
                console.error('WebSocket error:', error);
                this.showNotification('error', 'Connection error: ' + error.message);
            });
            
            this.socket.on('server_shutdown', (data) => {
                this.showNotification('warning', 'Server is shutting down');
            });
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.updateConnectionStatus(false);
        }
    }
    
    async loadInitialData() {
        try {
            // Load system info and stats
            await Promise.all([
                this.loadSystemInfo(),
                this.loadSystemStats(),
                this.loadProjects(),
                this.loadPlugins()
            ]);
            
            this.hideLoadingState();
            this.updateDashboard();
            
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showNotification('error', 'Failed to load dashboard data');
        }
    }
    
    async loadSystemInfo() {
        try {
            const response = await fetch('/api/v1/system/info');
            const result = await response.json();
            
            if (result.status === 'success') {
                this.systemInfo = result.data;
                document.getElementById('system-version').textContent = result.data.version;
            }
        } catch (error) {
            console.error('Failed to load system info:', error);
        }
    }
    
    async loadSystemStats() {
        try {
            const response = await fetch('/api/v1/system/stats');
            const result = await response.json();
            
            if (result.status === 'success') {
                this.systemStats = result.data;
                this.updateSystemStats();
            }
        } catch (error) {
            console.error('Failed to load system stats:', error);
        }
    }
    
    async loadProjects() {
        try {
            const response = await fetch('/api/v1/projects');
            const result = await response.json();
            
            if (result.status === 'success') {
                this.projects = result.data;
                this.updateProjectsList();
            }
        } catch (error) {
            console.error('Failed to load projects:', error);
            this.projects = [];
        }
    }
    
    async loadPlugins() {
        try {
            const response = await fetch('/api/v1/plugins');
            const result = await response.json();
            
            if (result.status === 'success') {
                this.plugins = result.data;
                this.updatePluginsCount();
            }
        } catch (error) {
            console.error('Failed to load plugins:', error);
            this.plugins = [];
        }
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.querySelector('.navbar-text');
        const icon = statusElement.querySelector('i');
        const text = statusElement.querySelector('span') || statusElement;
        
        if (connected) {
            icon.className = 'fas fa-circle text-success me-1';
            text.innerHTML = '<i class="fas fa-circle text-success me-1"></i>Connected';
        } else {
            icon.className = 'fas fa-circle text-danger me-1';
            text.innerHTML = '<i class="fas fa-circle text-danger me-1"></i>Disconnected';
        }
    }
    
    updateDashboard() {
        // Update stats cards
        document.getElementById('active-projects').textContent = this.projects.length;
        document.getElementById('loaded-plugins').textContent = this.plugins.length;
        document.getElementById('active-users').textContent = '1'; // Placeholder
        
        this.updateSystemStats();
        this.updateRecentActivity();
    }
    
    updateSystemStats() {
        if (this.systemStats.uptime) {
            document.getElementById('system-uptime').textContent = this.systemStats.uptime;
        }
        if (this.systemStats.memory_usage) {
            document.getElementById('memory-usage').textContent = this.systemStats.memory_usage;
        }
        if (this.systemStats.cpu_usage) {
            document.getElementById('cpu-usage').textContent = this.systemStats.cpu_usage;
        }
    }
    
    updateProjectsList() {
        const tbody = document.querySelector('#recent-projects-table tbody');
        
        if (this.projects.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="5" class="text-center text-muted">
                        <i class="fas fa-folder-open me-2"></i>
                        No projects found
                    </td>
                </tr>
            `;
            return;
        }
        
        tbody.innerHTML = this.projects.slice(0, 5).map(project => `
            <tr>
                <td>
                    <strong>${this.escapeHtml(project.name)}</strong>
                    <br>
                    <small class="text-muted">${this.escapeHtml(project.description || 'No description')}</small>
                </td>
                <td>${this.escapeHtml(project.source_language)}</td>
                <td>${this.escapeHtml(project.target_language)}</td>
                <td>
                    <span class="status-badge ${this.getStatusClass(project.status)}">
                        ${this.escapeHtml(project.status || 'Unknown')}
                    </span>
                </td>
                <td>
                    <button class="btn btn-sm btn-outline-primary" onclick="dashboard.openProject('${project.id}')">
                        <i class="fas fa-external-link-alt"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-secondary" onclick="dashboard.analyzeProject('${project.id}')">
                        <i class="fas fa-search"></i>
                    </button>
                </td>
            </tr>
        `).join('');
    }
    
    updatePluginsCount() {
        const enabledPlugins = this.plugins.filter(p => p.state === 'enabled').length;
        document.getElementById('loaded-plugins').textContent = enabledPlugins;
    }
    
    updateRecentActivity() {
        const activityContainer = document.getElementById('recent-activity');
        
        // This would be populated with actual activity data
        const activities = [
            {
                icon: 'fas fa-plus',
                iconClass: 'bg-primary text-white',
                title: 'Project created',
                description: 'New Python to Pomegranate project',
                time: '5 minutes ago'
            },
            {
                icon: 'fas fa-cog',
                iconClass: 'bg-success text-white',
                title: 'Plugin loaded',
                description: 'Python language plugin activated',
                time: '10 minutes ago'
            }
        ];
        
        if (activities.length === 0) {
            activityContainer.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-history me-2"></i>
                    No recent activity
                </div>
            `;
            return;
        }
        
        activityContainer.innerHTML = activities.map(activity => `
            <div class="activity-item d-flex">
                <div class="activity-icon ${activity.iconClass}">
                    <i class="${activity.icon}"></i>
                </div>
                <div class="activity-content">
                    <div class="fw-bold">${activity.title}</div>
                    <div class="text-muted small">${activity.description}</div>
                    <div class="activity-time">${activity.time}</div>
                </div>
            </div>
        `).join('');
    }
    
    handleRealtimeUpdate(data) {
        console.log('Realtime update:', data);
        
        switch (data.type) {
            case 'project_created':
                this.loadProjects();
                break;
            case 'project_updated':
                this.loadProjects();
                break;
            case 'plugin_loaded':
                this.loadPlugins();
                break;
            case 'system_stats':
                this.systemStats = data.data;
                this.updateSystemStats();
                break;
        }
    }
    
    setupEventListeners() {
        // Refresh data periodically
        setInterval(() => {
            if (this.isConnected) {
                this.loadSystemStats();
            }
        }, 30000); // Every 30 seconds
        
        // Handle form submissions
        document.getElementById('newProjectForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.submitNewProject();
        });
    }
    
    // UI Helper Methods
    showLoadingState() {
        document.body.classList.add('loading');
    }
    
    hideLoadingState() {
        document.body.classList.remove('loading');
    }
    
    showNotification(type, message) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 1rem; right: 1rem; z-index: 1060; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }
    
    getStatusClass(status) {
        const statusMap = {
            'running': 'status-running',
            'completed': 'status-completed',
            'error': 'status-error',
            'pending': 'status-pending'
        };
        return statusMap[status] || 'status-pending';
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // Action Methods
    async createNewProject() {
        const modal = new bootstrap.Modal(document.getElementById('newProjectModal'));
        modal.show();
    }
    
    async submitNewProject() {
        const form = document.getElementById('newProjectForm');
        const formData = new FormData(form);
        
        const projectData = {
            name: document.getElementById('projectName').value,
            description: document.getElementById('projectDescription').value,
            source_language: document.getElementById('sourceLanguage').value,
            target_language: document.getElementById('targetLanguage').value,
            source_path: document.getElementById('sourcePath').value
        };
        
        try {
            const response = await fetch('/api/v1/projects', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(projectData)
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showNotification('success', 'Project created successfully');
                bootstrap.Modal.getInstance(document.getElementById('newProjectModal')).hide();
                form.reset();
                this.loadProjects();
            } else {
                this.showNotification('error', result.error || 'Failed to create project');
            }
            
        } catch (error) {
            console.error('Failed to create project:', error);
            this.showNotification('error', 'Failed to create project');
        }
    }
    
    openProject(projectId) {
        window.location.href = `/project/${projectId}`;
    }
    
    async analyzeProject(projectId) {
        try {
            this.showNotification('info', 'Starting project analysis...');
            
            const response = await fetch(`/api/v1/projects/${projectId}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ analysis_type: 'full' })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showNotification('success', 'Analysis completed');
            } else {
                this.showNotification('error', result.error || 'Analysis failed');
            }
            
        } catch (error) {
            console.error('Failed to analyze project:', error);
            this.showNotification('error', 'Failed to start analysis');
        }
    }
    
    loadPlugins() {
        window.location.href = '/plugins';
    }
    
    viewAnalytics() {
        window.location.href = '/analytics';
    }
    
    openSettings() {
        window.location.href = '/settings';
    }
}

// Global functions for onclick handlers
window.createNewProject = function() {
    dashboard.createNewProject();
};

window.submitNewProject = function() {
    dashboard.submitNewProject();
};

window.loadPlugins = function() {
    dashboard.loadPlugins();
};

window.viewAnalytics = function() {
    dashboard.viewAnalytics();
};

window.openSettings = function() {
    dashboard.openSettings();
};

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.dashboard = new PomegranateDashboard();
});