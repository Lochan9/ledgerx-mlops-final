// =====================================================================
// LedgerX Application JavaScript - Final Working Version
// =====================================================================

const API = 'https://ledgerx-api-671429123152.us-central1.run.app';
let token = null;

console.log('🚀 LedgerX v3.1 Final');
console.log('🎯 Init');

// =====================================================================
// HELPER: API CALL
// =====================================================================
async function api(endpoint, options = {}) {
    const headers = {};
    if (token && !options.skipAuth) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    if (options.body && typeof options.body === 'string') {
        headers['Content-Type'] = 'application/json';
    }
    
    try {
        const response = await fetch(`${API}${endpoint}`, {
            ...options,
            headers: headers
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('json')) {
            return await response.json();
        }
        return await response.text();
        
    } catch (error) {
        console.error('❌ API Error:', error);
        throw error;
    }
}

// =====================================================================
// HEALTH CHECK
// =====================================================================
async function checkHealth() {
    try {
        const health = await api('/health', { skipAuth: true });
        console.log('✅ Health:', health);
        document.getElementById('connectionStatus').className = 'status-indicator connected';
        document.getElementById('connectionStatus').innerHTML = '<div class="status-dot"></div><span>Connected</span>';
    } catch (error) {
        console.error('❌ Health check failed');
    }
}

// =====================================================================
// AUTHENTICATION
// =====================================================================
async function login() {
    const username = document.getElementById('loginUsername').value.trim();
    const password = document.getElementById('loginPassword').value;
    const errorDiv = document.getElementById('loginError');
    
    if (!username || !password) {
        errorDiv.textContent = 'Please enter username and password';
        errorDiv.style.display = 'block';
        return;
    }
    
    try {
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);
        
        const response = await fetch(`${API}/token`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Invalid credentials');
        }
        
        const data = await response.json();
        token = data.access_token;
        
        console.log('✅ Login OK');
        
        // Update UI
        document.getElementById('loginModal').classList.remove('active');
        document.getElementById('userInfo').textContent = `User: ${username}`;
        document.getElementById('connectionStatus').className = 'status-indicator connected';
        document.getElementById('connectionStatus').innerHTML = '<div class="status-dot"></div><span>Connected</span>';
        
        // Clear form
        errorDiv.style.display = 'none';
        
        // Load data
        await loadInvoices();
        await loadUsage();
        
    } catch (error) {
        console.error('❌ Login failed');
        errorDiv.textContent = error.message;
        errorDiv.style.display = 'block';
    }
}

// =====================================================================
// LOAD INVOICES FROM DATABASE
// =====================================================================
async function loadInvoices() {
    console.log('📥 Loading invoices');
    if (!token) return;
    
    try {
        const data = await api('/user/invoices');
        
        const invoiceTable = document.getElementById('invoiceTable');
        const historyTable = document.getElementById('historyTable');
        
        if (!data.invoices || !data.invoices.length) {
            const empty = '<tr><td colspan="6" class="empty-state"><div class="empty-icon"><i class="fas fa-inbox"></i></div><p>No invoices yet. Upload one to get started!</p></td></tr>';
            if (invoiceTable) invoiceTable.innerHTML = empty;
            if (historyTable) historyTable.innerHTML = empty;
            
            // Update stats
            ['kpiTotal', 'statTotal', 'statApproved', 'statPending', 'statRisk'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.textContent = '0';
            });
            return;
        }
        
        // Render table rows
        const rows = data.invoices.map(inv => {
            const qualityClass = inv.quality_prediction === 'good' ? 'success' : 'danger';
            const riskClass = (inv.risk_prediction === 'safe' || inv.risk_prediction === 'low') ? 'success' : 'warning';
            
            // Escape single quotes for onclick
            const invStr = JSON.stringify(inv).replace(/'/g, "\\'");
            
            return `
                <tr onclick='showModal(${invStr})'>
                    <td><strong>${inv.invoice_number || inv.id || 'N/A'}</strong></td>
                    <td>${inv.vendor_name || 'Unknown'}</td>
                    <td>$${(inv.total_amount || 0).toFixed(2)}</td>
                    <td><span class="badge badge-${qualityClass}">${(inv.quality_prediction || 'N/A').toUpperCase()}</span></td>
                    <td><span class="badge badge-${riskClass}">${(inv.risk_prediction || 'N/A').toUpperCase()}</span></td>
                    <td>${inv.created_at ? new Date(inv.created_at).toLocaleDateString() : 'N/A'}</td>
                </tr>
            `;
        }).join('');
        
        if (invoiceTable) invoiceTable.innerHTML = rows;
        if (historyTable) historyTable.innerHTML = rows;
        
        // Update stats
        const total = data.invoices.length;
        const approved = data.invoices.filter(i => i.quality_prediction === 'good' && (i.risk_prediction === 'safe' || i.risk_prediction === 'low')).length;
        const review = data.invoices.filter(i => i.quality_prediction === 'good' && i.risk_prediction !== 'safe' && i.risk_prediction !== 'low').length;
        const highRisk = data.invoices.filter(i => i.quality_prediction !== 'good' || i.risk_prediction === 'high').length;
        
        ['kpiTotal', 'statTotal'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = total;
        });
        
        const approvedEl = document.getElementById('statApproved');
        if (approvedEl) approvedEl.textContent = approved;
        
        const pendingEl = document.getElementById('statPending');
        if (pendingEl) pendingEl.textContent = review;
        
        const riskEl = document.getElementById('statRisk');
        if (riskEl) riskEl.textContent = highRisk;
        
        console.log(`✅ Loaded ${total} invoices`);
        
    } catch (error) {
        console.error('❌ Load invoices failed:', error);
    }
}

// =====================================================================
// LOAD USAGE STATS
// =====================================================================
async function loadUsage() {
    console.log('📊 Usage');
    if (!token) return;
    
    try {
        const usage = await api('/admin/document-ai-usage');
        const pages = usage.usage_this_month || 0;
        const cost = usage.cost_this_month || 0;
        const percent = (pages / 1000 * 100).toFixed(1);
        
        ['usagePages', 'kpiDocAI'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = pages;
        });
        
        const costEl = document.getElementById('usageCost');
        if (costEl) costEl.textContent = `$${cost.toFixed(2)}`;
        
        const percentEl = document.getElementById('usagePercent');
        if (percentEl) percentEl.textContent = `${percent}%`;
        
        const bar = document.getElementById('usageBar');
        if (bar) bar.style.width = `${Math.min(100, percent)}%`;
        
        const remaining = document.getElementById('usageRemaining');
        if (remaining) remaining.textContent = Math.max(0, 1000 - pages);
        
        console.log(`✅ Usage: ${pages}/1000 pages`);
        
    } catch (error) {
        console.warn('⚠️ Usage failed (admin required)');
    }
}

// =====================================================================
// FILE UPLOAD
// =====================================================================
const zone = document.getElementById('uploadZone');
const input = document.getElementById('fileInput');

if (zone && input) {
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        zone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
    });
    
    // Highlight drop zone
    ['dragenter', 'dragover'].forEach(eventName => {
        zone.addEventListener(eventName, () => {
            zone.classList.add('dragover');
        });
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        zone.addEventListener(eventName, () => {
            zone.classList.remove('dragover');
        });
    });
    
    // Handle drop
    zone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        upload(files);
    });
    
    // Handle file select
    input.addEventListener('change', (e) => {
        const files = e.target.files;
        upload(files);
    });
}

async function upload(files) {
    if (!files || !files.length) return;
    if (!token) {
        alert('Please login first');
        return;
    }
    
    const file = files[0];
    const resultDiv = document.getElementById('uploadResult');
    const contentDiv = document.getElementById('resultContent');
    
    if (!resultDiv || !contentDiv) return;
    
    resultDiv.style.display = 'block';
    contentDiv.innerHTML = '<div style="text-align:center;padding:40px"><div class="spinner"></div><p style="margin-top:20px;color:var(--text-muted)">Processing with Document AI...</p></div>';
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API}/upload/image`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            },
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Upload failed: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('✅ Upload OK:', result);
        
        // Parse response - handle both formats
        const quality = result.result?.quality_assessment || result.quality || {};
        const failure = result.result?.failure_risk || result.failure || {};
        
        const qualityPred = quality.quality || quality.prediction || 'N/A';
        const qualityConf = quality.confidence || 0;
        const failurePred = failure.risk || failure.prediction || 'N/A';
        const failureProb = failure.probability || failure.confidence || 0;
        
        const qc = qualityPred === 'good' ? 'success' : 'danger';
        const fc = failurePred === 'safe' || failurePred === 'low' ? 'success' : 'warning';
        
        contentDiv.innerHTML = `
            <div class="alert alert-success">
                <i class="fas fa-check-circle"></i>
                <div><strong>Success!</strong><br>${file.name} processed</div>
            </div>
            <div class="detail-grid">
                <div class="detail-item">
                    <div class="detail-label">QUALITY</div>
                    <div class="detail-value"><span class="badge badge-${qc}">${qualityPred.toUpperCase()}</span></div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">CONFIDENCE</div>
                    <div class="detail-value">${(qualityConf * 100).toFixed(1)}%</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">RISK</div>
                    <div class="detail-value"><span class="badge badge-${fc}">${failurePred.toUpperCase()}</span></div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">RISK PROBABILITY</div>
                    <div class="detail-value">${(failureProb * 100).toFixed(1)}%</div>
                </div>
            </div>
            <div style="text-align:right;margin-top:20px">
                <button class="btn btn-secondary" onclick="document.getElementById('uploadResult').style.display='none'">
                    <i class="fas fa-times"></i> Close
                </button>
                <button class="btn btn-primary" onclick="showPage('history')">
                    <i class="fas fa-history"></i> View History
                </button>
            </div>
        `;
        
        // Reload invoices after 1 second
        setTimeout(() => {
            loadInvoices();
            loadUsage();
        }, 1000);
        
    } catch (error) {
        console.error('❌ Upload failed:', error);
        contentDiv.innerHTML = `
            <div class="alert alert-error">
                <i class="fas fa-times-circle"></i>
                <div><strong>Upload Failed</strong><br>${error.message}</div>
            </div>
            <button class="btn btn-secondary" onclick="document.getElementById('uploadResult').style.display='none'" style="margin-top:16px">
                <i class="fas fa-times"></i> Close
            </button>
        `;
    }
}

// =====================================================================
// SHOW INVOICE MODAL
// =====================================================================
function showModal(invoiceData) {
    const inv = typeof invoiceData === 'string' ? JSON.parse(invoiceData) : invoiceData;
    
    const qualityScore = parseFloat(inv.quality_score) || 0;
    const riskScore = parseFloat(inv.risk_score) || 0;
    const ocrConf = parseFloat(inv.ocr_confidence) || 0;
    
    const qualityPercent = (qualityScore * 100).toFixed(1);
    const riskPercent = (riskScore * 100).toFixed(1);
    const ocrPercent = (ocrConf * 100).toFixed(1);
    
    document.getElementById('modalTitle').textContent = `Invoice: ${inv.invoice_number || inv.id}`;
    document.getElementById('modalScore').textContent = qualityPercent;
    document.getElementById('modalScore').style.color = qualityScore > 0.7 ? 'var(--success)' : qualityScore > 0.4 ? 'var(--warning)' : 'var(--danger)';
    document.getElementById('modalSubtitle').textContent = qualityScore > 0.7 ? 'High quality' : qualityScore > 0.4 ? 'Medium quality' : 'Low quality';
    document.getElementById('modalVendor').textContent = inv.vendor_name || 'Unknown';
    document.getElementById('modalAmount').textContent = `$${(inv.total_amount || 0).toFixed(2)}`;
    document.getElementById('modalDate').textContent = inv.invoice_date || inv.created_at?.split('T')[0] || 'N/A';
    document.getElementById('modalQuality').textContent = (inv.quality_prediction || 'N/A').toUpperCase();
    document.getElementById('modalRisk').textContent = (inv.risk_prediction || 'N/A').toUpperCase();
    document.getElementById('modalConf').textContent = `${ocrPercent}%`;
    
    document.getElementById('invoiceModal').classList.add('active');
}

function closeModal() {
    document.getElementById('invoiceModal').classList.remove('active');
}

// =====================================================================
// NAVIGATION
// =====================================================================
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    
    const page = document.getElementById(pageId);
    if (page) page.classList.add('active');
    
    const navItem = document.querySelector(`.nav-item[data-page="${pageId}"]`);
    if (navItem) navItem.classList.add('active');
    
    // Refresh data when switching pages
    if (pageId === 'history') {
        loadInvoices();
    }
    if (pageId === 'usage') {
        loadUsage();
    }
}

// =====================================================================
// INITIALIZATION
// =====================================================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎯 DOM Ready');
    
    // Setup navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const page = item.getAttribute('data-page');
            showPage(page);
        });
    });
    
    // Check health
    checkHealth();
    
    console.log('✅ Ready');
});