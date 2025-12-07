// ===================================================================
// LedgerX - Complete Application JavaScript
// Features: Authentication, Cloud SQL, Document AI Tracking, Real API Calls
// ===================================================================

// ===================================================================
// GLOBAL STATE
// ===================================================================
const STATE = {
    apiUrl: 'https://ledgerx-api-671429123152.us-central1.run.app',
    token: null,
    username: null,
    invoices: [],
    billing: {
        gcpCredits: { total: 300, used: 10, remaining: 290 },
        documentAI: { freeLimit: 1000, used: 0 }
    }
};

// ===================================================================
// AUTHENTICATION FUNCTIONS
// ===================================================================

async function attemptLogin() {
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

        const res = await fetch(`${STATE.apiUrl}/token`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: formData
        });

        if (!res.ok) {
            throw new Error('Invalid credentials');
        }

        const data = await res.json();
        STATE.token = data.access_token;
        STATE.username = username;

        // Hide login modal
        document.getElementById('loginModal').style.display = 'none';
        
        // Update UI
        document.getElementById('userInfo').textContent = `User: ${username}`;
        document.getElementById('connectionStatus').innerHTML = '<div class="status-dot"></div><span>Connected</span>';
        document.getElementById('connectionStatus').className = 'status-indicator connected';

        // Clear form
        document.getElementById('loginUsername').value = '';
        document.getElementById('loginPassword').value = '';
        errorDiv.style.display = 'none';

        console.log('✅ Login successful');
        
        // Load user's invoices from Cloud SQL
        await loadInvoicesFromDatabase();
        
        // Refresh Document AI usage
        await refreshDocAIUsage();

    } catch (error) {
        errorDiv.textContent = error.message;
        errorDiv.style.display = 'block';
        console.error('Login failed:', error);
    }
}

function logout() {
    STATE.token = null;
    STATE.username = null;
    STATE.invoices = [];
    document.getElementById('loginModal').style.display = 'flex';
    updateAll();
}

function checkAuth() {
    if (!STATE.token) {
        alert('Please login first');
        document.getElementById('loginModal').style.display = 'flex';
        return false;
    }
    return true;
}

// ===================================================================
// LOCAL STORAGE FUNCTIONS (Backup/Offline Support)
// ===================================================================

function saveToLocalStorage() {
    try {
        localStorage.setItem('ledgerx_invoices', JSON.stringify(STATE.invoices));
        localStorage.setItem('ledgerx_username', STATE.username);
        console.log('💾 Saved to localStorage (backup)');
    } catch (e) {
        console.warn('⚠️ localStorage not available:', e);
    }
}

function loadFromLocalStorage() {
    try {
        const saved = localStorage.getItem('ledgerx_invoices');
        if (saved) {
            STATE.invoices = JSON.parse(saved);
            console.log(`📂 Loaded ${STATE.invoices.length} invoices from localStorage (offline backup)`);
            return true;
        }
    } catch (e) {
        console.warn('⚠️ Could not load from localStorage:', e);
    }
    return false;
}

// ===================================================================
// LOCAL STORAGE FUNCTIONS (Backup/Offline Support)
// ===================================================================

function saveToLocalStorage() {
    try {
        localStorage.setItem('ledgerx_invoices', JSON.stringify(STATE.invoices));
        localStorage.setItem('ledgerx_username', STATE.username);
        console.log('💾 Saved to localStorage (backup)');
    } catch (e) {
        console.warn('⚠️ localStorage not available:', e);
    }
}

function loadFromLocalStorage() {
    try {
        const saved = localStorage.getItem('ledgerx_invoices');
        if (saved) {
            STATE.invoices = JSON.parse(saved);
            console.log(`📂 Loaded ${STATE.invoices.length} invoices from localStorage (offline backup)`);
            updateAll();
            return true;
        }
    } catch (e) {
        console.warn('⚠️ Could not load from localStorage:', e);
    }
    return false;
}

// ===================================================================
// CLOUD SQL INTEGRATION FUNCTIONS
// ===================================================================

async function loadInvoicesFromDatabase() {
    try {
        if (!STATE.token) {
            // Try loading from localStorage as fallback
            loadFromLocalStorage();
            return;
        }
        
        console.log('📥 Loading invoices from Cloud SQL...');
        
        const res = await fetch(`${STATE.apiUrl}/user/invoices`, {
            headers: { 'Authorization': `Bearer ${STATE.token}` }
        });
        
        if (res.ok) {
            const data = await res.json();
            STATE.invoices = Array.isArray(data.invoices) ? data.invoices : Object.values(data.invoices || {});
            
            // Save to localStorage as backup
            saveToLocalStorage();
            
            updateAll();
            console.log(`✅ Loaded ${STATE.invoices.length} invoices from Cloud SQL`);
        } else {
            console.warn('⚠️ Failed to load from Cloud SQL:', res.status);
            // Fallback to localStorage
            loadFromLocalStorage();
        }
    } catch (error) {
        console.warn('⚠️ Cloud SQL load failed, using localStorage:', error);
        loadFromLocalStorage();
    }
}

async function saveInvoiceToDatabase(invoice) {
    try {
        if (!STATE.token) return;
        
        const payload = {
            invoice_number: invoice.invoice_number || 'N/A',
            vendor_name: invoice.vendor_name || 'Unknown',
            total_amount: invoice.total_amount || 0,
            currency: invoice.currency || 'USD',
            invoice_date: invoice.invoice_date || new Date().toISOString().split('T')[0],
            quality_prediction: invoice.quality || 'unknown',
            quality_score: invoice.qualityScore || 0,
            risk_prediction: invoice.risk || 'unknown',
            risk_score: invoice.riskScore || 0,
            file_name: invoice.fileName || 'unknown',
            file_type: invoice.fileType || 'IMAGE',
            file_size_kb: (invoice.fileSize || 0) / 1024,
            ocr_confidence: invoice.ocr_confidence || 0,
            ocr_method: invoice.ocrMethod || 'document_ai',
            subtotal: invoice.subtotal || invoice.total_amount || 0,
            tax_amount: invoice.tax_amount || 0,
            discount_amount: invoice.discount_amount || 0
        };
        
        console.log('💾 Saving to Cloud SQL:', payload);
        
        const res = await fetch(`${STATE.apiUrl}/user/invoices/save`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${STATE.token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (res.ok) {
            const result = await res.json();
            console.log('✅ Invoice saved to Cloud SQL:', result);
            
            // Also save to localStorage as backup
            saveToLocalStorage();
        } else {
            const errorText = await res.text();
            console.error('❌ Failed to save to Cloud SQL:', res.status, errorText);
        }
    } catch (err) {
        console.error('❌ Could not save to Cloud SQL:', err);
    }
}

async function deleteInvoiceFromDatabase(invoiceId) {
    try {
        if (!STATE.token) return;
        
        const res = await fetch(`${STATE.apiUrl}/user/invoices/${invoiceId}`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${STATE.token}` }
        });
        
        if (res.ok) {
            console.log('✅ Invoice deleted from Cloud SQL');
        }
    } catch (err) {
        console.warn('⚠️ Could not delete from Cloud SQL:', err);
    }
}

// ===================================================================
// DOCUMENT AI USAGE TRACKING
// ===================================================================

async function fetchDocumentAIUsage() {
    try {
        if (!STATE.token) return 0;
        
        console.log('📊 Fetching Document AI usage...');
        
        const res = await fetch(`${STATE.apiUrl}/admin/document-ai-usage`, {
            headers: { 'Authorization': `Bearer ${STATE.token}` }
        });
        
        if (res.ok) {
            const data = await res.json();
            const usage = data.usage_this_month || 0;
            console.log(`✅ Document AI usage: ${usage} pages this month`);
            return usage;
        } else {
            console.warn('⚠️ Failed to fetch Document AI usage:', res.status);
            return 0;
        }
    } catch (error) {
        console.error('❌ Error fetching Document AI usage:', error);
        return 0;
    }
}

async function refreshDocAIUsage() {
    try {
        const usage = await fetchDocumentAIUsage();
        const remaining = 1000 - usage;
        const percent = (usage / 1000 * 100).toFixed(1);
        
        // Update Settings page
        if (document.getElementById('settingsDocAiUsed')) {
            document.getElementById('settingsDocAiUsed').textContent = usage;
            document.getElementById('settingsDocAiRemaining').textContent = remaining;
            document.getElementById('settingsDocAiPercent').textContent = `${percent}%`;
            document.getElementById('settingsDocAiBar').style.width = `${percent}%`;
            
            // Change color based on usage
            const bar = document.getElementById('settingsDocAiBar');
            if (usage > 900) {
                bar.style.background = 'var(--danger)';
                document.getElementById('settingsDocAiPercent').style.color = 'var(--danger)';
            } else if (usage > 700) {
                bar.style.background = 'var(--warning)';
                document.getElementById('settingsDocAiPercent').style.color = 'var(--warning)';
            } else {
                bar.style.background = 'var(--success)';
                document.getElementById('settingsDocAiPercent').style.color = 'var(--success)';
            }
        }
        
        // Update Billing page
        if (document.getElementById('docAiUsed')) {
            document.getElementById('docAiUsed').textContent = usage;
            document.getElementById('docAiRemaining').textContent = remaining;
            document.getElementById('docAiPercent').textContent = `${percent}%`;
            document.getElementById('docAiProgressBar').style.width = `${percent}%`;
            document.getElementById('docAiTableUsage').textContent = `${usage} / 1,000 free`;
            
            // Update cost if over free tier
            const costPerPage = 0.0015;
            const overageCost = Math.max(0, (usage - 1000) * costPerPage);
            document.getElementById('docAICost').textContent = `$${overageCost.toFixed(2)}`;
        }
        
        console.log(`✅ Document AI updated: ${usage} / 1,000 pages (${percent}%)`);
        
    } catch (error) {
        console.error('❌ Failed to refresh Document AI usage:', error);
        alert('⚠️ Could not fetch usage data. Check console.');
    }
}

// ===================================================================
// BILLING STATS REFRESH
// ===================================================================

async function refreshBillingStats() {
    try {
        if (!checkAuth()) return;
        
        console.log('📊 Refreshing billing statistics...');
        
        // Refresh Document AI usage
        await refreshDocAIUsage();
        
        // Fetch cache statistics
        const cacheRes = await fetch(`${STATE.apiUrl}/admin/cache`, {
            headers: { 'Authorization': `Bearer ${STATE.token}` }
        });
        
        if (cacheRes.ok) {
            const cacheData = await cacheRes.json();
            if (document.getElementById('cacheHitRate')) {
                document.getElementById('cacheHitRate').textContent = cacheData.performance?.hit_rate || '0%';
            }
        }
        
        // Fetch cost statistics  
        const costRes = await fetch(`${STATE.apiUrl}/admin/costs`, {
            headers: { 'Authorization': `Bearer ${STATE.token}` }
        });
        
        if (costRes.ok) {
            const costData = await costRes.json();
            console.log('💰 Cost data:', costData);
        }
        
        alert('✅ Billing stats refreshed!');
        
    } catch (error) {
        console.error('❌ Failed to refresh billing stats:', error);
        alert('⚠️ Could not fetch live stats (requires admin role)');
    }
}

// ===================================================================
// FILE UPLOAD & PROCESSING
// ===================================================================

async function processFile(file) {
    if (!checkAuth()) return;

    const valid = ['.jpg', '.jpeg', '.png', '.pdf'];
    if (!valid.some(ext => file.name.toLowerCase().endsWith(ext))) {
        alert('Invalid file type. Please upload JPG, PNG, or PDF.');
        return;
    }

    const fmt = file.type.startsWith('image/') ? 'IMAGE' : 'PDF';
    const queueDiv = document.getElementById('processingQueue');
    const cardId = `proc-${Date.now()}`;
    
    queueDiv.innerHTML += `
        <div class="card" id="${cardId}" style="text-align: center;">
            <div class="spinner"></div>
            <h3 style="margin-top: 20px;">${file.name}</h3>
            <p style="color: var(--text-muted); margin-top: 8px;">${fmt} • ${(file.size/1024).toFixed(2)} KB</p>
            <p style="color: var(--primary); margin-top: 12px; font-weight: 600;">
                <i class="fas fa-magic"></i> Processing with OCR...
            </p>
        </div>
    `;

    try {
        // REAL API CALL - Upload to backend
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch(`${STATE.apiUrl}/upload/image`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${STATE.token}`
            },
            body: formData
        });

        if (!res.ok) {
            throw new Error(`API Error: ${res.status} ${res.statusText}`);
        }

        const result = await res.json();
        const data = result.extracted_data || {};
        
        // Create invoice object from API response
        const invoice = {
            id: Date.now(),
            fileName: file.name,
            fileType: fmt,
            fileSize: file.size,
            invoice_number: data.invoice_number || 'N/A',
            vendor_name: data.vendor_name || 'Unknown Vendor',
            total_amount: data.total_amount || 0,
            currency: data.currency || 'USD',
            invoice_date: data.invoice_date || new Date().toISOString().split('T')[0],
            blur_score: data.blur_score || 0,
            ocr_confidence: data.ocr_confidence || 0,
            quality: result.quality?.prediction || result.quality?.quality || 'unknown',
            qualityScore: result.quality?.probability || result.quality?.probabilities?.good || 0,
            risk: result.failure?.prediction || result.failure?.risk || 'unknown',
            riskScore: result.failure?.probability || result.failure?.probabilities?.risk || 0,
            timestamp: new Date().toISOString(),
            ocrMethod: result.ocr_method || 'document_ai'
        };

        // Add to state
        STATE.invoices.unshift(invoice);
        
        // Save to Cloud SQL
        await saveInvoiceToDatabase(invoice);

        // Show success message
        document.getElementById(cardId).innerHTML = `
            <i class="fas fa-check-circle" style="font-size: 60px; color: var(--success);"></i>
            <h3 style="margin-top: 16px; color: var(--success);">✅ Processed Successfully!</h3>
            <p style="margin-top: 12px;">${file.name}</p>
            <p style="margin-top: 8px; font-size: 14px; color: var(--text-muted);">
                ${invoice.vendor_name} • ${invoice.currency} ${invoice.total_amount.toFixed(2)}
            </p>
            <div style="margin-top: 16px; display: flex; gap: 12px; justify-content: center;">
                <span class="badge badge-${invoice.quality === 'good' ? 'success' : 'danger'}">
                    Quality: ${invoice.quality}
                </span>
                <span class="badge badge-${invoice.risk === 'safe' || invoice.risk === 'low' ? 'success' : 'warning'}">
                    Risk: ${invoice.risk}
                </span>
            </div>
        `;

        // Remove success card after 3 seconds
        setTimeout(() => {
            const card = document.getElementById(cardId);
            if (card) card.remove();
        }, 3000);
        
        // Update all displays
        updateAll();
        
        // Refresh Document AI usage after processing
        await refreshDocAIUsage();

    } catch (error) {
        console.error('❌ Upload error:', error);
        
        document.getElementById(cardId).innerHTML = `
            <i class="fas fa-times-circle" style="font-size: 60px; color: var(--danger);"></i>
            <h3 style="margin-top: 16px; color: var(--danger);">❌ Processing Failed</h3>
            <p style="margin-top: 8px; color: var(--text-muted);">${error.message}</p>
            <p style="margin-top: 8px; font-size: 12px; color: var(--text-muted);">
                ${error.message.includes('401') ? '🔒 Authentication error - please login again' : 'Check console for details'}
            </p>
        `;
    }
}

// ===================================================================
// UI UPDATE FUNCTIONS
// ===================================================================

function updateAll() {
    // Update KPIs
    document.getElementById('kpiTotalInvoices').textContent = STATE.invoices.length;
    
    if (STATE.invoices.length > 0) {
        const avgQ = (STATE.invoices.reduce((s, i) => s + (i.qualityScore || 0), 0) / STATE.invoices.length * 100).toFixed(1);
        const avgR = (STATE.invoices.reduce((s, i) => s + (i.riskScore || 0), 0) / STATE.invoices.length * 100).toFixed(1);
        document.getElementById('avgQuality').textContent = avgQ + '%';
        document.getElementById('avgRisk').textContent = avgR + '%';
    } else {
        document.getElementById('avgQuality').textContent = '-';
        document.getElementById('avgRisk').textContent = '-';
    }

    // Update today's count
    const today = new Date().toISOString().split('T')[0];
    const todayCount = STATE.invoices.filter(i => i.timestamp && i.timestamp && i.timestamp && i.timestamp.startsWith(today)).length;
    document.getElementById('todayCount').textContent = todayCount;

    // Render tables
    renderHistory();
    renderRecent();
}

function renderHistory() {
    const tbody = document.getElementById('historyTable');
    
    if (STATE.invoices.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="7" class="empty-state">
                    <div class="empty-icon"><i class="fas fa-inbox"></i></div>
                    <div>No invoices in history yet</div>
                    <div style="margin-top: 16px;">
                        <button class="btn btn-primary" onclick="showPage('upload')">
                            <i class="fas fa-upload"></i> Upload Your First Invoice
                        </button>
                    </div>
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = STATE.invoices.map(inv => `
        <tr onclick="showDetails(${inv.id})">
            <td><strong>${inv.fileName}</strong></td>
            <td>${inv.vendor_name}</td>
            <td>${inv.currency} ${inv.total_amount.toFixed(2)}</td>
            <td><span class="badge badge-${inv.quality === 'good' ? 'success' : 'danger'}">${inv.quality}</span></td>
            <td><span class="badge badge-${inv.risk === 'safe' || inv.risk === 'low' ? 'success' : 'warning'}">${inv.risk}</span></td>
            <td>${new Date(inv.timestamp).toLocaleDateString()}</td>
            <td>
                <button class="btn-icon" onclick="event.stopPropagation(); deleteInvoice(${inv.id})">
                    <i class="fas fa-trash"></i>
                </button>
            </td>
        </tr>
    `).join('');
}

function renderRecent() {
    const tbody = document.getElementById('recentActivity');
    const recent = STATE.invoices.slice(0, 5);
    
    if (recent.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="empty-state">
                    <div class="empty-icon"><i class="fas fa-inbox"></i></div>
                    <div>No invoices processed yet</div>
                    <div style="margin-top: 16px;">
                        <button class="btn btn-primary" onclick="showPage('upload')">
                            <i class="fas fa-upload"></i> Upload Invoice
                        </button>
                    </div>
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = recent.map(inv => `
        <tr onclick="showDetails(${inv.id})">
            <td><strong>${inv.invoice_number}</strong></td>
            <td>${inv.vendor_name}</td>
            <td>${inv.currency} ${inv.total_amount.toFixed(2)}</td>
            <td><span class="badge badge-${inv.quality === 'good' ? 'success' : 'danger'}">${inv.quality}</span></td>
            <td><span class="badge badge-${inv.risk === 'safe' || inv.risk === 'low' ? 'success' : 'warning'}">${inv.risk}</span></td>
            <td>${new Date(inv.timestamp).toLocaleTimeString()}</td>
        </tr>
    `).join('');
}

// ===================================================================
// INVOICE DETAILS MODAL
// ===================================================================

function showDetails(id) {
    const inv = STATE.invoices.find(i => i.id === id);
    if (!inv) return;

    const qualityColor = (inv.qualityScore || 0) > 0.8 ? 'var(--success)' : 
                        (inv.qualityScore || 0) > 0.6 ? 'var(--warning)' : 'var(--danger)';
    const riskColor = (inv.riskScore || 0) < 0.3 ? 'var(--success)' : 
                      (inv.riskScore || 0) < 0.7 ? 'var(--warning)' : 'var(--danger)';

    document.getElementById('modalTitle').textContent = `Invoice: ${inv.invoice_number}`;
    document.getElementById('modalBody').innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 24px;">
            <div style="padding: 20px; background: #f8fafc; border-radius: 10px; border-left: 4px solid var(--primary);">
                <div style="font-size: 12px; color: #64748b; margin-bottom: 8px; font-weight: 600; text-transform: uppercase;">Vendor</div>
                <div style="font-size: 18px; font-weight: 700;">${inv.vendor_name}</div>
            </div>
            <div style="padding: 20px; background: #f8fafc; border-radius: 10px; border-left: 4px solid var(--success);">
                <div style="font-size: 12px; color: #64748b; margin-bottom: 8px; font-weight: 600; text-transform: uppercase;">Amount</div>
                <div style="font-size: 18px; font-weight: 700;">${inv.currency} ${inv.total_amount.toFixed(2)}</div>
            </div>
            <div style="padding: 20px; background: #f8fafc; border-radius: 10px; border-left: 4px solid ${qualityColor};">
                <div style="font-size: 12px; color: #64748b; margin-bottom: 8px; font-weight: 600; text-transform: uppercase;">Quality</div>
                <div style="font-size: 18px; font-weight: 700; color: ${qualityColor};">
                    ${inv.quality} (${(inv.qualityScore * 100).toFixed(1)}%)
                </div>
            </div>
            <div style="padding: 20px; background: #f8fafc; border-radius: 10px; border-left: 4px solid ${riskColor};">
                <div style="font-size: 12px; color: #64748b; margin-bottom: 8px; font-weight: 600; text-transform: uppercase;">Risk</div>
                <div style="font-size: 18px; font-weight: 700; color: ${riskColor};">
                    ${inv.risk} (${(inv.riskScore * 100).toFixed(1)}%)
                </div>
            </div>
            <div style="padding: 20px; background: #f8fafc; border-radius: 10px;">
                <div style="font-size: 12px; color: #64748b; margin-bottom: 8px; font-weight: 600; text-transform: uppercase;">Date</div>
                <div style="font-size: 18px; font-weight: 700;">${inv.invoice_date}</div>
            </div>
            <div style="padding: 20px; background: #f8fafc; border-radius: 10px;">
                <div style="font-size: 12px; color: #64748b; margin-bottom: 8px; font-weight: 600; text-transform: uppercase;">OCR Confidence</div>
                <div style="font-size: 18px; font-weight: 700;">${(inv.ocr_confidence * 100).toFixed(1)}%</div>
            </div>
        </div>
        
        <div style="margin-top: 24px; padding: 20px; background: var(--bg-main); border-radius: 12px;">
            <h4 style="margin-bottom: 12px; font-weight: 700;">📊 Processing Details</h4>
            <div style="font-size: 14px; color: var(--text-muted); line-height: 1.8;">
                <div>• OCR Method: ${inv.ocrMethod || 'Document AI'}</div>
                <div>• File Type: ${inv.fileType}</div>
                <div>• File Size: ${(inv.fileSize / 1024).toFixed(2)} KB</div>
                <div>• Processed: ${new Date(inv.timestamp).toLocaleString()}</div>
            </div>
        </div>
        
        <div style="margin-top: 24px; display: flex; gap: 12px; justify-content: flex-end;">
            <button class="btn btn-secondary" onclick="closeModal()">
                <i class="fas fa-times"></i> Close
            </button>
            <button class="btn btn-primary" onclick="exportInvoice(${inv.id})">
                <i class="fas fa-download"></i> Export
            </button>
        </div>
    `;

    document.getElementById('invoiceModal').classList.add('active');
}

function closeModal() {
    document.getElementById('invoiceModal').classList.remove('active');
}

function deleteInvoice(id) {
    if (!confirm('Delete this invoice from history?')) return;
    
    // Delete from Cloud SQL
    deleteInvoiceFromDatabase(id);
    
    // Delete from local state
    STATE.invoices = STATE.invoices.filter(i => i.id !== id);
    updateAll();
}

function exportInvoice(id) {
    const inv = STATE.invoices.find(i => i.id === id);
    if (!inv) return;
    
    const json = JSON.stringify(inv, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${inv.invoice_number}_report.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// ===================================================================
// NAVIGATION
// ===================================================================

function showPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    
    document.getElementById(pageId).classList.add('active');
    const navItem = document.querySelector(`.nav-item[data-page="${pageId}"]`);
    if (navItem) navItem.classList.add('active');
    
    // Refresh data when switching to certain pages
    if (pageId === 'billing' || pageId === 'settings') {
        refreshDocAIUsage();
    }
}

// Setup navigation click handlers
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const page = item.getAttribute('data-page');
            showPage(page);
        });
    });
});

// ===================================================================
// DRAG & DROP SETUP
// ===================================================================

function setupUpload() {
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('fileInput');

    if (!zone || !input) return;

    zone.onclick = () => input.click();
    
    input.onchange = (e) => {
        Array.from(e.target.files).forEach(f => processFile(f));
        input.value = '';
    };

    zone.ondragover = (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    };
    
    zone.ondragleave = () => {
        zone.classList.remove('dragover');
    };
    
    zone.ondrop = (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        Array.from(e.dataTransfer.files).forEach(f => processFile(f));
    };
}

// ===================================================================
// SETTINGS FUNCTIONS
// ===================================================================

function saveSettings() {
    const newUrl = document.getElementById('apiUrlSetting').value;
    if (newUrl) {
        STATE.apiUrl = newUrl;
        alert('✅ Settings saved! Reconnect to apply changes.');
    }
}

function exportAllData() {
    if (STATE.invoices.length === 0) {
        alert('No data to export');
        return;
    }
    
    const json = JSON.stringify(STATE.invoices, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ledgerx_export_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
    alert('✅ Data exported!');
}

function clearAllData() {
    if (!confirm('⚠️ Delete all invoice history? This cannot be undone.')) return;
    
    STATE.invoices = [];
    updateAll();
    alert('✅ History cleared!');
}

function exportHistory() {
    if (STATE.invoices.length === 0) {
        alert('No data to export');
        return;
    }
    
    const csv = [
        ['File', 'Vendor', 'Amount', 'Quality', 'Risk', 'Date'].join(','),
        ...STATE.invoices.map(i => [
            i.fileName,
            i.vendor_name,
            i.total_amount,
            i.quality,
            i.risk,
            i.invoice_date
        ].join(','))
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ledgerx_history_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}

// ===================================================================
// INITIALIZATION
// ===================================================================

window.onload = async () => {
    console.log('🚀 LedgerX Application Starting...');
    
    // Setup drag & drop
    setupUpload();
    
    // Show login modal
    document.getElementById('loginModal').style.display = 'flex';
    
    console.log('✅ Application initialized');
};
