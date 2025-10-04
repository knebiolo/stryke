/**
 * Auto-Save System for Stryke Web App
 * Automatically saves form data to localStorage every 3 minutes
 * Detects and restores previous sessions on page load
 */

(function() {
    'use strict';

    const AUTO_SAVE_INTERVAL = 180000; // 3 minutes in milliseconds
    const STORAGE_KEY = 'stryke_autosave';
    const TIMESTAMP_KEY = 'stryke_autosave_timestamp';
    let autoSaveTimer = null;
    let lastSaveTime = null;

    // Initialize auto-save system
    function init() {
        // Check for existing auto-saved data on page load
        checkForAutoSavedData();
        
        // Start auto-save timer
        startAutoSave();
        
        // Update "last saved" indicator every 30 seconds
        setInterval(updateLastSavedIndicator, 30000);
        
        // Save on page unload as backup
        window.addEventListener('beforeunload', saveFormData);
    }

    // Check if there's auto-saved data and prompt user to restore
    function checkForAutoSavedData() {
        const savedData = localStorage.getItem(STORAGE_KEY);
        const savedTimestamp = localStorage.getItem(TIMESTAMP_KEY);
        
        if (savedData && savedTimestamp) {
            // Check if saved data has meaningful content (not just empty form)
            try {
                const parsedData = JSON.parse(savedData);
                const hasContent = Object.keys(parsedData).length > 0 && 
                                  Object.values(parsedData).some(val => val && val.toString().trim() !== '');
                
                if (!hasContent) {
                    // No meaningful data, clear it
                    clearAutoSave();
                    return;
                }
            } catch (e) {
                // Invalid data, clear it
                clearAutoSave();
                return;
            }
            
            const saveDate = new Date(parseInt(savedTimestamp));
            const minutesAgo = Math.floor((Date.now() - saveDate.getTime()) / 60000);
            
            // Only offer to restore if saved within last 24 hours AND more than 3 minutes ago
            // (avoid showing prompt for recently auto-saved data from same session)
            if (minutesAgo >= 3 && minutesAgo < 1440) {
                const message = `Found auto-saved work from ${minutesAgo} minute${minutesAgo !== 1 ? 's' : ''} ago. Would you like to restore it?`;
                
                // Create restore prompt
                showRestorePrompt(message, savedData);
            } else if (minutesAgo >= 1440) {
                // Clear old auto-save data
                clearAutoSave();
            }
            // If minutesAgo <= 0, it's too recent, don't prompt
        }
    }

    // Show restore prompt to user
    function showRestorePrompt(message, savedData) {
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 10000;
            display: flex;
            align-items: center;
            justify-content: center;
        `;

        const dialog = document.createElement('div');
        dialog.style.cssText = `
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            max-width: 500px;
            text-align: center;
        `;

        dialog.innerHTML = `
            <h3 style="margin-top: 0; color: #0056b3;">💾 Restore Previous Session?</h3>
            <p style="font-size: 16px; margin: 20px 0;">${message}</p>
            <button id="restoreBtn" style="padding: 10px 20px; background: #28a745; color: white; border: none; border-radius: 4px; margin-right: 10px; cursor: pointer; font-size: 16px;">✅ Restore</button>
            <button id="discardBtn" style="padding: 10px 20px; background: #dc3545; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px;">❌ Discard</button>
        `;

        overlay.appendChild(dialog);
        document.body.appendChild(overlay);

        // Handle restore
        document.getElementById('restoreBtn').addEventListener('click', function() {
            restoreFormData(savedData);
            document.body.removeChild(overlay);
            showNotification('✅ Session restored successfully!', 'success');
        });

        // Handle discard
        document.getElementById('discardBtn').addEventListener('click', function() {
            clearAutoSave();
            document.body.removeChild(overlay);
            showNotification('Session discarded', 'info');
        });
    }

    // Start auto-save timer
    function startAutoSave() {
        if (autoSaveTimer) {
            clearInterval(autoSaveTimer);
        }
        
        autoSaveTimer = setInterval(saveFormData, AUTO_SAVE_INTERVAL);
        
        // Create and insert last saved indicator
        createLastSavedIndicator();
    }

    // Save all form data to localStorage
    function saveFormData() {
        const formData = {};
        
        // Find all forms on the page
        const forms = document.querySelectorAll('form');
        
        forms.forEach(form => {
            const inputs = form.querySelectorAll('input, select, textarea');
            
            inputs.forEach(input => {
                if (input.name && input.type !== 'file' && input.type !== 'submit') {
                    if (input.type === 'checkbox') {
                        formData[input.name] = input.checked;
                    } else if (input.type === 'radio') {
                        if (input.checked) {
                            formData[input.name] = input.value;
                        }
                    } else {
                        formData[input.name] = input.value;
                    }
                }
            });
        });
        
        // Also save any textarea content (like CSV data)
        const textareas = document.querySelectorAll('textarea');
        textareas.forEach(textarea => {
            if (textarea.id) {
                formData[`textarea_${textarea.id}`] = textarea.value;
            }
        });
        
        // Only save if there's actual data
        if (Object.keys(formData).length > 0) {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(formData));
            localStorage.setItem(TIMESTAMP_KEY, Date.now().toString());
            lastSaveTime = Date.now();
            updateLastSavedIndicator();
            console.log('[Auto-Save] Data saved to localStorage');
        }
    }

    // Restore form data from localStorage
    function restoreFormData(savedDataStr) {
        try {
            const formData = JSON.parse(savedDataStr);
            
            // Restore regular form fields
            Object.keys(formData).forEach(key => {
                if (key.startsWith('textarea_')) {
                    // Restore textarea by ID
                    const textareaId = key.replace('textarea_', '');
                    const textarea = document.getElementById(textareaId);
                    if (textarea) {
                        textarea.value = formData[key];
                    }
                } else {
                    // Restore regular inputs
                    const inputs = document.querySelectorAll(`[name="${key}"]`);
                    inputs.forEach(input => {
                        if (input.type === 'checkbox') {
                            input.checked = formData[key];
                        } else if (input.type === 'radio') {
                            if (input.value === formData[key]) {
                                input.checked = true;
                            }
                        } else {
                            input.value = formData[key];
                        }
                    });
                }
            });
            
            lastSaveTime = parseInt(localStorage.getItem(TIMESTAMP_KEY));
            updateLastSavedIndicator();
            
        } catch (error) {
            console.error('[Auto-Save] Error restoring data:', error);
            showNotification('⚠️ Error restoring session data', 'error');
        }
    }

    // Create last saved indicator
    function createLastSavedIndicator() {
        const indicator = document.createElement('div');
        indicator.id = 'autoSaveIndicator';
        indicator.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #f8f9fa;
            padding: 10px 15px;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
            font-size: 13px;
            color: #666;
            z-index: 9999;
            border-left: 4px solid #28a745;
        `;
        indicator.innerHTML = '💾 Auto-save active';
        document.body.appendChild(indicator);
        
        // Update initial timestamp
        const savedTimestamp = localStorage.getItem(TIMESTAMP_KEY);
        if (savedTimestamp) {
            lastSaveTime = parseInt(savedTimestamp);
            updateLastSavedIndicator();
        }
    }

    // Update last saved indicator
    function updateLastSavedIndicator() {
        const indicator = document.getElementById('autoSaveIndicator');
        if (!indicator) return;
        
        if (lastSaveTime) {
            const minutesAgo = Math.floor((Date.now() - lastSaveTime) / 60000);
            if (minutesAgo === 0) {
                indicator.innerHTML = '💾 Last saved: just now';
            } else if (minutesAgo === 1) {
                indicator.innerHTML = '💾 Last saved: 1 minute ago';
            } else if (minutesAgo < 60) {
                indicator.innerHTML = `💾 Last saved: ${minutesAgo} minutes ago`;
            } else {
                const hoursAgo = Math.floor(minutesAgo / 60);
                indicator.innerHTML = `💾 Last saved: ${hoursAgo} hour${hoursAgo !== 1 ? 's' : ''} ago`;
            }
        } else {
            indicator.innerHTML = '💾 Auto-save active';
        }
    }

    // Clear auto-save data
    function clearAutoSave() {
        localStorage.removeItem(STORAGE_KEY);
        localStorage.removeItem(TIMESTAMP_KEY);
        lastSaveTime = null;
        console.log('[Auto-Save] Cleared auto-save data');
    }

    // Show notification to user
    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        const bgColor = {
            'success': '#28a745',
            'error': '#dc3545',
            'info': '#17a2b8'
        }[type] || '#17a2b8';
        
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${bgColor};
            color: white;
            padding: 15px 20px;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            z-index: 10001;
            font-size: 15px;
            animation: slideIn 0.3s ease-out;
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => document.body.removeChild(notification), 300);
        }, 3000);
    }

    // Add CSS animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(400px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(400px); opacity: 0; }
        }
    `;
    document.head.appendChild(style);

    // Expose public API
    window.StrykeAutoSave = {
        save: saveFormData,
        clear: clearAutoSave,
        restore: checkForAutoSavedData
    };

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
