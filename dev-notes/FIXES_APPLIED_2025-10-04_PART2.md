# Additional Fixes Applied - October 4, 2025 (Part 2)

## Issues Fixed

### Issue 1: ✅ Missing Save Project Button on Operating Scenarios Page
**Solution:** Added Save Project section to `operating_scenarios.html`
- Yellow highlight box matching other pages
- Single "💾 Save Project" button
- Consistent styling with create_project, facilities, unit_parameters pages

### Issue 2: ⚠️ File Save Dialog and Default Directory
**Problem:** Browser automatically downloads to Downloads folder and adds (1), (2), etc. for duplicate filenames instead of showing a "Save As" dialog.

**Technical Limitation:** 
- Web browsers do NOT allow web applications to:
  - Show a native "Save As" dialog
  - Set the default download directory
  - Prevent automatic renaming of duplicate files
- This is a security feature to prevent malicious websites from accessing user file systems

**Current Behavior:**
- Files download directly to user's Downloads folder (browser default)
- If file exists, browser automatically appends (1), (2), (3), etc.
- User must manually move/rename files after download

**Potential Workarounds:**
1. **User Browser Settings:** Users can configure their browser to "Ask where to save each file"
   - Chrome: Settings → Downloads → Enable "Ask where to save each file before downloading"
   - Edge: Settings → Downloads → Enable "Ask me what to do with each download"
   - Firefox: Settings → General → Downloads → Select "Always ask you where to save files"

2. **Version Naming Convention:** We could add timestamp back (removed earlier at user request):
   ```python
   filename = f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.stryke"
   ```

3. **Keep Current Behavior:** User manually manages versions in Downloads folder

**Recommendation:** Document in user guide that users should enable "Ask where to save" in browser settings if they want to control save location.

### Issue 3: ✅ Missing Save Project Button on Migratory Route Creation Page (Graph Editor)
**Solution:** Replaced the "Graph/Network Template" section with "Save Project" section
- Removed confusing page-specific template buttons
- Added consistent Save Project button matching other pages
- Yellow highlight box with standard styling

### Issue 4: ✅ Auto-Save Restore Prompt Appearing Too Soon (2 minutes ago)
**Problem:** Auto-save was prompting to restore data saved just 1+ minutes ago, which was annoying when user just navigated between pages.

**Solution:** Updated `autosave.js` line 58:
- Changed threshold from `minutesAgo > 0` to `minutesAgo >= 3`
- Now only prompts if auto-save is 3+ minutes old
- Prevents prompt during normal navigation within same session
- Still prompts for legitimately old data (3 min to 24 hours)

**Code Change:**
```javascript
// Before:
if (minutesAgo > 0 && minutesAgo < 1440) {

// After:
if (minutesAgo >= 3 && minutesAgo < 1440) {
```

### Issue 5: ✅ Progress Bar Always at 100%
**Problem:** Progress calculation was dividing by `totalDays` but only checking `if (totalDays && currentDay > 0)`, which could cause issues when currentDay hasn't been set yet.

**Solution:** Updated `simulation_logs.html` progress calculation:
- Changed condition from `if (totalDays && currentDay > 0)` to `if (totalDays && totalDays > 0)`
- Now correctly calculates percentage as `(currentDay / totalDays) * 100`
- For 920 days, day 1 = 0.1%, day 460 = 50%, day 920 = 100%

**Code Change:**
```javascript
// Before:
if (totalDays && currentDay > 0) {

// After:
if (totalDays && totalDays > 0) {
```

### Issue 6: ✅ Connection Lost Error During Simulation
**Problem:** SSE (Server-Sent Events) connection was dropping during long simulations with no error message or recovery.

**Solutions Applied:**

#### A. Increased Server-Side Timeout (`app.py`)
- Changed queue timeout from 20 seconds to 30 seconds
- Reduces likelihood of premature connection closure
- Added exception handling for queue errors

**Code Change:**
```python
# Before:
msg = q.get(timeout=20)

# After:
msg = q.get(timeout=30)  # Increased timeout to 30 seconds
```

#### B. Added Client-Side Error Handling (`simulation_logs.html`)
- Added `onerror` event handler to EventSource
- Displays clear error message when connection drops
- Updates status indicator to show "Connection Lost" with ⚠️ icon
- Informs user simulation may still be running
- Suggests checking results page or refreshing

**Code Added:**
```javascript
eventSource.onerror = function(error) {
    console.error('EventSource error:', error);
    if (eventSource.readyState === EventSource.CLOSED) {
        logOutput.innerHTML += '<div class="log-error">❌ Connection lost. The simulation may still be running on the server. Please check the results page or refresh to reconnect.</div>';
        statusText.textContent = 'Connection Lost';
        statusIcon.textContent = '⚠️';
    }
};
```

## Files Modified

### Templates
1. `webapp/templates/operating_scenarios.html` - Added Save Project section
2. `webapp/templates/graph_editor.html` - Replaced Template section with Save Project section
3. `webapp/templates/simulation_logs.html` - Fixed progress calculation and added error handling

### JavaScript
4. `webapp/static/autosave.js` - Changed restore prompt threshold from 1 to 3 minutes

### Backend
5. `webapp/app.py` - Increased SSE timeout and added error handling

## Testing Checklist

- [x] Save project button appears on Operating Scenarios page
- [x] Save project button appears on Graph Editor page (Migratory Route Creation)
- [x] Auto-save doesn't prompt when navigating pages within 3 minutes
- [x] Auto-save still prompts for data 3+ minutes old
- [ ] Progress bar shows actual progress (0% → 100%) during simulation
- [ ] Connection lost error displays helpful message
- [ ] Long simulations complete without connection drops
- [ ] Users can configure browser to show "Save As" dialog

## Known Limitations

### Browser Download Behavior (Issue #2)
**Cannot be fixed via web application:**
- Cannot force "Save As" dialog (browser security restriction)
- Cannot set default save directory (browser security restriction)
- Cannot prevent (1), (2), (3) appending for duplicates (browser behavior)

**User Workarounds:**
1. Enable "Ask where to save" in browser settings
2. Use operating system file management after download
3. Apply version naming conventions when saving projects

## Recommendations for User Documentation

Add to user guide:
1. **File Management Section:**
   - Explain that .stryke files download to Downloads folder by default
   - Show how to enable "Ask where to save" in Chrome/Edge/Firefox
   - Recommend version naming convention (e.g., MyProject_v1, MyProject_v2)
   - Suggest organizing projects in dedicated folders

2. **Troubleshooting Section:**
   - "Connection Lost" during simulation: Wait and check results page
   - Long simulations: Keep browser tab open, don't close/minimize
   - Progress bar stuck: Refresh page to reconnect to simulation
   - Multiple file versions: Browser auto-adds (1), (2) - rename files manually

## Future Enhancements (Optional)

1. **Alternative Save Method:** Add a "Copy to Clipboard" option that copies JSON to clipboard, user can paste into text editor and save wherever they want

2. **Server-Side Storage:** Implement project storage on server with project library/management UI (more complex)

3. **Desktop App Version:** Create Electron wrapper that would allow full file system access and native "Save As" dialogs

4. **Version Naming Prompt:** Before save, show modal asking user to enter version number/name (e.g., "v1", "final", "2025-10-04")
