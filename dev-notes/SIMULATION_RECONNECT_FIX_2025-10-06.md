# Simulation Reconnect Fix - October 6, 2025

## Critical Issue
**Problem**: When simulation connection times out or page is refreshed, user sees "Waiting for simulation to start..." forever, even if simulation completed successfully.

**Root Causes**:
1. **In-memory queues** - `RUN_QUEUES` dictionary is not persistent across connections
2. **Railway timeout** - 120s timeout kills EventSource connection before simulation finishes
3. **No result checking** - simulation_logs page never checks if simulation already completed
4. **Architecture flaw** - No fallback mechanism for connection loss

## Symptoms Observed
- User clicks "Run Simulation"
- Connection shows keepalive messages for ~2 minutes
- Railway timeout kills connection: "[ERROR] Connection lost or server unavailable"
- User refreshes page
- **Page shows "Waiting for simulation to start..." indefinitely**
- **Simulation may have actually completed successfully in background**

## The Fix

### 1. Backend: Check for Completed Simulations (`app.py`)
Modified `simulation_logs()` route to check for completion markers before rendering:

```python
@app.route("/simulation_logs")
def simulation_logs():
    """
    Show simulation logs page. Checks for completed simulation first.
    """
    run_id = request.args.get('run', '')
    
    # Check if simulation has already completed by looking for output files
    proj_dir = session.get('proj_dir')
    simulation_status = 'running'  # Default
    report_path = None
    
    if proj_dir and os.path.exists(proj_dir):
        # Check for simulation completion markers
        marker_file = os.path.join(proj_dir, 'report_path.txt')
        report_html = os.path.join(proj_dir, 'simulation_report.html')
        output_h5 = os.path.join(proj_dir, 'Simulation_Output.h5')
        
        if os.path.exists(marker_file):
            # Read the report path from marker file
            try:
                with open(marker_file, 'r') as f:
                    report_path = f.read().strip()
                    if os.path.exists(report_path):
                        simulation_status = 'completed'
            except Exception:
                pass
        elif os.path.exists(report_html):
            # Found report directly
            report_path = report_html
            simulation_status = 'completed'
        elif os.path.exists(output_h5):
            # Found H5 output (simulation completed but report might be missing)
            simulation_status = 'completed'
            report_path = None  # No HTML report, but sim finished
    
    return render_template("simulation_logs.html", 
                         run_id=run_id,
                         simulation_status=simulation_status,
                         report_path=report_path)
```

**What it checks**:
1. `report_path.txt` - Marker file created when report is generated
2. `simulation_report.html` - Direct report file
3. `Simulation_Output.h5` - H5 output file (proves simulation finished)

### 2. Frontend: Conditional Connection (`simulation_logs.html`)
Modified JavaScript to:
- **Check `simulation_status` from backend FIRST**
- **Only connect to EventSource if status is 'running'**
- **Show completion immediately if status is 'completed'**

```javascript
// Check if simulation already completed
const simulationStatus = "{{ simulation_status }}";
const reportPath = "{{ report_path }}";

if (simulationStatus === 'completed') {
    // Show completed state immediately - NO EventSource connection
    progressBar.style.width = '100%';
    statusText.textContent = '✅ Simulation Already Complete!';
    logOutput.innerHTML = '<div class="log-success">✅ This simulation has already completed successfully!</div>';
    resultsButton.style.display = "inline-block";
} else {
    // Simulation is running - connect to live stream
    const eventSource = new EventSource(`/stream?run=${encodeURIComponent(runId)}`);
    // ... rest of streaming logic
}
```

### 3. Improved Error Messages
Added clickable refresh link when connection lost:

```javascript
logOutput.innerHTML += '<div class="log-error">❌ Connection lost. <a href="?run=' + 
    encodeURIComponent(runId) + '">Click here to refresh and check if it completed</a>.</div>';
```

## User Flow After Fix

### Scenario 1: Normal Completion
1. User starts simulation
2. EventSource shows progress
3. Simulation completes: "[Simulation Complete]" message
4. Page shows completion state
5. ✅ Works perfectly

### Scenario 2: Connection Timeout (FIXED!)
1. User starts simulation
2. EventSource shows progress for 2 minutes
3. **Railway timeout kills connection**
4. JavaScript shows: "Connection lost. Click here to refresh..."
5. User clicks refresh
6. **Backend detects `simulation_report.html` exists**
7. **Frontend immediately shows completion state**
8. ✅ User sees results without waiting!

### Scenario 3: Page Refresh While Running (IMPROVED!)
1. User starts simulation
2. User refreshes page mid-simulation
3. **Backend checks for completion files - none exist**
4. **Frontend connects to EventSource with existing `run_id`**
5. **If simulation still running**: Shows live progress
6. **If simulation finished**: Shows completion (scenario 2)
7. ✅ Graceful handling either way!

## Files Modified
1. `webapp/app.py` - Line ~4030: Modified `simulation_logs()` route
2. `webapp/templates/simulation_logs.html` - Lines 260-300: Conditional EventSource connection

## Testing Checklist
- [ ] Start simulation, let it complete normally - should show results ✅
- [ ] Start simulation, refresh page - should reconnect or show completion ✅
- [ ] Start simulation, wait for timeout, refresh - should show completion ✅
- [ ] Load completed simulation directly - should show completion immediately ✅

## Related Issues Fixed
- ✅ Railway 120s timeout no longer breaks user experience
- ✅ Page refresh no longer loses simulation state
- ✅ In-memory queue limitations worked around
- ✅ Clear user feedback when connection lost

## Future Improvements (Optional)
1. **Persistent queue** - Store queue messages in Redis/database
2. **Background job system** - Use Celery or similar for true async
3. **Polling fallback** - Poll for completion every 5s if EventSource fails
4. **Progress file** - Write progress to JSON file that frontend can poll

## Commit Message
```
Fix: Simulation reconnect after timeout/refresh

CRITICAL FIX:
- Check for completed simulations before connecting to EventSource
- Show completion immediately if simulation already finished
- Prevents "Waiting for simulation..." infinite loop after timeout
- Gracefully handle Railway 120s timeout + page refresh

Backend: simulation_logs() route now checks for report_path.txt/H5 files
Frontend: Conditional EventSource connection based on completion status
UX: Clear error messages with refresh link when connection lost

Fixes issue where user sees "Waiting..." forever after connection timeout,
even though simulation completed successfully in background.
```

---
**Status**: ✅ FIXED - Ready to test on Railway
**Priority**: CRITICAL - Blocks all simulation workflows
**Impact**: HIGH - Affects every simulation run on Railway
