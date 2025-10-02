# Security Patches Applied to app.py
**Date:** October 2, 2025  
**Purpose:** Prevent session bleed-over between concurrent users

---

## 🔧 Patches Applied

### **Priority 1: Logger Handler Cleanup (CRITICAL)**

#### **Issue**
Log handlers were attached to global logger objects per-run but not properly cleaned up, causing:
- Cross-user log message contamination
- Memory leaks from accumulated handlers
- Race conditions under concurrent load

#### **Fix Applied**
1. **Added `_detach_queue_log_handler()` function** (Lines ~165-172)
   - Properly removes handlers from all target loggers
   - Restores `propagate=True` to reset logger state
   - Wrapped in try-except for safety

2. **Disabled logger propagation during runs** (Line ~173)
   - Set `lg.propagate = False` when attaching handlers
   - Prevents logs from bubbling up to root logger and cross-contaminating

3. **Added handler cleanup to `run_simulation_in_background_custom()`** (Lines ~3012, ~3078)
   - Attached handler using `h, targets = _attach_queue_log_handler(q)`
   - Added cleanup in `finally` block: `_detach_queue_log_handler(h, targets)`
   - Guaranteed cleanup even if simulation crashes

4. **Existing XLS simulation already had cleanup** (Lines ~449-454)
   - Confirmed proper handler removal in `run_xls_simulation_in_background()`

**Impact:** ✅ Eliminates cross-user log contamination

---

### **Priority 2: Matplotlib Figure Isolation**

#### **Issue**
Global matplotlib state (`plt.clf()`, `plt.close()`) could bleed between concurrent requests, causing:
- Corrupted plots when multiple users generate figures simultaneously
- Race conditions on the global pyplot state machine

#### **Fix Applied**
1. **Replaced global pyplot calls with explicit figures** (Lines ~759-775)
   - Changed `plt.clf(); fish.plot(); plt.savefig(); plt.close()`
   - To: `fig1 = plt.figure(); fish.plot(); fig1.savefig(); plt.close(fig1)`
   
2. **Isolated histogram generation** (Lines ~768-775)
   - Changed `plt.clf(); plt.figure(); plt.hist(); plt.savefig(); plt.close()`
   - To: `fig2 = plt.figure(); plt.hist(); fig2.savefig(); plt.close(fig2)`

**Impact:** ✅ Each request gets isolated figure objects, preventing state bleed-over

---

### **Priority 3: Session Folder Enforcement**

#### **Issue**
`/fit` route had a fallback to global `SIM_PROJECT_FOLDER`, causing:
- Multiple users writing to the same directory (rare edge case)
- File naming collisions
- Potential security issue (users seeing each other's data)

#### **Fix Applied**
1. **Removed global folder fallback** (Line ~657)
   - Changed: `sim_folder = g.get("user_sim_folder", SIM_PROJECT_FOLDER)`
   - To: `sim_folder = g.get("user_sim_folder")`
   
2. **Added session validation** (Lines ~658-660)
   ```python
   if not sim_folder:
       flash('Session expired. Please log in again.')
       return redirect(url_for('login'))
   ```

**Impact:** ✅ Forces re-authentication if session expires, no shared folder fallback

---

## 🧪 Testing Recommendations

### **Test 1: Concurrent User Logging**
1. Open two browser sessions (different users)
2. Run simulations simultaneously
3. Check that logs don't cross-contaminate in `/stream` SSE endpoint
4. Verify no duplicate handlers in logger after simulation completes

### **Test 2: Concurrent Plot Generation**
1. Two users access `/fit` simultaneously
2. Generate distribution plots at the same time
3. Verify both users get correct, non-corrupted PNG files
4. Check that `fitting_results.png` is unique per user session folder

### **Test 3: Session Expiry Handling**
1. Log in, start a project
2. Clear session cookies manually
3. Try to access `/fit` route
4. Should redirect to login, not write to global folder

---

## 📊 Performance Impact

- **Memory:** Slightly lower (proper handler cleanup prevents leaks)
- **CPU:** Negligible (explicit figure creation is same cost)
- **Concurrency:** Improved (no more global state bottlenecks)

---

## 🔍 Additional Observations

### **Already Good (No Changes Needed)**
1. ✅ Per-user session folders with UUID isolation (`session['user_dir']`)
2. ✅ Per-run sandboxes using `run_id = uuid.uuid4().hex`
3. ✅ Per-run queues in `RUN_QUEUES` dictionary
4. ✅ Flask `g` object is request-scoped (thread-safe)
5. ✅ Cleanup timer already checks file age (24-hour cutoff)

### **Not Critical (But Monitored)**
1. ⚠️ `SESSION_LOCKS` defined but never used
   - Recommendation: Use locks when modifying session-critical data
   - Example: `with get_session_lock(user_key): session['proj_dir'] = run_dir`
   
2. ⚠️ Module-level patching of `read_csv_if_exists`
   - Currently safe (read-only utility function)
   - Avoid patching stateful methods in the future

---

## ✅ Verification Checklist

- [x] Logger handlers cleaned up in all simulation paths
- [x] Matplotlib figures isolated per-request
- [x] Session folder enforcement (no global fallback)
- [x] All changes tested in local environment
- [ ] Load testing with 5+ concurrent users
- [ ] Production deployment with monitoring

---

## 🚀 Next Steps (Optional Enhancements)

1. **Add Session Lock Usage**
   ```python
   user_key = session.get('user_dir')
   if user_key:
       with get_session_lock(user_key):
           session['proj_dir'] = run_dir
           session['last_run_id'] = run_id
   ```

2. **Add Request ID to All Logs**
   ```python
   request_id = uuid.uuid4().hex[:8]
   log.info(f"[{request_id}] Simulation started...")
   ```

3. **Add Metrics/Monitoring**
   - Track concurrent simulations
   - Monitor queue sizes
   - Alert on handler leak detection

---

## 📝 Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-02 | Initial security patches applied |

---

**Questions?** Contact the development team or review this document before deploying to production.
