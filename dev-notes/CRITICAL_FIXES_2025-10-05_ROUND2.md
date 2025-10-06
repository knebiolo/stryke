# Critical Fixes Applied - October 5, 2025 (Round 2)

## Issues Fixed

### Issue #1: Probability of Entrainment = 3574% ✅ FIXED
**File**: `Stryke/stryke.py` lines 2883-2897

**Problem**: 
The calculation was summing entrainments across ALL days (e.g., 92 days) and dividing by a single day's population:
```python
iter_entrained = iter_data['num_entrained'].sum()  # Sum across 92 days = 35,000+
iter_pop = iter_data['pop_size'].iloc[0]           # First day pop = 1,478
prob = 35000 / 1478 = 23.68 = 2368%!
```

**Root Cause**: Fish can be counted multiple times across different days. If 500 fish are entrained on day 1, those same fish (or their descendants) can be counted again on day 2, etc. Summing across days double/triple counts fish.

**Fix**: Calculate average **daily** entrainment probability:
```python
# For each day: num_entrained / pop_size (fish at risk that day)
daily_probs = iter_data['num_entrained'] / iter_data['pop_size']
mean_daily_prob = daily_probs.mean()  # Average across all days
```

**Expected Result**: Probability should now be 0.30-0.40 (30-40%), displayed correctly as a percentage.

---

### Issue #2: Routing Order Bug ✅ FIXED
**File**: `Stryke/stryke.py` lines 1451-1455

**Problem**:
The loop processes neighbors in arbitrary order. If spillway was processed BEFORE units, the code tried to sum `route_flows` before any units were added, resulting in wrong spillway flow calculation.

**Example**:
```python
neighbors = ['spillway', 'Unit 1', 'Unit 2']  # Spillway first!
# When processing spillway:
total_unit_flow = sum(route_flows)  # = 0 (units not processed yet!)
spill_Q = curr_Q - 0 = curr_Q  # Spillway gets ALL flow!
```

**Fix**: Sort neighbors to process units first, then spillway:
```python
sorted_nbors = sorted(nbors, key=lambda x: (
    0 if 'U' in x else (2 if 'spill' in x else 1)
))
# Processing order: Units (0), Other (1), Spillway (2)
```

**Expected Result**: Units get flow first, spillway gets remainder.

---

### Issue #3: Request Context Warning ✅ FIXED
**File**: `webapp/app.py` line 4606

**Problem**:
Report generation runs in background thread without Flask request context. Tried to access `session.get('proj_dir')` which fails outside request.

**Fix**:
```python
# Before:
session_proj_dir = session.get('proj_dir')  # ❌ Fails in background thread

# After:
session_proj_dir = getattr(sim, 'proj_dir', None)  # ✅ Use sim object
```

**Expected Result**: No warning about "Working outside of request context"

---

### Issue #4: Skull Emoji ✅ FIXED
**File**: `webapp/app.py` line 4442

Removed 💀 from "Mortality Factor Breakdown" heading.

---

## Testing Checklist

After redeploying, verify:

1. **✅ Probability of Entrainment**: Should be 30-40%, not 3500%
2. **✅ No Warning**: Request context warning should be gone
3. **⚠️ Fish Distribution**: Check Beta Distributions table
   - Should show fish in BOTH Unit 1 and Unit 2
   - If still 100% through Unit 1, routing may need more debugging
4. **⚠️ Entrainment Rate**: Check "Total Fish Simulated" vs "Entrained"
   - Should be <100% (some fish through spillway)
   - If still 100%, routing bug persists
5. **⚠️ Blade Strike Count**: Should decrease if routing fixes work
   - If still showing 284K from 1,478 fish, indicates aggregation issue

---

## Known Remaining Issues

If after this fix you still see:
- **100% entrainment**: Routing logic needs more investigation
  - Check if operational order logic is working
  - Verify spillway flow calculation
  - Add debug logging to movement() function

- **Excessive blade strike counts**: May be related to:
  - Multiple counting across days (similar to probability bug)
  - Mortality aggregation logic
  - Need to investigate mortality calculation functions

---

## Files Changed

1. **Stryke/stryke.py**:
   - Lines 1451-1455: Sort neighbors for correct processing order
   - Lines 2883-2897: Fix probability calculation (daily average not cumulative sum)

2. **webapp/app.py**:
   - Line 4606: Fix session access in background thread
   - Line 4442: Remove skull emoji

---

## Commit Message

```
Fix entrainment probability calculation and routing order

- Fix prob_entrainment: use daily average instead of cumulative sum (was showing 3574%)
- Fix routing order: process units before spillway to ensure correct flow allocation  
- Fix request context warning in background thread
- Remove skull emoji from report
```

---

## Next Steps

1. **Deploy these fixes** (push to GitHub, redeploy)
2. **Run simulation** with same_old.stryke
3. **Check probability** - should be ~35% not 3574%
4. **Check routing** - should show fish in both units and spillway
5. **If routing still broken**: Add debug logging to movement() function
6. **If mortality counts still wrong**: Investigate mortality aggregation logic

