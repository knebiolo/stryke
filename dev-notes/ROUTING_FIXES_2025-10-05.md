# Routing Logic Fixes - October 5, 2025

## Issues Fixed in stryke.py movement() function

### Bug #1: Spillway Getting Zero Flow (Issue #4 & #8)
**Location**: Lines 1518-1532 (after fix)
**Problem**: When spillway and units are neighbors of the same node (river_node_0), the spillway was being handled in the "else" clause meant for bypass paths. The code tried to look up the spillway in `unit_fac_dict`, got `None`, then looked up `bypass_Q_dict[None]` which returned 0.0.

**Result**: Spillway got 0% probability, all fish routed through units

**Fix**: Added explicit `elif 'spill' in i:` block that properly calculates spillway flow:
```python
elif 'spill' in i:
    # Spillway gets remaining flow after units, environmental, and bypass
    total_unit_flow = sum(route_flows)  # Sum of all unit flows calculated above
    
    if curr_Q > total_sta_cap + total_env_Q + total_bypass_Q:
        # Excess flow goes to spillway
        spill_Q = curr_Q - total_sta_cap - total_env_Q - total_bypass_Q
    elif curr_Q > total_env_Q + total_bypass_Q:
        # Some production, rest to spillway as environmental/bypass
        spill_Q = total_env_Q + total_bypass_Q
    else:
        # All flow through spillway (low flow condition)
        spill_Q = curr_Q
```

---

### Bug #2: All Fish Through Unit 1 (Issue #8)
**Location**: Lines 1487-1501 (after fix)
**Problem**: The operational order logic was using unit **capacities** from `Q_dict` instead of **actually allocated flows**. 

**Example with 300 cms total flow**:
- Unit 1 (order=1, cap=522.66): Gets min(300, 522.66) = 300 cms ✓
- Unit 2 (order=2, cap=522.66): prev_Q = 522.66 (Unit 1's CAPACITY, not allocated flow!)
  - Since prev_Q (522.66) >= prod_Q (300), Unit 2 gets 0 cms ✗

**Result**: Unit 1 got all production flow, Unit 2 got 0 flow

**Fix**: Track allocated flow in a dictionary as we process units:
```python
# Track allocated flow per facility as we process units
allocated_flow_by_facility = {f: 0.0 for f in facilities_at_node}

# Inside unit loop:
prev_Q = allocated_flow_by_facility.get(facility, 0.0)  # Use allocated, not capacity
u_Q = min(prod_Q - prev_Q, unit_cap)
allocated_flow_by_facility[facility] = prev_Q + u_Q  # Update tracker
```

---

## Expected Results After Fix

### Flow Distribution (300 cms example):
- **Before**: Unit 1 = 300 cms (100%), Unit 2 = 0 cms (0%), Spillway = 0 cms (0%)
- **After**: 
  - If 300 < sta_cap (1045): Unit 1 = 150 cms (50%), Unit 2 = 150 cms (50%), Spillway = 0 cms (0%)
  - If 300 > sta_cap: Units share station capacity, spillway gets excess

### Fish Routing:
- **Before**: 100% through Unit 1
- **After**: 50% Unit 1, 50% Unit 2 (when both operating), some through spillway at high flows

### Entrainment:
- **Before**: 100% entrainment (all through turbines)
- **After**: Variable based on spillway availability (~60-70% entrainment typical)

---

## Related Issues That Should Now Be Fixed

✅ **Issue #4**: All 1,478 fish entrained (expected 2/3)
- Spillway now gets proper flow allocation
- Fish will route through spillway at appropriate rates

✅ **Issue #8**: All fish through Unit 1 (Beta Distribution table)
- Flow now distributed between Unit 1 and Unit 2 based on operational order
- Should see ~50/50 split when both units operating

⚠️ **Issue #1**: Probability = 4125%
- May still need investigation - could be related to aggregation logic

---

## Testing Recommendations

1. **Run simulation with fixed code**
2. **Check Beta Distributions table**: Should show both Unit 1 and Unit 2 with fish
3. **Check Top Routes by Discharge**: Should show spillway with significant flow (especially at high discharge days)
4. **Check entrainment rate**: Should be < 100%, probably 60-70% depending on hydrograph
5. **Verify flow allocation**: Sum of unit flows + spillway flow should equal total daily flow

---

## Code Changes Summary

**File**: `Stryke/stryke.py`
**Function**: `movement()` (lines 1405-1580)

**Changes**:
1. Added `allocated_flow_by_facility` tracker dictionary (line 1451)
2. Changed unit flow calculation to use allocated flows not capacities (lines 1487-1501)
3. Added explicit spillway handling block (lines 1518-1532)
4. Fixed spillway flow calculation logic

**Lines Changed**: ~50 lines modified in movement() function

---

## Next Steps

After testing these routing fixes:
1. Investigate Issue #1 (prob_entrainment = 4125%)
2. Investigate Issue #5 (negative length = -0.39m)
3. Investigate Issue #6 (252K mortalities from 1,478 fish)
4. QC Issue #7 (574 impingements)

