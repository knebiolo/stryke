# Simulation Report Issues - October 5, 2025

## Summary
First successful simulation run after implementing project load functionality! However, 9 issues identified in the report output.

## Test Case
- **Project**: same_old.stryke
- **Species**: Smallmouth bass (Micropterus)
- **Scenario**: Summer (June-August)
- **Facility**: Matabitchuan GS (2 Francis turbines, 305m head)
- **Population**: 1,478 fish sampled
- **Iterations**: 10

## Issues Found

### Issue #1: Probability of Entrainment = 4125% ❌
**Observed**: Report shows "4125%"
**Expected**: Should be around 40-70% based on typical entrainment rates

**Analysis**:
- Display code: `{prob_entr:.4f} ({prob_entr*100:.2f}%)`
- If showing 4125%, then `prob_entr` = 41.25 (not 0.4125)
- Calculation in stryke.py line 2856: `prob_entr = np.mean(iteration_probs)`
- Where `iteration_probs = iter_entrained / iter_pop`
- This should yield a decimal 0-1, not a percentage

**Hypothesis**: Something wrong with the calculation logic where entrained count or population size is incorrect

---

### Issue #2: No Project Notes Displayed ❌
**Observed**: Project notes not showing in report
**Expected**: Should display "let's see if the AI picks this up. tell me you saw this message please."

**Location**: Report generation in `generate_report()` function
**Fix**: Add project notes section to report template

---

### Issue #3: "Units" Line Should Be Removed ❌
**Observed**: Report shows "Units: metric" line
**Expected**: Remove this line (cosmetic cleanup)

**Location**: Report generation header section
**Fix**: Remove the units display line

---

### Issue #4: All 1,478 Fish Entrained (Expected 2/3) ❌
**Observed**: All fish entrained
**Expected**: ~66% entrained, ~33% bypassed via spillway

**Analysis**:
- Model has 3 routes: Unit 1, Unit 2, Spillway
- Each should get proportional flow allocation
- 100% entrainment suggests routing logic error

**Hypothesis**: 
- Fish routing through units vs spillway not working
- Bifurcation weights not being applied correctly
- Could be related to Issue #8 (all fish through Unit 1)

---

### Issue #5: Mean Modeled Length = -0.39 meters ❌
**Observed**: Negative fish length
**Expected**: Positive length in mm or m

**Analysis**:
- Population CSV has: `length location: -0.3861` (log-normal distribution location parameter)
- This is a distribution parameter, NOT the actual length
- Length should be sampled from: Log Normal(shape=0.9155, location=-0.3861, scale=3.8903)

**Hypothesis**: Code is displaying the distribution location parameter instead of sampling actual lengths

---

### Issue #6: Total Mortalities = 252,312 but Only 1,478 Fish ❌
**Observed**: 252,312 mortalities analyzed
**Expected**: Max mortalities should be ≤ 1,478 (total fish)

**Analysis**:
- This is 170x the population size!
- Suggests double-counting or iteration multiplication error

**Hypothesis**: 
- Mortalities being summed across all days AND all iterations without proper aggregation
- Should be: mean across iterations, then sum across days OR sum per iteration, then mean
- Related to Issue #6: prob_entrainment calculation may have similar aggregation issue

---

### Issue #7: Impingement Count = 574 (QC Needed) ⚠️
**Observed**: 574 impingements
**Expected**: TBD - need to verify calculation

**Analysis**:
- 574 / 1,478 = 38.8% impingement rate
- Is this reasonable for Francis turbines with 0.328m rack spacing?

**Action**: Review impingement calculation logic in stryke.py

---

### Issue #8: All Fish Through Unit 1 (Beta Distribution Table) ❌
**Observed**: Beta distributions table shows 100% through Unit 1, 0% through Unit 2
**Expected**: Fish should split between units based on flow/discharge

**Analysis**:
- Related to Issue #4 (routing problem)
- Bifurcation logic not working correctly
- Need to check passage route probability calculations

**Critical Files**:
- `stryke.py`: Routing and bifurcation functions
- Movement functions that allocate fish to different routes

---

### Issue #9: Top Routes by Discharge Shows "river_node_1" ❌
**Observed**: Machine IDs showing in table (river_node_1)
**Expected**: Human-readable names ("Spillway", "Unit 1", etc.)

**Location**: Report generation - Top Routes section
**Fix**: Map Location IDs to human-friendly labels using nodes lookup

---

## Priority Fix Order

### Phase 1: Critical Simulation Logic (DO FIRST)
1. **Issue #8**: Fix routing - all fish through Unit 1
   - Check bifurcation weight application
   - Verify flow allocation logic
   - Test with simple 50/50 split case

2. **Issue #4**: Fix overall routing - spillway vs units
   - Related to #8, may be same root cause
   - Ensure spillway gets flow allocation

3. **Issue #1**: Fix probability calculation
   - Verify entrained count and population size
   - Check aggregation across iterations
   - Ensure result is 0-1 decimal

### Phase 2: Data Correctness
4. **Issue #6**: Fix mortality counting
   - Review aggregation logic
   - Ensure no double-counting
   - Verify sum vs mean operations

5. **Issue #5**: Fix length sampling
   - Ensure sampling from distribution, not using parameter
   - Display actual sampled lengths

### Phase 3: Report Display
6. **Issue #2**: Add project notes
7. **Issue #3**: Remove Units line  
8. **Issue #9**: Map route IDs to names

### Phase 4: QC
9. **Issue #7**: Verify impingement calculation

---

## Root Cause Hypothesis

**Primary Issue**: Routing/bifurcation logic not working correctly
- Manifests as Issue #4 (all entrained) and #8 (all through Unit 1)
- Fish not being allocated to multiple routes based on flow
- May be using wrong node IDs or edge weights

**Secondary Issues**: 
- Aggregation errors in summary statistics (Issues #1, #6)
- Display vs actual value confusion (Issue #5)
- Report cosmetics (Issues #2, #3, #9)

---

## Next Steps

1. Review routing logic in stryke.py - focus on bifurcation functions
2. Add debug logging to track fish movement through routes
3. Verify edge weights are being applied correctly
4. Check if node IDs match between graph and movement logic
5. Fix issues one at a time, re-run simulation after each fix
6. Create unit tests for routing logic

---

## Files to Review

1. `Stryke/stryke.py` - Routing, bifurcation, movement functions
2. `webapp/app.py` - Report generation (generate_report function)
3. `population.csv` - Verify distribution parameters
4. `graph.json` - Verify edge weights and node IDs
5. Simulation HDF5 output - Inspect raw data for patterns

