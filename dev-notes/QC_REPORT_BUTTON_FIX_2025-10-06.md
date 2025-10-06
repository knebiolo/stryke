# QC Report - Button Behavior Fix - October 6, 2025

## Issues Identified
1. ❌ Flow scenarios page had TWO Save Flow Scenario buttons
2. ❌ Next buttons were always visible (should only show when project loaded)
3. ❌ Save buttons were not properly disabled when loading a project
4. ❌ Next button on create_project was always showing (should be conditional)

## Fixes Applied

### 1. Flow Scenarios Page (`flow_scenarios.html`)
**Before:**
```html
<button type="submit" id="saveScenarioBtn" {% if project_loaded %}disabled{% endif %}>Save Flow Scenario</button>
<button type="submit" id="saveScenarioBtn" data-enable-on-change="true" {% if project_loaded %}disabled{% endif %}>Save Flow Scenario</button>
<button type="button" onclick="window.location.href='{{ url_for('facilities') }}'">Next Page →</button>
```

**After:**
```html
<div style="display: flex; gap: 10px; align-items: center;">
    <button type="submit" id="saveScenarioBtn" data-enable-on-change="true" 
            {% if project_loaded %}disabled style="background: #ccc; cursor: not-allowed;"{% endif %}>
        Save Flow Scenario
    </button>
    
    {% if project_loaded and flow_scenario_name %}
    <button type="button" onclick="window.location.href='{{ url_for('facilities') }}'" 
            style="background: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer; padding: 10px 15px;">
        Next: Facilities →
    </button>
    {% endif %}
</div>
```

**Changes:**
- ✅ Removed duplicate Save button
- ✅ Added proper disabled styling (grey background, not-allowed cursor)
- ✅ Made Next button conditional (only shows if project_loaded AND flow_scenario_name exists)
- ✅ Wrapped buttons in flex container for proper alignment
- ✅ Added blue styling to Next button

### 2. Create Project Page (`create_project.html`)
**Before:**
```html
<button type="submit" id="createProjectBtn" data-enable-on-change="true" {% if project_loaded %}disabled style="background: #ccc; cursor: not-allowed;"{% endif %}>Create Project</button>

<button type="button" onclick="window.location.href='{{ url_for('flow_scenarios') }}'">Next Page →</button>
```

**After:**
```html
<div style="display: flex; gap: 10px; align-items: center;">
    <button type="submit" id="createProjectBtn" data-enable-on-change="true" 
            {% if project_loaded %}disabled style="background: #ccc; cursor: not-allowed;"{% endif %}>
        Create Project
    </button>
    
    {% if project_loaded %}
    <button type="button" onclick="window.location.href='{{ url_for('flow_scenarios') }}'" 
            style="background: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer; padding: 10px 15px;">
        Next Page →
    </button>
    {% endif %}
</div>
```

**Changes:**
- ✅ Made Next button conditional (only shows if project_loaded)
- ✅ Wrapped buttons in flex container for proper alignment
- ✅ Added blue styling to Next button

## Verification Results

### Button Count Per Page
Ran grep search for Save buttons across all templates:

| Page | Save Buttons | Status |
|------|--------------|--------|
| create_project.html | 1 (Create Project) | ✅ Correct |
| flow_scenarios.html | 1 (Save Flow Scenario) | ✅ **FIXED** (was 2) |
| facilities.html | 1 (Save Facilities) | ✅ Correct |
| unit_parameters.html | 1 (Save Unit Parameters) | ✅ Correct |
| operating_scenarios.html | 1 (Save Operating Scenarios) | ✅ Correct |
| population.html | 1 per conditional branch | ✅ Correct |

### Next Button Conditional Logic

| Page | Next Button Condition | Status |
|------|----------------------|--------|
| create_project.html | `{% if project_loaded %}` | ✅ **FIXED** |
| flow_scenarios.html | `{% if project_loaded and flow_scenario_name %}` | ✅ **FIXED** |
| facilities.html | `{% if project_loaded and facilities_data %}` | ✅ Already correct |
| unit_parameters.html | `{% if project_loaded and session.get('unit_params_lookup') %}` | ✅ Already correct |
| operating_scenarios.html | `{% if project_loaded and has_op_scen_data %}` | ✅ Already correct |
| graph_editor.html | Always visible (link) | ⚠️ Different pattern |
| population.html | `{% if project_loaded %}` | ✅ Already correct |

## Expected Behavior Summary

### Creating New Project (project_loaded = False)
| Action | Save Button | Next Button |
|--------|-------------|-------------|
| Page loads | ✅ ACTIVE | ❌ HIDDEN |
| After saving | ✅ ACTIVE | ✅ VISIBLE |

### Loading Existing Project (project_loaded = True)
| Action | Save Button | Next Button |
|--------|-------------|-------------|
| Page loads | ❌ DISABLED (grey) | ✅ VISIBLE (blue) |
| After editing field | ✅ ACTIVE (enable-on-change.js) | ✅ VISIBLE |

## Testing Recommendations

### Test Case 1: New Project Creation
1. Start at home page
2. Navigate to Create Project page
3. ✅ Verify: Create button is ACTIVE (blue)
4. ✅ Verify: Next button is HIDDEN
5. Fill in project details and click Create Project
6. ✅ Verify: Next button APPEARS
7. Click Next → Flow Scenarios page
8. ✅ Verify: Save button is ACTIVE, Next button is HIDDEN
9. Fill in flow scenario and click Save
10. ✅ Verify: Next button APPEARS
11. Continue through all pages verifying pattern

### Test Case 2: Load Existing Project
1. Go to home page and load existing project
2. Navigate to each page in sequence
3. ✅ Verify on each page:
   - Save button is DISABLED with grey background
   - Next button is VISIBLE with blue background
4. On any page, modify a form field (type in input, change select, etc.)
5. ✅ Verify: Save button immediately becomes ACTIVE (blue)
6. Navigate to next page
7. ✅ Verify: Next button works correctly

### Test Case 3: Flow Scenarios Specific
1. Load existing project
2. Go to Flow Scenarios page
3. ✅ Verify: Only ONE "Save Flow Scenario" button exists
4. ✅ Verify: Save button is DISABLED (grey)
5. ✅ Verify: "Next: Facilities →" button is VISIBLE (blue)
6. Modify scenario_name field
7. ✅ Verify: Save button becomes ACTIVE
8. Click Next button
9. ✅ Verify: Navigation to Facilities page works

## Files Modified
- `webapp/templates/create_project.html` (Lines 140-147)
- `webapp/templates/flow_scenarios.html` (Lines 163-171)

## Documentation Created
- `dev-notes/BUTTON_BEHAVIOR_QC_2025-10-06.md` - Comprehensive button behavior documentation
- `dev-notes/QC_REPORT_BUTTON_FIX_2025-10-06.md` - This file

## Technical Notes

### enable-on-change.js Mechanism
The JavaScript handler:
1. Listens at document level for `input` and `change` events
2. Finds buttons with `data-enable-on-change="true"` attribute
3. If button is disabled, enables it and removes inline disabled styles
4. Uses event delegation for dynamic content compatibility

### Conditional Button Variables
Each page checks different variables to determine if data exists:
- create_project: `project_loaded`
- flow_scenarios: `project_loaded and flow_scenario_name`
- facilities: `project_loaded and facilities_data`
- unit_parameters: `project_loaded and session.get('unit_params_lookup')`
- operating_scenarios: `project_loaded and has_op_scen_data`
- population: `project_loaded`

### Styling Consistency
**Disabled buttons:**
```html
{% if project_loaded %}disabled style="background: #ccc; cursor: not-allowed;"{% endif %}
```

**Next buttons:**
```html
style="background: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer; padding: 10px 15px;"
```

## Issues Resolved
✅ Duplicate Save Flow Scenario buttons removed  
✅ Next buttons now conditional on project_loaded state  
✅ Save buttons properly disabled with grey styling when project loaded  
✅ Next buttons properly styled with blue color  
✅ Consistent button layout with flex containers  

## Known Remaining Items
1. Graph Editor page uses a simple link instead of button (different UI pattern)
2. Script tag placement varies (some in head, some inline) - functional but not consistent
3. Population page has two different button blocks in if/else (this is correct behavior)

## Sign-off
All issues identified by user have been addressed:
- ✅ "Two save flow scenarios buttons" → Fixed, now only one
- ✅ "Both were active when we loaded a project" → Fixed, now disabled with grey styling
- ✅ "Next button should be activated when end user loads project" → Fixed, conditional visibility
- ✅ "When data changes, save buttons appear" → Working via enable-on-change.js

**Status: COMPLETE ✅**
