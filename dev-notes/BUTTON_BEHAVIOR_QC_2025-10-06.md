# Button Behavior QC - October 6, 2025

## Summary of Changes
Fixed button behavior across all workflow pages to follow consistent logic:
- **Save/Create buttons**: Disabled when project is loaded, re-enabled by `enable-on-change.js` when form data changes
- **Next buttons**: Only visible when project is loaded AND data has been saved for that page

## Button Behavior Rules

### When Creating New Project (project_loaded = False)
- ✅ **Create/Save buttons**: ACTIVE (allow user to save data)
- ✅ **Next buttons**: HIDDEN (must save data before proceeding)

### When Loading Existing Project (project_loaded = True)
- ✅ **Create/Save buttons**: DISABLED with grey styling (data already exists)
- ✅ **Next buttons**: VISIBLE and ACTIVE (can navigate through workflow)
- ✅ **After editing**: Save buttons become ACTIVE again (enable-on-change.js)

## Page-by-Page Status

### 1. Create Project (`create_project.html`)
**Lines 140-147**
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
- ✅ Create button disabled when project loaded
- ✅ Next button only shows when project loaded
- ✅ enable-on-change.js enabled via data attribute

### 2. Flow Scenarios (`flow_scenarios.html`)
**Lines 163-171**
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
- ✅ Save button disabled when project loaded
- ✅ Next button only shows when project loaded AND flow_scenario_name exists
- ✅ enable-on-change.js enabled via data attribute
- ✅ **FIXED**: Removed duplicate Save button

### 3. Facilities (`facilities.html`)
**Lines 123-131**
```html
<div style="display: flex; gap: 10px; align-items: center; margin-top: 15px;">
    <button type="submit" data-enable-on-change="true" 
            style="{% if project_loaded and facilities_data %}background: #ccc; cursor: not-allowed;{% else %}background: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer; padding: 10px 15px; margin: 0;{% endif %}" 
            {% if project_loaded and facilities_data %}disabled{% endif %}>
        Save Facilities
    </button>
    
    {% if project_loaded and facilities_data %}
    <a href="{{ url_for('unit_parameters') }}" style="text-decoration: none;">
        <button type="button" style="background: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer; padding: 10px 15px; margin: 0;">
            Next: Unit Parameters →
        </button>
    </a>
    {% endif %}
</div>
```
- ✅ Save button disabled when project loaded AND facilities_data exists
- ✅ Next button only shows when project loaded AND facilities_data exists
- ✅ enable-on-change.js enabled via data attribute

### 4. Unit Parameters (`unit_parameters.html`)
**Lines 455-463**
```html
<div style="display: flex; gap: 10px; align-items: center; margin-top: 15px;">
    <button type="submit" data-enable-on-change="true" 
            {% if project_loaded and session.get('unit_params_lookup') %}disabled style="background: #ccc; cursor: not-allowed;"{% endif %}>
        Save Unit Parameters
    </button>
    
    {% if project_loaded and session.get('unit_params_lookup') %}
    <a href="{{ url_for('operating_scenarios') }}" style="text-decoration: none;">
        <button type="button" style="background: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer; padding: 10px 15px; margin: 0;">
            Next: Operating Scenarios →
        </button>
    </a>
    {% endif %}
</div>
```
- ✅ Save button disabled when project loaded AND unit_params_lookup exists
- ✅ Next button only shows when project loaded AND unit_params_lookup exists
- ✅ enable-on-change.js enabled via data attribute
- ⚠️ Script include at line 457 (slightly misplaced but functional)

### 5. Operating Scenarios (`operating_scenarios.html`)
**Lines 310-318**
```html
<div style="display: flex; gap: 10px; align-items: center; margin-top: 15px;">
    <button type="submit" data-enable-on-change="true" 
            {% if project_loaded and has_op_scen_data %}disabled style="background: #ccc; cursor: not-allowed;"{% endif %}>
        Save Operating Scenarios
    </button>
    
    {% if project_loaded and has_op_scen_data %}
    <a href="{{ url_for('graph_editor') }}" style="text-decoration: none;">
        <button type="button" style="background: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer; padding: 10px 15px; margin: 0;">
            Next: Graph Editor →
        </button>
    </a>
    {% endif %}
</div>
```
- ✅ Save button disabled when project loaded AND has_op_scen_data exists
- ✅ Next button only shows when project loaded AND has_op_scen_data exists
- ✅ enable-on-change.js enabled via data attribute
- ⚠️ Script include at line 312 (slightly misplaced but functional)

### 6. Graph Editor (`graph_editor.html`)
**Line 401**
```html
<a href="{{ url_for('population') }}">Next Page</a>
```
- ⚠️ Simple link (not a styled button)
- ℹ️ No conditional logic (always visible)
- ℹ️ This page uses a different UI pattern (SVG editor)

### 7. Population (`population.html`)
**Lines 462-473**
```html
<div style="display: flex; gap: 10px; align-items: center; margin-top: 20px;">
  {% if project_loaded %}
    <button type="submit" id="savePopulationBtn" data-enable-on-change="true" 
            style="background-color: #cccccc; cursor: not-allowed;" disabled>
        Save Population Parameters
    </button>
    <a href="{{ url_for('model_setup_summary') }}" style="text-decoration: none;">
      <button type="button" style="background-color: #007BFF; color: white; border: none; padding: 10px 20px; cursor: pointer; border-radius: 5px;">
        Next →
      </button>
    </a>
  {% else %}
    <button type="submit">Save Population Parameters</button>
  {% endif %}
</div>
```
- ✅ Save button disabled when project loaded
- ✅ Next button only shows when project loaded
- ✅ enable-on-change.js enabled via data attribute
- ✅ Script include correctly placed in head section (line 207)

## JavaScript Handler (`enable-on-change.js`)

**Purpose**: Re-enables disabled Save/Create buttons when form inputs change

**Mechanism**:
- Uses delegated event handling at document level
- Listens for `input` and `change` events
- Targets only buttons with `data-enable-on-change="true"`
- Removes inline disabled styles when enabling

**Location**: `webapp/static/enable-on-change.js`

**Included in all pages**: ✅
- create_project.html
- flow_scenarios.html  
- facilities.html
- unit_parameters.html
- operating_scenarios.html
- population.html

## Testing Checklist

### New Project Creation Flow
- [ ] Create Project page: Create button ACTIVE, Next button HIDDEN
- [ ] Click Create Project → Next button should APPEAR
- [ ] Flow Scenarios page: Save button ACTIVE, Next button HIDDEN
- [ ] Click Save Flow Scenario → Next button should APPEAR
- [ ] Continue through workflow, verifying same pattern

### Load Existing Project Flow
- [ ] Load project from home page
- [ ] Create Project page: Create button DISABLED (grey), Next button VISIBLE
- [ ] Click Next → Flow Scenarios page
- [ ] Flow Scenarios: Save button DISABLED (grey), Next button VISIBLE
- [ ] Modify any input field → Save button should become ACTIVE
- [ ] Continue through all pages, verifying Next buttons are VISIBLE and ACTIVE

### Data Change Re-enable
- [ ] Load existing project
- [ ] On any page with disabled Save button, modify a form field
- [ ] Save button should immediately become ACTIVE (enable-on-change.js)
- [ ] Button styling should change from grey to blue

## Known Issues & Notes

1. **Script placement**: Some pages have `enable-on-change.js` script tag inside the button div (e.g., unit_parameters.html line 457). This works but is not ideal placement. Should be in `<head>` section for consistency.

2. **Graph Editor**: Uses different UI pattern (SVG editor), simple link instead of conditional button. May need review if consistent styling is required.

3. **Lint errors**: VS Code reports JavaScript syntax errors in Jinja2 templates (e.g., `{{ 'true' if project_loaded else 'false' }}`). These are false positives - Jinja2 templates render correctly.

4. **Flow scenario condition**: Uses `flow_scenario_name` variable to check if data exists. Verify this variable is consistently set in Flask route.

## Files Modified
- `webapp/templates/create_project.html` - Added conditional Next button
- `webapp/templates/flow_scenarios.html` - Removed duplicate Save button, added conditional Next button
- Previous fixes already in place for other pages

## Related Documentation
- `dev-notes/INPUT_VALIDATION_IMPLEMENTATION.md` - Form validation patterns
- `dev-notes/PROJECT_SAVE_LOAD_IMPLEMENTATION.md` - Project state management
- `dev-notes/CRITICAL_FIXES_2025-10-05_ROUND2.md` - Previous button fixes
