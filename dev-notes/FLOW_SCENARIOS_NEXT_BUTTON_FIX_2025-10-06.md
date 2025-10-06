# Flow Scenarios Next Button Fix - October 6, 2025

## Issue
Next button did not appear on flow scenarios page even when project was loaded.

## Root Cause
**Variable name mismatch** between Flask route and Jinja2 template:
- Flask route passes: `scenario_name`
- Template checked for: `flow_scenario_name`

### Code Comparison

**Flask Route (app.py line 1626):**
```python
return render_template('flow_scenarios.html',
                     units=units,
                     project_loaded=project_loaded,
                     scenario_name=scenario_name,  # <-- Passes 'scenario_name'
                     scenario_number=scenario_number,
                     season=season,
                     months=months,
                     scenario_type=scenario_type,
                     discharge=discharge,
                     hydrograph_data=hydrograph_data)
```

**Template BEFORE (line 166):**
```html
{% if project_loaded and flow_scenario_name %}  <!-- Wrong variable name! -->
<button type="button">Next: Facilities →</button>
{% endif %}
```

**Template AFTER (line 166):**
```html
{% if project_loaded and scenario_name %}  <!-- Matches Flask variable -->
<button type="button">Next: Facilities →</button>
{% endif %}
```

## Fix Applied
Changed template condition from:
```jinja2
{% if project_loaded and flow_scenario_name %}
```

To:
```jinja2
{% if project_loaded and scenario_name %}
```

## Pattern Consistency
This fix aligns with the pattern used on other pages:
- **facilities.html**: `{% if project_loaded and facilities_data %}`
- **unit_parameters.html**: `{% if project_loaded and session.get('unit_params_lookup') %}`
- **operating_scenarios.html**: `{% if project_loaded and has_op_scen_data %}`
- **flow_scenarios.html**: `{% if project_loaded and scenario_name %}` ← Now correct!

## Why This Happened
Likely a typo during the earlier button fix session - the variable was called `flow_scenario_name` instead of matching the actual Flask variable `scenario_name`.

## Testing
After fix, when loading a project with flow scenario data:
- ✅ `project_loaded` = True
- ✅ `scenario_name` = loaded scenario name (e.g., "Summer Low Flow")
- ✅ Both conditions met → Next button appears

## File Modified
- `webapp/templates/flow_scenarios.html` (line 166)

## Status: FIXED ✅
Next button will now properly appear when a project is loaded and has a scenario name.
