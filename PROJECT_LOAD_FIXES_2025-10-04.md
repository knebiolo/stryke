# Project Load and Save Fixes - October 4, 2025

## Issues Addressed

### 1. Missing Save Project Button on Facilities Page
**Problem:** The facilities page had no way to save the full project.

**Solution:** Added a "Save Project" section to `facilities.html` matching the styling from `create_project.html`:
- Yellow highlight box with 💾 icon
- Single "💾 Save Project" button
- Tip message about saving to Downloads folder
- Location: After the "Save Facilities" button

### 2. Unit Parameters Page Had Different Nomenclature
**Problem:** Unit parameters page referred to "Template" instead of "Project" and had different styling (blue instead of yellow).

**Solution:** Replaced the entire Template section with a matching "Save Project" section:
- Changed from blue (#e8f4f8) to yellow (#fff3cd) background
- Removed Import/Export Template buttons (these were page-specific templates, not full projects)
- Added single "💾 Save Project" button matching other pages
- Updated text to match create_project.html

### 3. Export Template Cleared Everything
**Problem:** The export_unit_params_template route returns a Response object for file download, which doesn't redirect properly and left the page in a strange state.

**Note:** This issue was resolved by removing the template export/import section entirely from unit_parameters.html and replacing it with the project save button. The template routes still exist in app.py but are no longer exposed in the UI.

### 4. Loading .stryke File Doesn't Populate Form Fields
**Problem:** The `load_project` route only wrote CSV files to disk but didn't update session variables or render templates with the loaded data.

**Solution:** Implemented comprehensive data loading across all pages:

#### A. Updated `load_project` Route (app.py lines 997-1038)
- Added session variable population when restoring project_info
- Added session variable population when restoring flow_scenarios
- Added scenario_type detection (static vs hydrograph)
- Added unit_params_file session variable

#### B. Updated `create_project` Route (app.py lines 1273-1293)
- Added GET request handler that checks for existing project.csv
- Reads CSV and passes data to template as kwargs
- Template updated to use `{{ project_name or '' }}` pattern
- Added selected states to all dropdowns

#### C. Updated `flow_scenarios` Route (app.py lines 1407-1445)
- Added comprehensive GET request handler
- Loads flow.csv and hydrograph.csv if they exist
- Detects scenario_type (static vs hydrograph)
- Converts units back to display format (CFS → metric if needed)
- Formats hydrograph data as tab-delimited string for textarea

#### D. Updated `flow_scenarios.html` Template
- Added value attributes to all input fields
- Added selected states to scenario_type dropdown
- Added conditional visibility for static_fields vs hydrograph_fields
- Populates textarea with loaded hydrograph data

#### E. Updated `create_project.html` Template
- Added value="{{ project_name or '' }}" to all input fields
- Added selected states to units and model_setup dropdowns using Jinja2

## Files Modified

### Templates
1. `webapp/templates/facilities.html` - Added Save Project section
2. `webapp/templates/unit_parameters.html` - Replaced Template section with Save Project section
3. `webapp/templates/create_project.html` - Added value attributes and selected states
4. `webapp/templates/flow_scenarios.html` - Added value attributes, selected states, conditional visibility

### Backend
1. `webapp/app.py`:
   - `load_project()` - Added session variable population (lines 997-1038)
   - `create_project()` - Added GET handler to load existing data (lines 1273-1293)
   - `flow_scenarios()` - Added GET handler to load existing data (lines 1407-1445)

## Testing Checklist

- [x] Save project from create_project page
- [ ] Save project from facilities page
- [ ] Save project from unit_parameters page
- [ ] Load project and verify create_project fields populate
- [ ] Load project and verify flow_scenarios fields populate
- [ ] Load project with static discharge
- [ ] Load project with hydrograph data
- [ ] Verify unit conversion on load (metric projects)
- [ ] Verify hydrograph textarea populates correctly
- [ ] Verify scenario_type dropdown shows correct selection

## Additional Changes

### Removed Auto-Timestamp from Project Filename
- Changed from `{project_name}_{timestamp}.stryke` to `{project_name}.stryke`
- Users can now manage their own versioning
- Browser will auto-append (1), (2), etc. for duplicates

## Known Limitations

1. Facilities page fields are not yet populated from loaded CSV (facilities.csv exists but not being read on GET)
2. Unit parameters page shows blank - need to implement reading from unit_params.csv
3. Population and operating scenarios pages not yet implemented for load
4. Graph editor not yet implemented for load

## Next Steps

To fully complete project load functionality:
1. Update `facilities` route to load existing facilities.csv on GET
2. Update facilities.html template to populate dynamic facility forms
3. Update `unit_parameters` route to display loaded unit_params.csv
4. Update population route to load population.csv
5. Update operating_scenarios route to load operating_scenarios.csv
6. Update graph_editor route to load graph.json
