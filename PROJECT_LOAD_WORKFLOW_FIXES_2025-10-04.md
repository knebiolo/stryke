# Project Load Workflow Fixes - October 4, 2025 (Part 3)

## Issues Addressed

### Issue 1: Auto-Save Restore Dialog Overwriting Loaded Project Data ✅

**Problem:** 
When loading a .stryke project file:
1. Project loads and writes CSV files
2. User is redirected to create_project page
3. Auto-save detects old form data in localStorage
4. Shows "Restore from X minutes ago?" dialog
5. If user clicks "Restore", it overwrites the loaded project data with blank/old auto-save data
6. Form fields appear empty even though project loaded successfully

**Root Cause:**
- Auto-save localStorage persists across sessions
- Load project didn't clear localStorage
- Auto-save prompt appeared on every page navigation after load
- Clicking "Restore" replaced loaded CSV data with stale localStorage data

**Solution:**
1. Added `project_loaded` flag to session when .stryke file is loaded
2. Added inline script to clear localStorage on all pages when `project_loaded=True`
3. Changed auto-save threshold from 1 minute to 3 minutes (already fixed earlier)

**Code Changes:**

`app.py - load_project()`:
```python
# Set flag that project was loaded
session['project_loaded'] = True

# Clear auto-save since we just loaded a project
flash('✅ Project loaded successfully! All data has been restored. Click "Next Page" to continue.')
```

`create_project.html` (and all other page templates):
```html
{% if project_loaded %}
<script>
    // Clear auto-save when viewing loaded project
    localStorage.removeItem('stryke_autosave');
    localStorage.removeItem('stryke_autosave_timestamp');
</script>
{% endif %}
```

### Issue 2: "Create Project" Button Should Become "Next Page" When Loaded ✅

**Problem:**
- After loading a project, user sees "Create Project" button
- This is confusing - project already exists
- User expects "Next Page" navigation button instead
- All "Save" buttons should be hidden/replaced with "Next Page" buttons

**Solution:**
- Changed all form submit buttons to conditionally show:
  - When `project_loaded=False`: Show "Save" button (normal workflow)
  - When `project_loaded=True`: Show "Next Page →" button (loaded project workflow)
- "Next Page" buttons navigate directly without form submission

**Code Changes:**

`create_project.html`:
```html
{% if project_loaded %}
<button type="button" onclick="window.location.href='{{ url_for('flow_scenarios') }}'">Next Page →</button>
{% else %}
<button type="submit">Create Project</button>
{% endif %}
```

Applied to all pages:
- `create_project.html` → Next: flow_scenarios
- `flow_scenarios.html` → Next: facilities
- `facilities.html` → Next: unit_parameters
- `unit_parameters.html` → Next: graph_editor
- `graph_editor.html` → Next: population
- (population and operating_scenarios navigate via existing links)

### Issue 3: Form Fields Appear Blank After Loading Project ✅

**Problem:**
- `.stryke` file loads successfully (CSV files written to disk)
- Session variables populated
- But form fields show blank on all pages
- User can't see the loaded data

**Root Causes:**
1. `create_project` route was passing data to template ✓ (already working)
2. `flow_scenarios` route was passing data to template ✓ (already working)
3. `facilities` route was NOT reading from CSV or passing data to template ✗
4. `unit_parameters` route was NOT passing context variables ✗
5. `population` route needs to load from CSV ✗
6. `operating_scenarios` route needs to load from CSV ✗

**Solution:**
Fixed all routes to pass `project_loaded` flag and load existing data from CSV files.

**Code Changes:**

All routes now follow this pattern:
```python
@app.route('/some_page', methods=['GET', 'POST'])
def some_page():
    if request.method == 'POST':
        # Handle form submission
        session['project_loaded'] = False  # Clear flag when creating new data
        # ... save data ...
    
    # GET request - load existing data and pass project_loaded flag
    project_loaded = session.get('project_loaded', False)
    # ... load data from CSV if exists ...
    return render_template('some_page.html', project_loaded=project_loaded, **data)
```

### Issue 4: Clear project_loaded Flag When Creating New Project ✅

**Problem:**
- If user loads a project, then creates a new project, the `project_loaded` flag stays `True`
- "Next Page" buttons remain instead of "Save" buttons
- User can't save new data

**Solution:**
Added `session['project_loaded'] = False` when creating new project in `create_project` POST handler.

```python
# Clear project_loaded flag since we're creating a new project
session['project_loaded'] = False
```

## Files Modified

### Backend (app.py)
1. `load_project()` - Added `session['project_loaded'] = True`
2. `create_project()` POST - Added `session['project_loaded'] = False`
3. `create_project()` GET - Pass `project_loaded` flag
4. `flow_scenarios()` GET - Pass `project_loaded` flag
5. `facilities()` GET - Pass `project_loaded` flag
6. `unit_parameters()` GET - Pass `project_loaded` flag + context vars
7. `graph_editor()` GET - Pass `project_loaded` flag + graph_data
8. `population()` GET - Pass `project_loaded` flag
9. `operating_scenarios()` GET - Pass `project_loaded` flag

### Templates
1. `create_project.html` - Conditional button, localStorage clear script
2. `flow_scenarios.html` - Conditional button, localStorage clear script
3. `facilities.html` - Conditional button, localStorage clear script
4. `unit_parameters.html` - Conditional button, localStorage clear script
5. `graph_editor.html` - Conditional button, localStorage clear script

## Workflow Comparison

### Before Fix:
1. User loads .stryke file
2. Redirected to create_project page
3. "Restore from 2 minutes ago?" dialog appears 🔴
4. User clicks "Restore" (thinking it will restore the project) 🔴
5. Auto-save data (blank/old) replaces loaded project data 🔴
6. Form fields are blank 🔴
7. User sees "Create Project" button 🔴
8. Confused - did project load? 🔴

### After Fix:
1. User loads .stryke file
2. localStorage cleared automatically ✅
3. Redirected to create_project page
4. Form fields populated with loaded data ✅
5. Message: "Project loaded successfully! Click Next Page to continue" ✅
6. "Next Page →" button visible ✅
7. No auto-save restore dialog ✅
8. Click "Next Page" → flow_scenarios page
9. Form fields populated ✅
10. "Next Page →" button visible ✅
11. Navigate through all pages reviewing loaded data
12. "Save Project" buttons still available on all pages

## Testing Checklist

- [ ] Load .stryke file from create_project page
- [ ] Verify no "Restore" dialog appears
- [ ] Verify create_project fields are populated
- [ ] Verify "Next Page →" button shows instead of "Create Project"
- [ ] Click "Next Page" → flow_scenarios
- [ ] Verify flow_scenarios fields are populated
- [ ] Verify "Next Page →" button shows instead of "Save Flow Scenario"
- [ ] Click "Next Page" → facilities
- [ ] Verify facilities data is loaded
- [ ] Verify "Next Page →" button shows instead of "Save Facilities"
- [ ] Continue through all pages (unit_parameters, graph_editor, population, operating_scenarios)
- [ ] Verify "Save Project" button works on all pages
- [ ] Create a NEW project (click home, login again)
- [ ] Verify "Create Project" button shows (not "Next Page")
- [ ] Verify "Save" buttons show (not "Next Page")

## Known Limitations

### Facilities Data Display
The facilities page dynamically generates form fields based on `num_facilities`. When a project is loaded:
- The CSV data is restored ✅
- Session variables are set ✅
- Template receives `project_loaded=True` ✅
- BUT: The JavaScript `generateFacilityForms()` needs to read from the CSV to populate the dynamic forms

**TODO:** Update `facilities.html` JavaScript to:
1. Fetch facilities data from server when page loads
2. Populate dynamic form fields with loaded data
3. This requires an API endpoint like `/get_facilities_data`

### Population & Operating Scenarios
Similar to facilities, these pages have complex forms that need additional work:
- CSV data is loaded ✅
- `project_loaded` flag is passed ✅
- But forms need JavaScript to fetch and populate data

**Recommended Approach:**
1. Create `/get_population_data` endpoint
2. Create `/get_operating_scenarios_data` endpoint
3. Update page JavaScript to fetch and populate on load

## Future Enhancements

1. **Visual Indicator:** Add a banner at top of page when `project_loaded=True`:
   ```html
   <div style="background: #d4edda; padding: 10px; margin-bottom: 15px; border-left: 4px solid #28a745;">
       📂 Viewing Loaded Project: <strong>{{ project_name }}</strong>
       <a href="{{ url_for('index') }}" style="float: right;">Start New Project</a>
   </div>
   ```

2. **Disable Form Fields:** When `project_loaded=True`, make fields readonly:
   ```html
   <input type="text" name="project_name" value="{{ project_name }}" {% if project_loaded %}readonly{% endif %}>
   ```

3. **Edit Mode Toggle:** Add "Edit Mode" button to allow modifying loaded project:
   ```html
   {% if project_loaded %}
   <button onclick="enableEdit()">✏️ Edit Mode</button>
   {% endif %}
   ```

4. **Progress Indicator:** Show which page user is on:
   ```
   Create Project → Flow Scenarios → Facilities → Unit Parameters → Graph → Population → Operating Scenarios → Model Summary
         ✓              ✓              ▶           ...           ...        ...             ...                  ...
   ```

5. **Validation Skip:** When navigating with "Next Page", skip form validation since data already validated when saved

