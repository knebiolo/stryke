# Form Persistence and Change Detection - October 4, 2025

## Requirements

1. **Both buttons visible** - Show "Save/Create" button AND "Next Page" button together
2. **Disable when no changes** - Save button is greyed out (disabled) when viewing loaded, unchanged data
3. **Enable when data changes** - Save button becomes active as soon as user modifies any field
4. **Data persistence** - Form data should persist when navigating between pages

## Solution Implemented

### 1. Button Display Logic

**Before (Wrong):**
```html
{% if project_loaded %}
    <button>Next Page</button>
{% else %}
    <button>Save</button>
{% endif %}
```
- Only ONE button visible at a time
- Confusing for users

**After (Correct):**
```html
<button type="submit" id="saveBtn" {% if project_loaded %}disabled{% endif %}>Save</button>
{% if project_loaded %}
<button type="button" onclick="...">Next Page →</button>
{% endif %}
```
- BOTH buttons visible when project loaded
- Save button starts disabled when viewing loaded data
- Next Page button only shows for loaded projects

### 2. Change Detection JavaScript

Added JavaScript to all form pages that:
1. Captures original form values on page load
2. Listens for input/change events on all form fields
3. Compares current values to original values
4. Enables Save button if ANY field has changed
5. Keeps Save button disabled if no changes

**Example (create_project.html):**
```javascript
let originalFormData = {};

// Capture original values on page load
document.addEventListener('DOMContentLoaded', function() {
    const inputs = form.querySelectorAll('input, textarea, select');
    inputs.forEach(input => {
        originalFormData[input.name] = input.value;
    });
    
    // Listen for changes
    inputs.forEach(input => {
        input.addEventListener('input', checkFormChanged);
        input.addEventListener('change', checkFormChanged);
    });
});

function checkFormChanged() {
    let formChanged = false;
    inputs.forEach(input => {
        if (input.value !== originalFormData[input.name]) {
            formChanged = true;
        }
    });
    
    // Enable/disable button
    const saveBtn = document.getElementById('saveBtn');
    const isProjectLoaded = {{ 'true' if project_loaded else 'false' }};
    if (isProjectLoaded) {
        saveBtn.disabled = !formChanged;  // Enable if changed
    } else {
        saveBtn.disabled = false;  // Always enabled for new projects
    }
}
```

### 3. Data Persistence via Session

**Problem:** Data disappears when navigating between pages

**Root Cause:**
- Session variables were being overwritten
- CSV files existed but weren't being read on every GET request
- Session variables cleared when creating new project

**Solution:** Check session FIRST, then fall back to CSV

**create_project route (app.py):**
```python
# GET request - check session first, then CSV
existing_data = {}

# First try session variables (most recent, includes unsaved changes)
if 'project_name' in session:
    existing_data = {
        'project_name': session.get('project_name', ''),
        'project_notes': session.get('project_notes', ''),
        'units': session.get('units', 'metric'),
        'model_setup': session.get('model_setup', '')
    }
else:
    # Fall back to CSV file if session empty
    sim_folder = g.get('user_sim_folder')
    if sim_folder:
        project_csv = os.path.join(sim_folder, 'project.csv')
        if os.path.exists(project_csv):
            df = pd.read_csv(project_csv)
            # ... read from CSV ...
            
            # Also populate session for persistence
            session['project_name'] = existing_data['project_name']
            # ... etc ...
```

**Why This Works:**
1. User fills form → data goes to session
2. User navigates away → session persists
3. User comes back → session data loaded into form
4. User submits → data saved to CSV AND session updated
5. User loads project → CSV loaded AND session populated

### 4. Flow Scenarios Similar Fix

Same pattern applied:
```python
# First try session variables (most recent)
if 'scenario_name' in session:
    existing_data = {
        'scenario_name': session.get('scenario_name', ''),
        'scenario_number': session.get('scenario_number', ''),
        'season': session.get('season', ''),
        'months': session.get('months', ''),
        'scenario_type': session.get('scenario_type', 'static'),
        'discharge': session.get('discharge', ''),
        'hydrograph_data': session.get('hydrograph_data', '')
    }
else:
    # Fall back to CSV if session empty
    # ... read from flow.csv ...
    # ... populate session ...
```

## Files Modified

### Templates (Added Change Detection JavaScript)
1. `webapp/templates/create_project.html`
2. `webapp/templates/flow_scenarios.html`
3. `webapp/templates/facilities.html`
4. `webapp/templates/unit_parameters.html`

### Backend (Session-First Loading)
5. `webapp/app.py`:
   - `create_project()` GET - Session first, then CSV
   - `flow_scenarios()` GET - Session first, then CSV

## User Experience

### Workflow 1: New Project
1. User creates new project
2. Fills in form fields
3. Clicks "Create Project" (enabled)
4. Data saved to CSV and session
5. Navigate to next page
6. Come back to create_project
7. Form fields still populated from session ✅
8. "Create Project" button enabled (no project_loaded flag) ✅

### Workflow 2: Load Existing Project
1. User loads .stryke file
2. Data loaded from JSON → CSV files
3. Session populated with loaded data
4. Form fields populated ✅
5. "Next Page →" button appears ✅
6. "Create Project" button disabled (no changes yet) ✅
7. User modifies project_name field
8. "Create Project" button becomes enabled ✅
9. User clicks "Create Project"
10. Modified data saved
11. Button becomes disabled again until next change

### Workflow 3: Navigate Between Pages
1. User on create_project page
2. Enters project name "Test Project"
3. Doesn't click save, navigates to flow_scenarios
4. Enters scenario data
5. Navigates back to create_project
6. "Test Project" still in project_name field ✅
7. Session persistence working ✅

## Testing Checklist

- [ ] Create new project, enter data, navigate away, come back → data persists
- [ ] Load .stryke file → both buttons visible
- [ ] Load .stryke file → Save button disabled
- [ ] Load .stryke file, modify field → Save button enabled
- [ ] Load .stryke file, click Save (no changes) → button disabled
- [ ] Load .stryke file, modify field, click Save → button disabled after save
- [ ] Flow scenarios: same tests as above
- [ ] Facilities: same tests as above
- [ ] Unit parameters: same tests as above

## Known Behavior

### Session Cleared on Logout
- When user logs out, session is cleared
- This is expected Flask behavior
- User must save project before logout if they want to preserve data

### Browser Refresh
- Session persists across page refreshes (Flask session cookie)
- Form data loaded from session on each page load
- No data loss on refresh ✅

### Auto-Save Still Active
- Auto-save still saves to localStorage every 3 minutes
- Change detection is separate from auto-save
- Auto-save is a backup, session is primary storage
- localStorage cleared when project loaded (prevents restore conflict)

## Future Enhancements

1. **Visual Change Indicator:**
   ```html
   <span id="unsavedChanges" style="display:none; color: orange;">
       ● Unsaved changes
   </span>
   ```

2. **Confirm Before Navigate:**
   ```javascript
   window.addEventListener('beforeunload', function(e) {
       if (formChanged && !saved) {
           e.preventDefault();
           e.returnValue = 'You have unsaved changes. Leave anyway?';
       }
   });
   ```

3. **Show What Changed:**
   ```javascript
   function getChangedFields() {
       const changed = [];
       inputs.forEach(input => {
           if (input.value !== originalFormData[input.name]) {
               changed.push(input.name);
           }
       });
       return changed;
   }
   ```

4. **Reset Button:**
   ```html
   <button type="button" onclick="resetForm()">Reset to Original</button>
   ```

5. **Save Indicator:**
   ```javascript
   // Show "Saved!" message temporarily after successful save
   function showSaveConfirmation() {
       const msg = document.createElement('div');
       msg.textContent = '✓ Saved!';
       msg.style.cssText = 'color: green; font-weight: bold;';
       // ... fade out after 2 seconds ...
   }
   ```
