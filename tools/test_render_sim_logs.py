from importlib import import_module
import os

# Ensure the package path includes the workspace root
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if cwd not in sys.path:
    sys.path.insert(0, cwd)

os.environ['FLASK_ENV'] = 'development'

# Import the app
app_mod = import_module('webapp.app')
app = getattr(app_mod, 'app')

with app.test_client() as c:
    rv = c.get('/simulation_logs')
    print('STATUS', rv.status_code)
    data = rv.get_data(as_text=True)
    print('LEN', len(data))
    print('HAS BETA', 'Rare Event (Entrained)' in data or 'Rare Event (Mortality)' in data)
