from jinja2 import Environment, FileSystemLoader, select_autoescape
import os

tpl_dir = os.path.join(os.path.dirname(__file__), '..', 'webapp', 'templates')
env = Environment(loader=FileSystemLoader(tpl_dir), autoescape=select_autoescape(['html','xml']))
def url_for(endpoint, **kwargs):
    # minimal stub: return a plausible static path for testing
    if endpoint == 'static' and 'filename' in kwargs:
        return f"/static/{kwargs['filename']}"
    return f"/{endpoint}"

env.globals['url_for'] = url_for

tpl = env.get_template('create_project.html')
for val in (True, False):
    out = tpl.render(project_loaded=val, project_name='Test', project_notes='x', units='metric', model_setup='single_unit_simulated_entrainment')
    print('\n--- project_loaded =', val, '---')
    print('Has enable-on-change include?', 'enable-on-change.js' in out)
    print('Button has data-enable-on-change?', 'data-enable-on-change="true"' in out)
    print('Button disabled present?', 'disabled' in out)
    # print small snippet
    start = out.find('<button type="submit"')
    print(out[start:start+200])
