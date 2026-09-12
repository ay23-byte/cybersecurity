from starlette.templating import Jinja2Templates
from starlette.requests import Request
import pandas as pd
from types import SimpleNamespace

templates = Jinja2Templates(directory="templates")

df = pd.read_csv('valid_data/test.csv')
table_html = df.to_html(classes='table table-striped')

# Create a fake request-like object for rendering
fake_request = SimpleNamespace()

resp = templates.TemplateResponse(fake_request, 'table.html', {'table': table_html})
print(resp.body.decode())
