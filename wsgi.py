from app import app

# WSGI application for gunicorn
application = app

# Also provide the app object for ASGI
asgi_app = app
