from aiohttp.web_app import Application


def setup_routes(app: Application):
    from app.docs.routes import setup_routes as docs_setup_routes
    from app.web import views

    docs_setup_routes(app)
    app.router.add_get("/", views.index, name="home")
