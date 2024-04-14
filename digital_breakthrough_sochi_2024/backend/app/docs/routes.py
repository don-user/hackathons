import typing

from app.docs.views import (
    DocsAddView,
    DocsListView
)

if typing.TYPE_CHECKING:
    from app.web.app import Application


def setup_routes(app: "Application"):
    app.router.add_view("/docs.add", DocsAddView)
    app.router.add_view("/docs.list", DocsListView)
