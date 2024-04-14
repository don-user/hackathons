from typing import Optional
from aiohttp.web import (
    Application as AiohttpApplication,
    View as AiohttpView,
    Request as AiohttpRequest,
)

from app.store import setup_store, Store
from app.store.database.database import Database
from app.web.config import Config, setup_config
from app.web.logger import setup_logging
from app.web.middlewares import setup_middlewares
from app.web.routes import setup_routes

from aiohttp_session import setup as session_setup
from aiohttp_session.cookie_storage import EncryptedCookieStorage

from aiohttp_apispec import setup_aiohttp_apispec


class Application(AiohttpApplication):
    config: Optional[Config] = None
    store: Optional[Store] = None
    database: Optional[Database] = None


class Request(AiohttpRequest):

    @property
    def app(self) -> Application:
        return super().app()


class View(AiohttpView):
    @property
    def request(self) -> Request:
        return super().request

    @property
    def database(self):
        return self.request.app.database

    @property
    def store(self) -> Store:
        return self.request.app.store

    @property
    def data(self) -> dict:
        return self.request.get("data", {})


app = Application()

description = """

PRODOCS API

Разработанное решение позволяет, с применением технологий искусственного интеллекта,
определять типа документа с максимально возможной точностью.

При разработке решения использовались открытые программные средства и технологии.

Для возможности внедрения и взаимодействия с моделью дополнительно разработан UI - клиент (web-сервис)
и REST API. Данный подход позволяет легко и быстро масштабировать решение.
С целью удобства запуск сервиса осуществляется с помощью Docker

Основным языком программирования является Python.


**ЛИФТ**

Цифровой прорыв 2024, Сочи

"""


def setup_app(config_path: str) -> Application:
    setup_logging(app)
    setup_config(app, config_path)
    session_setup(
        app, storage=EncryptedCookieStorage(secret_key=app.config.session.key)
    )
    setup_aiohttp_apispec(
        app,
        title="Цифровой прорыв 2024, Сочи",
        version="0.0.1",
        swagger_path="/docs",
        url="/docs/json",
        info=dict(
            description=description,
            contact={
                "name": "Руслан Латипов",
                "url": "https://t.me/rus_lat116",
                "email": "rus_kadr03@mail.ru",
            },
        ),
    )
    setup_routes(app)
    setup_middlewares(app)
    setup_store(app)

    return app
