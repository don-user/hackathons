from aiohttp.web import Response


async def index(request):
    return Response(
        text="""<h1>Цифровой прорыв 2024, Сочи</h1>
        <p>Кейс №3 - "Семантическая классификация документов"</p> 
        <a href='/docs'>Документация PRODOCS API</a>
        <p>Команда "ЛИФТ"</p> 
        """,
        content_type="text/html",
    )