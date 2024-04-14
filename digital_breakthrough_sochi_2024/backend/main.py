import os
from aiohttp.web import run_app

from app.web.app import setup_app


if __name__ == "__main__":
    run_app(
        setup_app(
            config_path=os.path.join(
                os.path.dirname(__file__),
                "config.yml",
            )
        )
    )


# import asyncio
# import os
# import aiofiles
# from aiohttp import web


# app = web.Application()


# async def store_mp3_handler(request):

#     # WARNING: don't do that if you plan to receive large files!
#     # data = await request.post()

#     # mp3 = data['uploaded_file']

#     # # .filename contains the name of the file in string format.
#     # filename = data['filename'] # mp3.filename

#     # print(filename)
#     # # .file contains the actual file data that needs to be stored somewhere.
#     # mp3_file = mp3.file

#     # content = mp3_file.read()

#     # # asyncio.gather(
#     # #             save_file(content, filename),
#     # #         )

#     # with open(f"./backend/{filename}", "wb+") as file_object:
#     #     file_object.write(content)

#     # return web.Response(text='Ok')

#     reader = await request.multipart()

#     # /!\ Don't forget to validate your inputs /!\

#     # reader.next() will `yield` the fields of your form

#     field = await reader.next()
#     assert field.name == 'filename'
#     name = await field.read(decode=True)

#     filename = name.decode("utf-8")

#     field = await reader.next()
#     assert field.name == 'uploaded_file'

#     # You cannot rely on Content-Length if transfer is chunked.
#     size = 0
#     with open(os.path.join('./storage/', filename), 'wb') as f:
#         while True:
#             chunk = await field.read_chunk()  # 8192 bytes by default.
#             if not chunk:
#                 break
#             size += len(chunk)
#             f.write(chunk)

#     return web.Response(text='{} sized of {} successfully stored'
#                                 ''.format(filename, size))


# app.add_routes([web.post('/file/upload', store_mp3_handler)])

# web.run_app(app)
