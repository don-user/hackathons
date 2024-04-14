import json
import os
import pickle
import pandas as pd
import re
from typing import List
from aiohttp.web import HTTPConflict
from aiohttp_apispec import (
    docs,
    request_schema,
    response_schema,
    form_schema,
)
from aiohttp.web_response import Response
from sqlalchemy import exc
from pymystem3 import Mystem

from app.web.app import View
from app.web.utils import json_response
from app.docs.schemes import (
    DocsSchema,
    DocsRequestSchema,
    DocsListResponseSchema,
)
from app.docs.models import DocsBase
from app.docs.utils import GetTextContract

with open('app/pkl_object/labels.json', 'r', encoding='utf8') as f:
    kind_names = json.load(f)

with open('app/pkl_object/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    
with open('app/pkl_object/logit_grid_searcher.pkl', 'rb') as f:
    logit_grid_searcher = pickle.load(f)

MAX_WORD = 1000

m = Mystem() 

def ContractTransform(contract):

    contract = contract.lower()
    contract = re.sub('[^а-яА-ЯёЁ]', ' ', contract)
    lemm_text_list = [i for i in m.lemmatize(contract) if len(i.strip()) > 2]
    lemm_text_list = lemm_text_list[:MAX_WORD]
    
    return pd.Series(' '.join(lemm_text_list))

def ModelPredictProba(contract, kind_names, vectorizer, logit_grid_searcher):
    
    X = vectorizer.transform(contract)
    kind_name_pred = kind_names[str(logit_grid_searcher.predict(X)[0])]
        
    return kind_name_pred

class DocsAddView(View):
    # @form_schema(DocsRequestSchema, put_into=None) # , locations=["files"]
    @response_schema(DocsSchema, 200)
    @docs(
        tags=["docs"],
        summary="Add files add view",
        description="Add file to database",
        consumes=["multipart/form-data"],
    )
    async def post(self) -> Response:
        reader = await self.request.multipart()

        field = await reader.next()
        assert field.name == "filename"
        name = await field.read(decode=True)

        filename = name.decode("utf-8")

        field = await reader.next()
        assert field.name == "uploaded_file"

        size = 0
        with open(os.path.join("../storage/", filename), "wb") as f:
            while True:
                chunk = await field.read_chunk()  # 8192 bytes by default.
                if not chunk:
                    break
                size += len(chunk)
                f.write(chunk)

        content = GetTextContract("../storage/" + filename)

        contract = ContractTransform(content)
        kind_name_pred = ModelPredictProba(contract, kind_names, vectorizer, logit_grid_searcher)

        file = await self.store.files.create_doc(
            filename=filename, content=content, label=kind_name_pred
        )

        return json_response(data=DocsSchema().dump(file))


class DocsListView(View):
    @docs(
        tags=["docs"],
        summary="Add files list view",
        description="Get list files from database",
    )
    @response_schema(DocsListResponseSchema, 200)
    async def get(self) -> Response:
        files: List[DocsBase] = await self.store.files.list_docs()
        return json_response(DocsListResponseSchema().dump({"files": files}))
