import typing
from typing import List, Optional
from sqlalchemy import select, update, delete

from app.docs.models import DocsBase, DocsModel
from app.base.base_accessor import BaseAccessor


class DocsAccessor(BaseAccessor):
    async def get_by_doc(self, filename: str) -> Optional[DocsBase]:
        async with self.app.database.session() as session:
            query = select(DocsModel).where(DocsModel.filename == filename)
            file: Optional[DocsModel] = await session.scalar(query)

        if not file:
            return None

        return DocsBase(
            id=file.id,
            time_create=file.time_create,
            filename=file.filename,
            content=file.content,
            label=file.label,
        )

    async def get_by_doc_id(self, id: int) -> Optional[DocsBase]:
        async with self.app.database.session() as session:
            query = select(DocsModel).where(DocsModel.id == id)
            file: Optional[DocsModel] = await session.scalar(query)

        if not file:
            return None

        return DocsBase(
            id=file.id,
            time_create=file.time_create,
            filename=file.filename,
            content=file.content,
            label=file.label,
        )

    async def create_doc(
        self, filename: str, content: str, label: str
    ) -> Optional[DocsBase]:
        new_file: DocsBase = DocsModel(filename=filename, content=content, label=label)

        async with self.app.database.session.begin() as session:
            session.add(new_file)

        return DocsBase(
            id=new_file.id,
            time_create=new_file.time_create,
            filename=new_file.filename,
            content=new_file.content,
            label=new_file.label,
        )

    async def delete_doc(self, filename: str) -> Optional[DocsBase]:
        query = (
            delete(DocsModel).where(DocsModel.filename == filename).returning(DocsModel)
        )

        async with self.app.database.session.begin() as session:
            file: Optional[DocsModel] = await session.scalar(query)

        if not file:
            return None

        return DocsBase(
            id=file.id,
            time_create=file.time_create,
            filename=file.filename,
            content=file.content,
            label=file.label,
        )

    async def list_docs(self) -> List[Optional[DocsBase]]:
        query = select(DocsModel)

        async with self.app.database.session() as session:
            files: List[Optional[DocsModel]] = await session.scalars(query)

        if not files:
            return []

        return [
            DocsBase(
                id=file.id,
                time_create=file.time_create,
                filename=file.filename,
                content=file.content,
                label=file.label,
            )
            for file in files.all()
        ]
