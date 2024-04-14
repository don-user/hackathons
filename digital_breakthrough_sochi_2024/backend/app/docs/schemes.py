from marshmallow import Schema, fields


class DocsSchema(Schema):
    id = fields.Int()
    time_create = fields.DateTime(required=True)
    filename = fields.Str(required=True)
    content = fields.Str(required=True)
    label = fields.Str(required=True)


class DocsRequestSchema(Schema):
    filename = fields.Str(required=True)
    uploaded_file = fields.Raw(
        metadata={"type": "file", "description": "Загрузка файла"}, required=True
    )


class DocsListResponseSchema(Schema):
    files = fields.Nested(DocsSchema, many=True)
