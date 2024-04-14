import aspose.words as aw


def GetTextContract(path):

    data = aw.Document(path)
    contract = data.get_text()
    contract = contract.replace(
        "Evaluation Only. Created with Aspose.Words. Copyright 2003-2024 Aspose Pty Ltd.",
        "",
    )
    contract = contract.replace(
        "Created with an evaluation copy of Aspose.Words. To discover the full versions of our APIs please visit: https://products.aspose.com/words/",
        "",
    )

    return contract
