import chromadb
chroma_client=chromadb.Client()

collection=chroma_client.create_collection(name="mycollections")

collection.upsert(
    documents=[
        "Tech is shaping our lives.",
        "Technology impacts daily life.",
        "Tech transforms our world.",
        "Technology drives progress"
    ],
    ids=["id1","id2","id3","id4"]
)

result=collection.query(
    query_texts=["The text is about Technology"],
    n_results=4
)

print(result)