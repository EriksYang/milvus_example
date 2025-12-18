from embedding_common import get_bge_m3_embedding
from milvus_crud import MilvusClient


class MilvusBgeClient(MilvusClient):

    # -----------------------------
    # Insert（文本 → embedding → Milvus）
    # -----------------------------
    def insert_texts(self, ids, texts):
        vectors = get_bge_m3_embedding(texts)
        self.insert(ids=ids, vectors=vectors, texts=texts)

    # -----------------------------
    # Search（query → embedding → Milvus）
    # -----------------------------
    def search_text(
        self,
        query_text,
        top_k=5,
        expr=None
    ):
        query_vector = get_bge_m3_embedding([query_text])[0]
        return self.search(
            query_vectors=[query_vector],
            top_k=top_k,
            expr=expr
        )
