from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)


class MilvusClient:
    def __init__(
            self,
            host="127.0.0.1",
            port="19530",
            collection_name="demo_collection",
            dim=1024
    ):
        self.collection_name = collection_name
        self.dim = dim

        connections.connect(
            alias="default",
            host=host,
            port=port
        )

        if not utility.has_collection(collection_name):
            self._create_collection()

        self.collection = Collection(collection_name)
        self.collection.load()

    # -----------------------------
    # 创建集合
    # -----------------------------
    def _create_collection(self):
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dim
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=512
            )
        ]

        schema = CollectionSchema(
            fields=fields,
            description="CRUD demo collection"
        )

        collection = Collection(
            name=self.collection_name,
            schema=schema
        )

        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200}
        }

        collection.create_index(
            field_name="vector",
            index_params=index_params
        )

        print(f"Collection `{self.collection_name}` created.")

    # -----------------------------
    # CREATE / INSERT
    # -----------------------------
    def insert(self, ids, vectors, texts):
        """
        ids: List[int]
        vectors: List[List[float]]
        texts: List[str]
        """
        data = [ids, vectors, texts]
        self.collection.insert(data)
        self.collection.flush()

    # -----------------------------
    # READ / SEARCH
    # -----------------------------
    def search(
            self,
            query_vectors,
            top_k=5,
            expr=None
    ):
        """
        query_vectors: List[List[float]]
        """
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }

        results = self.collection.search(
            data=query_vectors,
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text"]
        )

        hits = []
        for hit in results[0]:
            hits.append({
                "id": hit.id,
                "score": hit.score,
                "text": hit.entity.get("text")
            })

        return hits

    # -----------------------------
    # UPDATE（本质是 delete + insert）
    # -----------------------------
    def update(self, id_, new_vector, new_text):
        self.delete(f"id == {id_}")
        self.insert(
            ids=[id_],
            vectors=[new_vector],
            texts=[new_text]
        )

    # -----------------------------
    # DELETE
    # -----------------------------
    def delete(self, expr):
        """
        expr: e.g. "id in [1,2,3]"
        """
        self.collection.delete(expr)
        self.collection.flush()

    # -----------------------------
    # QUERY（非向量查询）
    # -----------------------------
    def query(self, expr, output_fields=None):
        return self.collection.query(
            expr=expr,
            output_fields=output_fields or ["id", "text"]
        )

    # -----------------------------
    # DROP COLLECTION
    # -----------------------------
    def drop(self):
        utility.drop_collection(self.collection_name)
