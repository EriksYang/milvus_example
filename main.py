import random

from milvus_crud import MilvusClient

DIM = 1024


def random_vector():
    return [random.random() for _ in range(DIM)]


client = MilvusClient(
    collection_name="intent_demo",
    dim=DIM
)

# 1️⃣ Insert
# client.insert(
#     ids=[1, 2, 3],
#     vectors=[random_vector(), random_vector(), random_vector()],
#     texts=["查询天气", "播放音乐", "讲一个笑话"]
# )

# 2️⃣ Search
res = client.search(
    query_vectors=[random_vector()],
    top_k=3
)
print("Search result:", res)

# 3️⃣ Query
rows = client.query("id >= 2")
print("Query result:", rows)

# 4️⃣ Update
# client.update(
#     id_=1,
#     new_vector=random_vector(),
#     new_text="查询明天天气"
# )

# 5️⃣ Delete
# client.delete("id == 3")
