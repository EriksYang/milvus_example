from milvus_bge_client import MilvusBgeClient

client = MilvusBgeClient(
    collection_name="intent_bge_m3",
    dim=1024
)

# 1️⃣ 插入意图文本
intent_texts = [
    "查询天气",
    "播放音乐",
    "讲一个笑话",
    "查询航班信息",
    "打开应用"
]

client.insert_texts(
    ids=list(range(1, len(intent_texts) + 1)),
    texts=intent_texts
)

# 2️⃣ 用户 query
query = "明天北京天气怎么样？"

result = client.search_text(
    query_text=query,
    top_k=3
)

print("Search result:")
for r in result:
    print(r)
