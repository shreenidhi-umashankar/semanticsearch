import time
import json
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load documents
with open("documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

doc_texts = [doc["content"] for doc in documents]

# Load local embedding model (FREE)
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating document embeddings...")
doc_embeddings = model.encode(doc_texts)
doc_embeddings = np.array(doc_embeddings).astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(doc_embeddings)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings)

print("Embeddings ready!")

class SearchRequest(BaseModel):
    query: str
    k: int = 6
    rerank: bool = True
    rerankK: int = 4

@app.post("/search")
def search(req: SearchRequest):

    start_time = time.time()

    # Encode query
    query_embedding = model.encode([req.query])
    query_vector = np.array(query_embedding).astype("float32")

    faiss.normalize_L2(query_vector)

    similarities, indices = index.search(query_vector, req.k)

    results = []

    for i, idx in enumerate(indices[0]):
        score = float(similarities[0][i])
        score = max(0, min(1, score))

        results.append({
            "id": documents[idx]["id"],
            "score": score,
            "content": documents[idx]["content"],
            "metadata": documents[idx]["metadata"]
        })

    # Sort descending
    results.sort(key=lambda x: x["score"], reverse=True)

    # Return top rerankK
    results = results[:req.rerankK]

    latency = int((time.time() - start_time) * 1000)

    return {
        "results": results,
        "reranked": False,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
