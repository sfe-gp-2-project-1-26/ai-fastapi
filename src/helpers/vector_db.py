import os
import time
import pandas as pd
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

# Absolute Project Root Path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
QDRANT_STORAGE_PATH = os.path.join(BASE_DIR, "qdrant_db")
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "src", "assets", "Electronics_Products.csv")

COLLECTION_NAME = "electronics_products"
DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"

class VectorDBService:
    def __init__(self, storage_path: str = QDRANT_STORAGE_PATH):
        self.storage_path = storage_path
        print(f"[VectorDB] Connecting to local Qdrant at: {storage_path}")
        self.client = QdrantClient(path=storage_path)
        
        # Load embedding models
        print("[VectorDB] Loading Dense Embedding Model (BAAI/bge-small-en-v1.5)...")
        self.dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
        
        print("[VectorDB] Loading Sparse Embedding Model (Qdrant/bm25)...")
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    def init_collection(self, force_recreate: bool = False):
        """Initializes or resets the Qdrant collection for Dense + Sparse vectors."""
        collections = self.client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if exists and not force_recreate:
            print(f"[VectorDB] Collection '{COLLECTION_NAME}' already exists.")
            return

        if exists and force_recreate:
            self.client.delete_collection(COLLECTION_NAME)

        print(f"[VectorDB] Creating Qdrant Collection '{COLLECTION_NAME}'...")
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=384,  # BAAI/bge-small-en-v1.5 vector dimension
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            }
        )

    def build_product_string(self, row: pd.Series) -> str:
        """Concatenates product features into a single string for embedding."""
        pname = str(row.get("product_name", "")).strip()
        cat = str(row.get("category", "")).strip()
        about = str(row.get("about_product", "")).strip()
        
        single_string = f"Product: {pname} | Category: {cat} | Details: {about}"
        return single_string

    def parse_price(self, row: pd.Series) -> float:
        """Extracts discounted price if available, otherwise actual price."""
        discounted = str(row.get("discounted_price", "")).replace("$", "").replace(",", "").strip()
        actual = str(row.get("actual_price", "")).replace("$", "").replace(",", "").strip()
        
        for val in [discounted, actual]:
            try:
                if val:
                    return float(val)
            except ValueError:
                continue
        return 0.0

    def index_data(self, csv_path: str = DEFAULT_CSV_PATH, limit_rows: Optional[int] = None) -> Dict[str, float]:
        """
        Indexes products from CSV into Qdrant VDB with single-string embeddings and payload.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")

        df = pd.read_csv(csv_path)
        if limit_rows:
            df = df.head(limit_rows)
            print(f"[VectorDB] Indexing FIRST {limit_rows} products...")
        else:
            print(f"[VectorDB] Indexing ALL {len(df)} products...")

        start_time = time.perf_counter()

        documents = []
        payloads = []
        ids = []

        for _, row in df.iterrows():
            pid = int(row["product_id"])
            pstring = self.build_product_string(row)
            price = self.parse_price(row)

            documents.append(pstring)
            ids.append(pid)
            payloads.append({
                "product_id": pid,
                "price": price
            })

        # Embeddings Generation
        dense_embeddings = list(self.dense_model.embed(documents))
        sparse_embeddings = list(self.sparse_model.embed(documents))

        # Build Qdrant points
        points = []
        for pid, dense_vec, sparse_vec, payload in zip(ids, dense_embeddings, sparse_embeddings, payloads):
            points.append(
                models.PointStruct(
                    id=pid,
                    vector={
                        "dense": dense_vec.tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist()
                        )
                    },
                    payload=payload
                )
            )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time = total_time / len(points) if len(points) > 0 else 0.0

        print(f"[VectorDB] Completed indexing {len(points)} products in {total_time:.4f}s")

        return {
            "total_items": len(points),
            "total_time_seconds": total_time,
            "avg_time_per_insert_seconds": avg_time,
            "avg_time_per_insert_ms": avg_time * 1000
        }

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None
    ) -> List[int]:
        """
        Executes hybrid search (55% Dense + 45% Sparse) with price filtering.
        Returns ranked product_ids list.
        """
        query_dense = list(self.dense_model.embed([query]))[0].tolist()
        query_sparse_raw = list(self.sparse_model.embed([query]))[0]
        query_sparse = models.SparseVector(
            indices=query_sparse_raw.indices.tolist(),
            values=query_sparse_raw.values.tolist()
        )

        filter_conditions = []
        if min_price is not None or max_price is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="price",
                    range=models.Range(
                        gte=min_price,
                        lte=max_price
                    )
                )
            )

        query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

        search_results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=top_k * 2,
                    filter=query_filter
                ),
                models.Prefetch(
                    query=query_sparse,
                    using="sparse",
                    limit=top_k * 2,
                    filter=query_filter
                ),
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=top_k
        )

        product_ids = [hit.payload["product_id"] for hit in search_results.points]
        return product_ids

_db_service_instance = None

def get_db_service() -> VectorDBService:
    global _db_service_instance
    if _db_service_instance is None:
        _db_service_instance = VectorDBService()
    return _db_service_instance
