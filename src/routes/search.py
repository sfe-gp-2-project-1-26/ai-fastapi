from fastapi import APIRouter, HTTPException, Query
from models.schemas import SearchQueryRequest, SearchQueryResponse
from helpers.vector_db import get_db_service

router = APIRouter(prefix="/api/v1", tags=["Product Search"])

@router.post("/search", response_model=SearchQueryResponse)
async def search_products(payload: SearchQueryRequest):
    """
    Accepts user query in natural language (Arabic or English) and returns 
    a ranked list of product_ids (1 to N) from Qdrant Hybrid Vector Search.
    """
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        db_service = get_db_service()
        product_ids = db_service.hybrid_search(
            query=payload.query.strip(),
            top_k=payload.top_k,
            min_price=payload.min_price,
            max_price=payload.max_price
        )

        return SearchQueryResponse(
            query=payload.query,
            total_found=len(product_ids),
            product_ids=product_ids
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
