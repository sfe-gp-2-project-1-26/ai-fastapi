from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse
from helpers.vector_db import get_db_service
from helpers.catalog import get_catalog_service
from helpers.gemini_service import get_gemini_manager

router = APIRouter(prefix="/api/v1", tags=["Multi-Turn Chatbot"])

@router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(payload: ChatRequest):
    """
    Multi-turn Chat endpoint using Gemini HyDE Pre-Processing + Qdrant Hybrid Search.
    1. Extracts min/max price & generates HyDE query.
    2. Performs price-filtered Vector DB hybrid search.
    3. Streams to Gemini for natural multi-turn response.
    """
    if not payload.message or not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message string cannot be empty.")

    try:
        user_msg = payload.message.strip()
        gemini_manager = get_gemini_manager()

        # 1. HyDE & Price Filter Extraction via Gemini
        hyde_parsed = gemini_manager.parse_query_hyde(user_msg)
        hyde_query = hyde_parsed.get("hyde_query", user_msg)
        extracted_min_price = hyde_parsed.get("min_price")
        extracted_max_price = hyde_parsed.get("max_price")

        print(f"[HyDE] Generated HyDE Query: '{hyde_query[:80]}...'")
        print(f"[HyDE] Extracted Price Filters -> Min: {extracted_min_price}, Max: {extracted_max_price}")

        # 2. Hybrid Search in Vector DB using HyDE Query + Extracted Price Filters
        db_service = get_db_service()
        product_ids = db_service.hybrid_search(
            query=hyde_query,
            top_k=payload.top_k,
            min_price=extracted_min_price,
            max_price=extracted_max_price
        )

        # 3. Format Context String from CSV Catalog
        catalog_service = get_catalog_service()
        products_context = catalog_service.format_products_context(product_ids)

        # 4. Multi-Turn Session Response Generation via Gemini
        session_id, reply = gemini_manager.send_rag_message(
            user_message=user_msg,
            products_context=products_context,
            session_id=payload.session_id
        )

        return ChatResponse(
            session_id=session_id,
            reply=reply,
            recommended_product_ids=product_ids
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")
