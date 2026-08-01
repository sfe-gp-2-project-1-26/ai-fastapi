from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routes.search import router as search_router
from routes.chat import router as chat_router
from helpers.vector_db import get_db_service
from helpers.catalog import get_catalog_service
from helpers.gemini_service import get_gemini_manager
from helpers.prepare_data import prepare_csv

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure product_id column exists
    print("[Startup] Preparing CSV data...")
    prepare_csv()
    
    print("[Startup] Loading CSV Catalog into memory...")
    get_catalog_service()

    print("[Startup] Connecting to Local Qdrant Vector DB...")
    db_service = get_db_service()
    db_service.init_collection(force_recreate=False)
    
    # Auto-index if collection is empty
    count = db_service.client.count(collection_name="electronics_products").count
    if count == 0:
        print("[Startup] Qdrant collection is empty! Indexing all products...")
        db_service.index_data(limit_rows=None)
    else:
        print(f"[Startup] Qdrant collection ready with {count} indexed products!")
    
    print("[Startup] Initializing Gemini Chat Manager...")
    get_gemini_manager()

    yield
    
    print("[Shutdown] Server shutting down...")

app = FastAPI(
    title="Electronics Products Hybrid Search & Chatbot API",
    description="FastAPI Backend for Qdrant Hybrid Search and Multi-turn Gemini AI Chatbot.",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend/team integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(search_router)
app.include_router(chat_router)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Electronics Products Chatbot & Search API",
        "endpoints": {
            "search": "POST /api/v1/search",
            "chat": "POST /api/v1/chat"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
