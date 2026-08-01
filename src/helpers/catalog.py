import os
import pandas as pd
from typing import Dict, List, Any, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "src", "assets", "Electronics_Products.csv")

class CatalogService:
    def __init__(self, csv_path: str = DEFAULT_CSV_PATH):
        self.csv_path = csv_path
        self.catalog: Dict[int, Dict[str, Any]] = {}
        self.load_catalog()

    def load_catalog(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Catalog CSV not found at {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        print(f"[Catalog] Loading {len(df)} products into in-memory catalog...")
        
        for _, row in df.iterrows():
            pid = int(row["product_id"])
            pname = str(row.get("product_name", "")).strip()
            cat = str(row.get("category", "")).strip()
            about = str(row.get("about_product", "")).strip()
            img = str(row.get("img_link", "")).strip()
            
            # Clean prices
            discounted_str = str(row.get("discounted_price", "")).strip()
            actual_str = str(row.get("actual_price", "")).strip()
            discount_pct = str(row.get("discount_percentage", "")).strip()

            self.catalog[pid] = {
                "product_id": pid,
                "product_name": pname,
                "category": cat,
                "discounted_price": discounted_str,
                "actual_price": actual_str,
                "discount_percentage": discount_pct,
                "about_product": about,
                "img_link": img,
                "product_link": f"http://localhost:5174/products/{pid}"
            }
        print(f"[Catalog] Catalog loaded successfully with {len(self.catalog)} items.")

    def get_product(self, product_id: int) -> Optional[Dict[str, Any]]:
        return self.catalog.get(product_id)

    def format_products_context(self, product_ids: List[int]) -> str:
        """Formats a list of product_ids into a clean context string for Gemini prompt."""
        if not product_ids:
            return "No matching products found in catalog."

        context_lines = []
        for pid in product_ids:
            item = self.get_product(pid)
            if item:
                line = (
                    f"• Product ID: {item['product_id']}\n"
                    f"  Name: {item['product_name']}\n"
                    f"  Price: {item['discounted_price']} (Original: {item['actual_price']}, Discount: {item['discount_percentage']})\n"
                    f"  Category: {item['category']}\n"
                    f"  Product Link: http://localhost:5174/products/{item['product_id']}\n"
                    f"  Details: {item['about_product'][:300]}..."
                )
                context_lines.append(line)

        return "\n\n".join(context_lines)

_catalog_instance = None

def get_catalog_service() -> CatalogService:
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = CatalogService()
    return _catalog_instance
