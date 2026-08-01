import pandas as pd
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "src", "assets", "Electronics_Products.csv")

def prepare_csv(csv_path: str = DEFAULT_CSV_PATH):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Clean price column if needed and add product_id
    if "product_id" not in df.columns:
        print("Adding 'product_id' column...")
        df.insert(0, "product_id", range(1, len(df) + 1))
        df.to_csv(csv_path, index=False)
        print(f"Successfully added product_id (1 to {len(df)}) to {csv_path}")
    else:
        print(f"product_id already exists in {csv_path} with {len(df)} rows.")

    return df

if __name__ == "__main__":
    prepare_csv()
