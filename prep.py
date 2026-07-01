import os
import json
import time
import numpy as np
import faiss
from openai import OpenAI
from shl_recommender.core.config import settings
from shl_recommender.core.constants import CATEGORY_MAP

def main():
    # 1. Loading the catalog file
    # Allow reading from environment or fallback to standard filenames
    catalog_file = "shl_product_catalog.json"
    if not os.path.exists(catalog_file):
        catalog_file = "catalogue.json"
        
    print(f"Loading catalog from {catalog_file}...")
    
    try:
        with open(catalog_file, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Failed to read catalog file: {e}")
        return

    try:
        # Use strict=False to allow control characters like newlines in JSON strings
        catalog_data = json.loads(text, strict=False)
    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e.msg} at line {e.lineno}, col {e.colno}, char {e.pos}")
        start = max(0, e.pos - 50)
        end = min(len(text), e.pos + 50)
        print(f"Context: ... {text[start:end]} ...")
        raise e

    print(f"Successfully loaded {len(catalog_data)} items from catalog.")

    # 2. Filtering Pre-packaged Job Solutions and processing items
    cleaned_catalog = []
    excluded_names = []
    
    seq_id = 0
    for item in catalog_data:
        name = item.get("name", "").strip()
        
        # Heuristic: treat as Pre-packaged Job Solution if name ends with "Solution" (case-insensitive)
        if name.lower().endswith("solution"):
            excluded_names.append(name)
            continue
        
        # Get keys and map to letter codes
        keys = item.get("keys", [])
        test_type_letters = []
        for k in keys:
            if k in CATEGORY_MAP:
                test_type_letters.append(CATEGORY_MAP[k])
            else:
                print(f"Warning: Unrecognized category key '{k}' for item '{name}'")
        
        test_type_str = ", ".join(test_type_letters)
        
        cleaned_item = {
            "id": seq_id,
            "entity_id": item.get("entity_id"),
            "name": name,
            "test_type": test_type_str,
            "url": item.get("link", ""),
            "description": item.get("description", ""),
            "duration": item.get("duration", ""),
            "languages": item.get("languages", []),
            "job_levels": item.get("job_levels", []),
            "keys": keys
        }
        cleaned_catalog.append(cleaned_item)
        seq_id += 1

    print("\n--- Excluded Pre-packaged Job Solutions ---")
    for esc in sorted(excluded_names):
        print(f"Excluded: {esc}")
    print(f"Excluded {len(excluded_names)} items. Retained {len(cleaned_catalog)} items as Individual Test Solutions.\n")

    # 3. Formulate texts to embed
    texts_to_embed = []
    for item in cleaned_catalog:
        name = item["name"]
        desc = item["description"] or ""
        job_levels = ", ".join(item["job_levels"]) if item["job_levels"] else ""
        
        embed_text = f"Name: {name}\nDescription: {desc}"
        if job_levels:
            embed_text += f"\nJob Levels: {job_levels}"
        texts_to_embed.append(embed_text)

    # 4. Embedding using OpenRouter
    api_key = os.environ.get("OPENROUTER_API_KEY", settings.OPENROUTER_API_KEY)
    if not api_key or api_key == "dummy":
        print("Warning: OPENROUTER_API_KEY environment variable not set. Using dummy/mock key.")
        api_key = "dummy"

    client = OpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=api_key
    )

    print(f"Generating embeddings using OpenRouter ({settings.EMBEDDING_MODEL}, dimensions={settings.EMBEDDING_DIMENSIONS})...")
    
    all_embeddings = []
    batch_size = 32
    total_items = len(texts_to_embed)
    
    for i in range(0, total_items, batch_size):
        batch = texts_to_embed[i : i + batch_size]
        print(f"Embedding batch {i // batch_size + 1} / {int(np.ceil(total_items / batch_size))} ({len(batch)} items)...")
        
        retries = 5
        delay = 1.0
        success = False
        while retries > 0:
            try:
                response = client.embeddings.create(
                    model=settings.EMBEDDING_MODEL,
                    input=batch,
                    dimensions=settings.EMBEDDING_DIMENSIONS
                )
                embeddings = [x.embedding for x in response.data]
                all_embeddings.extend(embeddings)
                success = True
                break
            except Exception as e:
                print(f"API Error: {e}. Retrying in {delay} seconds...")
                retries -= 1
                time.sleep(delay)
                delay *= 2
        
        if not success:
            print(f"Failed to embed batch starting at index {i} after multiple retries. Inserting zero vectors as fallback.")
            all_embeddings.extend([[0.0] * settings.EMBEDDING_DIMENSIONS] * len(batch))

    # 5. Build and save FAISS index
    print("Building FAISS index...")
    embeddings_np = np.array(all_embeddings, dtype=np.float32)
    ids_np = np.array([item["id"] for item in cleaned_catalog], dtype=np.int64)
    
    quantizer = faiss.IndexFlatL2(settings.EMBEDDING_DIMENSIONS)
    index = faiss.IndexIDMap(quantizer)
    index.add_with_ids(embeddings_np, ids_np)
    
    print(f"FAISS index size: {index.ntotal} vectors.")
    
    # Save files
    print("Saving files...")
    faiss.write_index(index, settings.INDEX_FILE)
    
    with open(settings.CATALOG_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_catalog, f, indent=2, ensure_ascii=False)
        
    print(f"Data preparation complete! Generated {settings.INDEX_FILE} and {settings.CATALOG_FILE}.")

if __name__ == "__main__":
    main()
