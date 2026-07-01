import os
from openai import OpenAI
from shl_recommender.core.config import settings

def load_dotenv():
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY", settings.OPENROUTER_API_KEY)
client = OpenAI(
    base_url=settings.OPENROUTER_BASE_URL,
    api_key=api_key
)

try:
    response = client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=["Hello world"],
        dimensions=settings.EMBEDDING_DIMENSIONS,
        encoding_format="float"
    )
    print("Success with both!")
    print(f"Embedding length: {len(response.data[0].embedding)}")
except Exception as e:
    print("Failed with both:", e)
