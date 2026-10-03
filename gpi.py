import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("=== Available Models for Content Generation ===")
for model in client.models.list():
    # Filter for models that support text generation
    if "generateContent" in (model.supported_actions or []):
        print(model.name)